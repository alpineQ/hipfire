//! Concurrent NPU + iGPU dispatch demo — the architectural moat for
//! Strix Halo's dual-engine UMA. Shows that an NPU matmul running
//! through hipx can overlap with an iGPU memset running through the
//! HIP runtime; wall-clock for the overlapped pair is ~max(t_npu,
//! t_gpu) instead of t_npu + t_gpu.
//!
//! Run on hipx:
//!   cargo run --release -p engine --features deltanet,npu \
//!     --example hipfire_x_concurrent

#[cfg(feature = "npu")]
fn run() {
    use engine::npu::NpuRuntime;
    use hip_bridge::HipRuntime;
    use std::time::{Duration, Instant};

    println!("[concurrent] initializing NPU runtime...");
    let mut npu = match NpuRuntime::try_init() {
        Some(rt) => rt,
        None => {
            println!("[concurrent] no NPU detected; aborting");
            return;
        }
    };
    println!("[concurrent]   AIE: {:?}", npu.family());

    println!("[concurrent] loading HIP runtime...");
    let hip = match HipRuntime::load() {
        Ok(rt) => rt,
        Err(e) => {
            println!("[concurrent] HIP load failed: {e}");
            return;
        }
    };
    println!("[concurrent]   HIP loaded");

    // 1024^3 INT8 matmul: ~482 us / 4.5 TOp/s through hipx.
    let m = 1024usize;
    let mut a_i8 = vec![0i8; m * m];
    let mut b_i8 = vec![0i8; m * m];
    for i in 0..(m * m) {
        a_i8[i] = (i as i8) & 0x3;
        b_i8[i] = ((i + 1) as i8) & 0x3;
    }
    let mut c_i8 = vec![0i32; m * m];

    // First call to NPU primes the kernel (lazy init).
    println!("[concurrent] priming NPU matmul (lazy init)...");
    npu.matmul_i8_1024_4c(&a_i8, &b_i8, &mut c_i8)
        .expect("matmul prime");

    // iGPU work: a 256 MiB device memset. On gfx1151 / 256-bit GDDR
    // unified memory this is ~250-500 us depending on DPM state.
    let scratch_bytes = 256 * 1024 * 1024;
    let scratch = hip.malloc(scratch_bytes).expect("hip malloc");
    println!("[concurrent] iGPU scratch: {} MiB", scratch_bytes / 1024 / 1024);

    // Warm-up the iGPU side with a few memsets so DPM ramps up.
    for _ in 0..3 {
        hip.memset(&scratch, 0xAA, scratch_bytes).expect("hip memset");
        hip.device_synchronize().expect("hip sync");
    }

    // Bench A: serial NPU dispatch (no iGPU work)
    let n_iter = 30u32;
    let t = Instant::now();
    for _ in 0..n_iter {
        npu.matmul_i8_1024_4c(&a_i8, &b_i8, &mut c_i8)
            .expect("matmul A");
    }
    let serial_npu_us = t.elapsed().as_micros();
    println!(
        "[concurrent] A. NPU only ({n_iter}× 1024^3 i8 matmul): {serial_npu_us} us total ({} us/op)",
        serial_npu_us / n_iter as u128
    );

    // Bench B: serial iGPU memset (no NPU work)
    let t = Instant::now();
    for _ in 0..n_iter {
        hip.memset(&scratch, 0xBB, scratch_bytes).expect("hip memset B");
        hip.device_synchronize().expect("hip sync B");
    }
    let serial_gpu_us = t.elapsed().as_micros();
    println!(
        "[concurrent] B. iGPU only ({n_iter}× 256 MiB memset): {serial_gpu_us} us total ({} us/op)",
        serial_gpu_us / n_iter as u128
    );

    // Init the 1024^3 i8 zero-copy path. Pre-fill A/B once so each
    // dispatch is just submit + wait without a per-call memcpy.
    let _ = npu.matmul_i8_1024_4c_init().expect("mm 1024 init");
    {
        let abuf = npu.matmul_i8_1024_4c_a_buf().expect("a_buf");
        for r in 0..m {
            for k in 0..m {
                abuf[r * m + k] = (((r + k) as i8) & 0x3) as u8;
            }
        }
    }
    {
        let bbuf = npu.matmul_i8_1024_4c_b_buf().expect("b_buf");
        for k in 0..m {
            for c in 0..m {
                bbuf[k * m + c] = (((k + c) as i8) & 0x3) as u8;
            }
        }
    }
    let mut c_1024 = vec![0i32; m * m];

    // Bench C: NPU 1024^3 zero-copy alone (no iGPU work).
    let t = Instant::now();
    for _ in 0..n_iter {
        let seq = npu
            .matmul_i8_1024_4c_submit_zero_copy()
            .expect("npu zc submit");
        npu.matmul_i8_1024_4c_wait(seq, &mut c_1024).expect("npu zc wait");
    }
    let zc_npu_us = t.elapsed().as_micros();
    println!(
        "[concurrent] C. NPU 1024^3 zero-copy alone ({n_iter} iters): {zc_npu_us} us total ({} us/op)",
        zc_npu_us / n_iter as u128
    );

    // Bench D: zero-copy NPU 1024^3 + iGPU memset, fully overlapped.
    let t = Instant::now();
    for _ in 0..n_iter {
        let seq = npu
            .matmul_i8_1024_4c_submit_zero_copy()
            .expect("npu zc submit");
        hip.memset(&scratch, 0xCC, scratch_bytes).expect("hip memset D");
        hip.device_synchronize().expect("hip sync D");
        npu.matmul_i8_1024_4c_wait(seq, &mut c_1024).expect("npu wait D");
    }
    let concurrent_us = t.elapsed().as_micros();
    println!(
        "[concurrent] D. NPU 1024^3 zc + iGPU concurrent ({n_iter} iters): {concurrent_us} us total ({} us/op)",
        concurrent_us / n_iter as u128
    );

    // Bench E + F: smaller iGPU work (32 MiB memset, ~150 us) — better
    // matches the time-domain of typical per-layer iGPU dispatches in
    // an LLM forward pass, and demonstrates the "free overlap" case
    // where the iGPU work fits inside the NPU compute window.
    let small_bytes = 32 * 1024 * 1024;
    let small_scratch = hip.malloc(small_bytes).expect("hip malloc small");
    // Warm up
    for _ in 0..3 {
        hip.memset(&small_scratch, 0xDD, small_bytes).expect("memset wm");
        hip.device_synchronize().expect("sync wm");
    }
    let t = Instant::now();
    for _ in 0..n_iter {
        hip.memset(&small_scratch, 0xEE, small_bytes).expect("hip small");
        hip.device_synchronize().expect("hip small sync");
    }
    let small_gpu_us = t.elapsed().as_micros();
    println!(
        "[concurrent] E. iGPU only (30× 32 MiB memset): {small_gpu_us} us total ({} us/op)",
        small_gpu_us / n_iter as u128
    );

    let t = Instant::now();
    for _ in 0..n_iter {
        let seq = npu
            .matmul_i8_1024_4c_submit_zero_copy()
            .expect("npu zc submit F");
        hip.memset(&small_scratch, 0xFF, small_bytes).expect("hip small F");
        hip.device_synchronize().expect("hip small sync F");
        npu.matmul_i8_1024_4c_wait(seq, &mut c_1024).expect("npu wait F");
    }
    let small_concurrent_us = t.elapsed().as_micros();
    println!(
        "[concurrent] F. NPU 1024^3 zc + small iGPU concurrent: {small_concurrent_us} us total ({} us/op)",
        small_concurrent_us / n_iter as u128
    );

    println!();
    println!("[concurrent] Analysis (large iGPU — bandwidth contention regime):");
    let npu_zc_per = zc_npu_us / n_iter as u128;
    let gpu_per = serial_gpu_us / n_iter as u128;
    let conc_per = concurrent_us / n_iter as u128;
    let serial_sum = npu_zc_per + gpu_per;
    let saved = serial_sum.saturating_sub(conc_per);
    let pct = if serial_sum > 0 { 100 * saved / serial_sum } else { 0 };
    println!("  serial (NPU zc 1024^3 + 256 MiB iGPU): {npu_zc_per} + {gpu_per} = {serial_sum} us/op");
    println!("  concurrent (overlapped):                {conc_per} us/op");
    println!("  saved by overlap:                       {saved} us/op ({pct}% wall-clock)");

    println!();
    println!("[concurrent] Analysis (small iGPU — fits inside NPU window):");
    let small_gpu_per = small_gpu_us / n_iter as u128;
    let small_conc_per = small_concurrent_us / n_iter as u128;
    let small_serial = npu_zc_per + small_gpu_per;
    let small_saved = small_serial.saturating_sub(small_conc_per);
    let small_pct = if small_serial > 0 { 100 * small_saved / small_serial } else { 0 };
    println!("  serial (NPU zc 1024^3 + 32 MiB iGPU): {npu_zc_per} + {small_gpu_per} = {small_serial} us/op");
    println!("  concurrent (overlapped):              {small_conc_per} us/op");
    println!("  saved by overlap:                     {small_saved} us/op ({small_pct}% wall-clock)");
    let macs = 2.0 * (m as f64).powi(3);
    let tops = macs / (npu_zc_per as f64 / 1e6) / 1e12;
    println!("  NPU compute delivered (zero-copy):    {tops:.2} TOp/s INT8 ({} GMACs/op)",
             (macs / 2e9) as u64);

    let _ = scratch;
    let _ = small_scratch;
    let _ = c_1024;

    // Avoid unused warning when the bench section is short.
    let _ = Duration::from_millis(0);
}

#[cfg(not(feature = "npu"))]
fn run() {
    println!("[concurrent] not compiled in (build with --features npu)");
}

fn main() {
    run();
}
