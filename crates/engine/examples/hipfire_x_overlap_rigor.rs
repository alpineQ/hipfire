//! Rigorous overlap bench. Replaces the toy memset proxy in
//! hipfire_x_concurrent with an actual fp16 GEMM via rocBLAS, and
//! reports median + p95 across 1000 iterations × 3 trials so the
//! "free overlap" claim has measurable variance behind it.
//!
//! Compares wall-clock per op of:
//!   * NPU 1024^3 i8 zero-copy alone
//!   * iGPU fp16 GEMM (rocBLAS, M=N=K matched to NPU dispatch time) alone
//!   * NPU + iGPU concurrent
//!
//! Trial = 1000 iters; bench = 3 trials. Reports per-trial
//! median/p95/min and inter-trial coefficient-of-variation on the
//! medians, so we can decide whether tiny "savings" sit inside the
//! noise band.
//!
//! Run on hipx:
//!   cargo run --release --features deltanet,npu \
//!     --example hipfire_x_overlap_rigor

#[cfg(feature = "npu")]
fn run() {
    use engine::npu::NpuRuntime;
    use hip_bridge::{HipRuntime, Rocblas, RocblasDatatype, RocblasOperation};
    use std::time::Instant;

    println!("[overlap] init NPU + HIP + rocBLAS...");
    let mut npu = match NpuRuntime::try_init() {
        Some(rt) => rt,
        None => { println!("no NPU"); return; }
    };
    let hip = match HipRuntime::load() {
        Ok(rt) => rt,
        Err(e) => { println!("HIP load: {e}"); return; }
    };
    let rocblas = match Rocblas::load() {
        Ok(rb) => rb,
        Err(e) => { println!("rocBLAS load: {e}"); return; }
    };

    // iGPU GEMM shape — M=N=K=2048, fp16, runs ~1 ms on gfx1151.
    // Real per-layer decode GEMV is bandwidth-bound on a single
    // token, but a square fp16 GEMM at this size has a similar
    // mix of read+compute and total time as per-layer attention
    // (without the model dependency). 17 GFLOPs of work.
    let mn = 2048i32;
    let kk = 2048i32;
    let n_elems = (mn * kk) as usize;
    let n_bytes_fp16 = n_elems * 2;

    let a_dev = hip.malloc(n_bytes_fp16).expect("A alloc");
    let b_dev = hip.malloc(n_bytes_fp16).expect("B alloc");
    let c_dev = hip.malloc((mn * mn) as usize * 2).expect("C alloc");

    // Fill with non-zero so the kernel can't shortcut. Bytes 0xAA
    // interpreted as fp16 are mostly normal-range values.
    hip.memset(&a_dev, 0xAA, n_bytes_fp16).expect("memset A");
    hip.memset(&b_dev, 0xAA, n_bytes_fp16).expect("memset B");
    hip.device_synchronize().expect("sync init");

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let do_gemm = || -> bool {
        unsafe {
            rocblas.gemm_ex(
                RocblasOperation::None, RocblasOperation::None,
                mn, mn, kk,
                &alpha as *const f32 as *const _,
                a_dev.as_ptr() as *const _, RocblasDatatype::F16, mn,
                b_dev.as_ptr() as *const _, RocblasDatatype::F16, kk,
                &beta as *const f32 as *const _,
                c_dev.as_ptr() as *const _, RocblasDatatype::F16, mn,
                c_dev.as_ptr() as *mut _,   RocblasDatatype::F16, mn,
                RocblasDatatype::F32, // accumulate fp32
            ).is_ok()
        }
    };

    // Warm-up: pin DPM, prime kernel JIT.
    for _ in 0..50 {
        do_gemm();
        hip.device_synchronize().expect("warm sync");
    }
    println!("[overlap] iGPU rocBLAS fp16 GEMM ready ({}^3, ~17 GFLOPs)", mn);

    // Init NPU 1024^3 i8 zero-copy.
    let m_npu = 1024usize;
    let _ = npu.matmul_i8_1024_4c_init().expect("npu init");
    {
        let buf = npu.matmul_i8_1024_4c_a_buf().expect("a buf");
        for r in 0..m_npu {
            for k in 0..m_npu {
                buf[r * m_npu + k] = (((r + k) as i8) & 0x3) as u8;
            }
        }
    }
    {
        let buf = npu.matmul_i8_1024_4c_b_buf().expect("b buf");
        for k in 0..m_npu {
            for c in 0..m_npu {
                buf[k * m_npu + c] = (((k + c) as i8) & 0x3) as u8;
            }
        }
    }
    npu.matmul_i8_1024_4c_sync_inputs().expect("npu sync");

    // NPU warm-up.
    for _ in 0..20 {
        let seq = npu.matmul_i8_1024_4c_submit_zero_copy().expect("npu warm submit");
        npu.matmul_i8_1024_4c_wait_no_copy(seq).expect("npu warm wait");
    }
    println!("[overlap] NPU 1024^3 i8 zero-copy ready");

    // Bench harness. Each trial = N iters; capture per-iter µs.
    fn pct(samples: &[u128], pct: f64) -> u128 {
        let mut s = samples.to_vec();
        s.sort_unstable();
        let idx = ((s.len() as f64 - 1.0) * pct).round() as usize;
        s[idx]
    }
    fn stats(samples: &[u128]) -> (u128, u128, u128, u128, u128) {
        let mut s = samples.to_vec();
        s.sort_unstable();
        let median = s[s.len() / 2];
        let p95 = pct(&s, 0.95);
        let p99 = pct(&s, 0.99);
        let min = *s.first().unwrap();
        let max = *s.last().unwrap();
        (median, p95, p99, min, max)
    }
    fn run_trial(n: usize, mut body: impl FnMut() -> u128) -> Vec<u128> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(body());
        }
        out
    }

    let n_iters: usize = 1000;
    let n_trials: usize = 3;

    println!("\n[overlap] running {n_trials} trials × {n_iters} iters each...\n");

    // Mode A: NPU only (1024^3 i8 zero-copy, no-copy wait).
    println!("Mode A: NPU only");
    let mut a_medians = Vec::new();
    for trial in 0..n_trials {
        let mut c_throw = vec![0i32; m_npu * m_npu];
        let _ = c_throw;
        let samples = run_trial(n_iters, || {
            let t = Instant::now();
            let seq = npu.matmul_i8_1024_4c_submit_zero_copy().expect("submit A");
            npu.matmul_i8_1024_4c_wait_no_copy(seq).expect("wait A");
            t.elapsed().as_micros()
        });
        let (median, p95, p99, min, max) = stats(&samples);
        a_medians.push(median);
        println!("  trial {trial}: median={median} p95={p95} p99={p99} min={min} max={max} us");
    }

    // Mode B: iGPU only (rocBLAS fp16 GEMM 2048^3).
    println!("\nMode B: iGPU only (rocBLAS fp16 GEMM 2048^3)");
    let mut b_medians = Vec::new();
    for trial in 0..n_trials {
        let samples = run_trial(n_iters, || {
            let t = Instant::now();
            do_gemm();
            hip.device_synchronize().expect("hip sync B");
            t.elapsed().as_micros()
        });
        let (median, p95, p99, min, max) = stats(&samples);
        b_medians.push(median);
        println!("  trial {trial}: median={median} p95={p95} p99={p99} min={min} max={max} us");
    }

    // Mode C: NPU + iGPU concurrent.
    println!("\nMode C: NPU 1024^3 + iGPU 2048^3 concurrent");
    let mut c_medians = Vec::new();
    for trial in 0..n_trials {
        let samples = run_trial(n_iters, || {
            let t = Instant::now();
            let seq = npu.matmul_i8_1024_4c_submit_zero_copy().expect("submit C");
            do_gemm();
            hip.device_synchronize().expect("hip sync C");
            npu.matmul_i8_1024_4c_wait_no_copy(seq).expect("wait C");
            t.elapsed().as_micros()
        });
        let (median, p95, p99, min, max) = stats(&samples);
        c_medians.push(median);
        println!("  trial {trial}: median={median} p95={p95} p99={p99} min={min} max={max} us");
    }

    // Inter-trial CV (coefficient of variation on the medians).
    fn cv(xs: &[u128]) -> f64 {
        let n = xs.len() as f64;
        let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = xs.iter().map(|&x| {
            let d = x as f64 - mean; d * d
        }).sum::<f64>() / n;
        var.sqrt() / mean
    }

    println!("\n[overlap] Cross-trial summary:");
    println!("  A NPU only median per trial:      {:?}", a_medians);
    println!("  B iGPU only median per trial:     {:?}", b_medians);
    println!("  C concurrent median per trial:    {:?}", c_medians);
    println!("  CV(medians) — A: {:.3}%  B: {:.3}%  C: {:.3}%",
             cv(&a_medians) * 100.0, cv(&b_medians) * 100.0, cv(&c_medians) * 100.0);

    // The actual claim. Compute "concurrent vs serial sum" using
    // PER-TRIAL medians (not an average), and report the saved-µs
    // distribution across the 3 trials.
    println!("\n[overlap] Saved-by-overlap (per trial, median-µs basis):");
    let mut saved = Vec::new();
    for t in 0..n_trials {
        let serial = a_medians[t] + b_medians[t];
        let conc = c_medians[t];
        let s = serial as i128 - conc as i128;
        let pct = 100.0 * s as f64 / serial as f64;
        println!("  trial {t}: serial={serial} concurrent={conc} saved={s} us ({pct:.1}%)");
        saved.push(s);
    }
    let saved_min = saved.iter().min().unwrap();
    let saved_max = saved.iter().max().unwrap();
    println!("  saved range across 3 trials:      {saved_min}..{saved_max} us");
    println!("  → if saved_min > 0 and CV(C) is small (<5%),");
    println!("    the overlap is real; otherwise it's noise.");

    let _ = a_dev;
    let _ = b_dev;
    let _ = c_dev;
}

#[cfg(not(feature = "npu"))]
fn run() {
    println!("[overlap] not compiled in (build with --features npu)");
}

fn main() {
    run();
}
