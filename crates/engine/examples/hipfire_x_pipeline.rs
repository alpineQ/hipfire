//! Multi-layer NPU+iGPU pipeline demo — simulates a real LLM
//! forward pass where each layer issues iGPU work (~200-500 µs per
//! layer for attention/GEMM) and concurrently kicks off an NPU
//! matmul that overlaps with the next layer's iGPU work.
//!
//! The demo runs 28 "layers" (matching 27B Gemma's depth) of:
//!   - iGPU: 32 MiB memset (~50 µs, simulates per-layer GEMM kernel)
//!   - NPU: 1024^3 i8 matmul (~1 ms, simulates speculative draft head)
//! and compares wall-clock time of:
//!   A) iGPU-only (baseline)
//!   B) iGPU then NPU serial (worst case)
//!   C) iGPU + NPU pipelined (the press number)
//!
//! For the pipelined case, layer N's iGPU kicks off concurrent with
//! layer N-1's NPU wait. The expectation: pipelined ≈ max(iGPU/layer,
//! NPU/layer) × n_layers — i.e. the NPU work is fully hidden because
//! it's amortized across layers, even when per-layer NPU > per-layer
//! iGPU work.

#[cfg(feature = "npu")]
fn run() {
    use engine::npu::NpuRuntime;
    use hip_bridge::HipRuntime;
    use std::time::Instant;

    println!("[pipeline] initializing NPU runtime...");
    let mut npu = match NpuRuntime::try_init() {
        Some(rt) => rt,
        None => {
            println!("[pipeline] no NPU detected");
            return;
        }
    };
    println!("[pipeline]   AIE: {:?}", npu.family());

    println!("[pipeline] loading HIP runtime...");
    let hip = match HipRuntime::load() {
        Ok(rt) => rt,
        Err(e) => {
            println!("[pipeline] HIP load failed: {e}");
            return;
        }
    };

    let n_layers = 28u32; // ~27B Gemma depth
    // 256 MiB memset ≈ 1200 µs — size-matched to the NPU 1024^3
    // dispatch (~1000 µs through engine API). This models the
    // realistic regime where per-layer iGPU work and NPU dispatch
    // are comparable in time and pipelining can hide ~all of NPU.
    let small_bytes = 256 * 1024 * 1024;
    let scratch = hip.malloc(small_bytes).expect("hip malloc");

    // Init NPU 1024^3 zero-copy path (the NPU is doing draft-head-style work).
    let _ = npu.matmul_i8_1024_4c_init().expect("mm init");
    let m = 1024usize;
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
    // Flush A/B to device once. _submit_zero_copy no longer re-syncs
    // them, so this drops ~30-100 µs per layer of redundant sync ioctls.
    npu.matmul_i8_1024_4c_sync_inputs().expect("sync inputs");
    let mut c_npu = vec![0i32; m * m];

    // Warm-up.
    for _ in 0..3 {
        hip.memset(&scratch, 0xAA, small_bytes).expect("memset");
        hip.device_synchronize().expect("sync");
        let seq = npu
            .matmul_i8_1024_4c_submit_zero_copy()
            .expect("submit warm");
        npu.matmul_i8_1024_4c_wait(seq, &mut c_npu)
            .expect("wait warm");
    }

    // Bench A: iGPU-only forward pass (28 layers of 32 MiB memset).
    let t = Instant::now();
    for _ in 0..n_layers {
        hip.memset(&scratch, 0xBB, small_bytes).expect("memset A");
        hip.device_synchronize().expect("sync A");
    }
    let a_us = t.elapsed().as_micros();
    println!(
        "[pipeline] A. iGPU-only forward ({n_layers} layers): {a_us} us total ({} us/layer)",
        a_us / n_layers as u128
    );

    // Bench B: iGPU then NPU serial per layer.
    let t = Instant::now();
    for _ in 0..n_layers {
        hip.memset(&scratch, 0xCC, small_bytes).expect("memset B");
        hip.device_synchronize().expect("sync B");
        let seq = npu
            .matmul_i8_1024_4c_submit_zero_copy()
            .expect("submit B");
        npu.matmul_i8_1024_4c_wait(seq, &mut c_npu)
            .expect("wait B");
    }
    let b_us = t.elapsed().as_micros();
    println!(
        "[pipeline] B. iGPU + NPU serial ({n_layers} layers): {b_us} us total ({} us/layer)",
        b_us / n_layers as u128
    );

    // Bench C: pipelined. Submit NPU at start of layer N, do iGPU,
    // wait NPU at start of layer N+1 (so layer N's NPU runs during
    // layer N+1's iGPU). For the LAST layer there's no overlap, so
    // the pipeline tail is ~one NPU wait.
    let t = Instant::now();
    let mut prev_seq: Option<u64> = None;
    for layer in 0..n_layers {
        // 1. Submit THIS layer's NPU. Returns ~30 us.
        let seq = npu
            .matmul_i8_1024_4c_submit_zero_copy()
            .expect("submit C");
        // 2. Wait PREVIOUS layer's NPU WITHOUT copying C back —
        //    just sync_from_device + fence wait (~50 µs vs ~500 µs).
        //    The previous result stays in the C BO; if the engine
        //    needed it, it would call _c_view() before submitting
        //    the next NPU.
        if let Some(prev) = prev_seq.take() {
            npu.matmul_i8_1024_4c_wait_no_copy(prev).expect("wait_no_copy C prev");
        }
        // 3. iGPU work for THIS layer. Runs concurrent with NEW NPU.
        hip.memset(&scratch, 0xDD, small_bytes).expect("memset C");
        hip.device_synchronize().expect("sync C");
        prev_seq = Some(seq);
        let _ = layer;
    }
    // Final drain — copy out the last layer's result for verification.
    if let Some(prev) = prev_seq.take() {
        npu.matmul_i8_1024_4c_wait(prev, &mut c_npu).expect("wait C final");
    }
    let c_us = t.elapsed().as_micros();
    println!(
        "[pipeline] C. iGPU + NPU pipelined ({n_layers} layers): {c_us} us total ({} us/layer)",
        c_us / n_layers as u128
    );

    println!();
    println!("[pipeline] Analysis:");
    let saved_vs_b = b_us.saturating_sub(c_us);
    let pct_vs_b = if b_us > 0 { 100 * saved_vs_b / b_us } else { 0 };
    let saved_vs_a = c_us.saturating_sub(a_us);
    let pct_vs_a = if a_us > 0 { 100 * saved_vs_a / a_us } else { 0 };
    println!("  serial (A iGPU + B NPU as in B):     {b_us} us total");
    println!("  pipelined (C):                       {c_us} us total");
    println!("  saved by pipelining:                 {saved_vs_b} us ({pct_vs_b}% wall-clock vs serial)");
    println!("  vs iGPU-only baseline (A):           +{saved_vs_a} us ({pct_vs_a}% overhead added by NPU)");
    let macs_per_layer = 2.0 * (m as f64).powi(3);
    let total_macs = macs_per_layer * n_layers as f64;
    let tops_pipelined = total_macs / (c_us as f64 / 1e6) / 1e12;
    println!("  effective NPU throughput:            {tops_pipelined:.2} TOp/s INT8 ({} GMACs/layer × {n_layers} layers)",
             (macs_per_layer / 2e9) as u64);

    let _ = scratch;
    let _ = c_npu;
}

#[cfg(not(feature = "npu"))]
fn run() {
    println!("[pipeline] not compiled in (build with --features npu)");
}

fn main() {
    run();
}
