//! Smoke-test for engine-side hipfire-x integration.
//!
//! Run on a Strix Halo box (hipx):
//!   cargo run -p engine --features deltanet,npu --example hipfire_x_init
//!
//! On non-Strix-Halo systems it should print "no NPU detected" and exit 0
//! — engine code path stays correct without hardware.

#[cfg(feature = "npu")]
fn run() {
    use engine::npu::{route, NpuRuntime};
    use hipx::dispatch::OpClass;

    println!("[hipfire-x] probing NPU...");
    let npu = NpuRuntime::try_init();

    match &npu {
        Some(rt) => {
            println!("[hipfire-x] NPU detected:");
            println!("  family:        {:?}", rt.family());
            println!("  cols:          {}", rt.cols());
            println!("  TOPS (INT8):   {}", rt.tops_int8());
            let ops = rt.available_ops();
            println!("  available ops:");
            println!("    passthrough_4k: {}", ops.passthrough_4k);
            println!("    kv_dequant:     {}", ops.kv_dequant);
            println!("    int8_gemm:      {}", ops.int8_gemm);
        }
        None => println!("[hipfire-x] no NPU detected"),
    }

    println!("\n[hipfire-x] route() smoke test (engine-call-site):");
    for (label, op) in [
        ("KV codec (asym3)", OpClass::KvCodec),
        ("INT8 GEMM 9B prefill", OpClass::Int8Gemm { m: 5120, n: 256, k: 5120 }),
        ("INT8 GEMM tiny", OpClass::Int8Gemm { m: 64, n: 64, k: 64 }),
        ("Sampler 32K vocab", OpClass::Sampler { vocab: 32768 }),
        ("Sampler 128K vocab", OpClass::Sampler { vocab: 131072 }),
        ("Embedding sidecar", OpClass::EmbeddingSidecar),
        ("Vision encoder", OpClass::VisionEncoder),
        ("Other (default)", OpClass::Other),
    ] {
        let target = route(&npu, op);
        println!("  {:30}  -> {:?}", label, target);
    }

    // Drive a real CU dispatch through the engine API (not just hipx
    // standalone). Confirms the engine → hipx → firmware → kernel
    // chain works with a real input/output buffer.
    if let Some(mut rt) = npu {
        println!("\n[hipfire-x] engine-API NPU dispatch (passthrough_4k):");
        let mut input = [0u8; 4096];
        for i in 0..4096 {
            input[i] = (i & 0xFF) as u8;
        }
        match rt.passthrough_4k(&input) {
            Ok(output) => {
                let mut errors = 0;
                for i in 0..4096 {
                    if output[i] != input[i] {
                        errors += 1;
                    }
                }
                if errors == 0 {
                    println!(
                        "  PASS — 4096 bytes round-tripped through NPU; first 8 = {:02x?}",
                        &output[..8]
                    );
                } else {
                    println!("  FAIL — {errors}/4096 mismatches");
                }
            }
            Err(e) => println!("  FAIL: {e}"),
        }

        // Real GEMM-class kernel: 288×288 i16 → i32 matvec.
        println!("\n[hipfire-x] engine-API NPU dispatch (matvec_i16_288x288):");
        let m = 288usize;
        let k = 288usize;
        let mut a = vec![0i16; m * k];
        let mut b = vec![0i16; k];
        for r in 0..m {
            for kk in 0..k {
                a[r * k + kk] = ((r + kk) as i16) & 0x7;
            }
        }
        for kk in 0..k {
            b[kk] = (kk as i16) & 0x7;
        }
        let mut c = vec![0i32; m];
        let t0 = std::time::Instant::now();
        match rt.matvec_i16_288x288(&a, &b, &mut c) {
            Ok(()) => {
                let first = std::time::Instant::now();
                let warm_us = first.duration_since(t0).as_micros();
                // verify
                let mut errors = 0;
                for r in 0..m {
                    let mut want: i32 = 0;
                    for kk in 0..k {
                        want += a[r * k + kk] as i32 * b[kk] as i32;
                    }
                    if c[r] != want {
                        errors += 1;
                    }
                }
                if errors == 0 {
                    // bench 50 dispatches steady-state
                    let n = 50u32;
                    let t = std::time::Instant::now();
                    for _ in 0..n {
                        rt.matvec_i16_288x288(&a, &b, &mut c).expect("matvec");
                    }
                    let mean_us = t.elapsed().as_micros() / n as u128;
                    let macs = 2.0 * m as f64 * k as f64;
                    let gops = macs / (mean_us as f64 / 1e6) / 1e9;
                    println!(
                        "  PASS — first {warm_us} us; warm mean {mean_us} us → {gops:.2} GOp/s ({n} iters, M=K={m})"
                    );

                    // Split-dispatch: shows that submit returns fast so
                    // the engine can do iGPU work in parallel. Times
                    //   submit
                    //   <host work simulating iGPU dispatch overlap>
                    //   wait
                    // and reports the submit-to-wait gap explicitly.
                    let n_split = 20u32;
                    let mut total_submit_us: u128 = 0;
                    let mut total_overlap_us: u128 = 0;
                    let mut total_wait_us: u128 = 0;
                    for _ in 0..n_split {
                        let t0 = std::time::Instant::now();
                        let seq = rt
                            .matvec_i16_288x288_submit(&a, &b)
                            .expect("submit");
                        let t1 = std::time::Instant::now();
                        // simulated concurrent host/iGPU work — a memcpy
                        // of the same volume the iGPU would touch on a
                        // partial decode (1 MiB). Real engine integration
                        // would call into rdna-compute here.
                        let mut sink = vec![0u8; 1 << 20];
                        for byte in sink.iter_mut() {
                            *byte = 0xAB;
                        }
                        std::hint::black_box(&sink[0]);
                        let t2 = std::time::Instant::now();
                        rt.matvec_i16_288x288_wait(seq, &mut c).expect("wait");
                        let t3 = std::time::Instant::now();
                        total_submit_us += t1.duration_since(t0).as_micros();
                        total_overlap_us += t2.duration_since(t1).as_micros();
                        total_wait_us += t3.duration_since(t2).as_micros();
                    }
                    println!(
                        "  split-dispatch ({n_split} iters): submit~{} us, overlap~{} us, wait~{} us, total~{} us",
                        total_submit_us / n_split as u128,
                        total_overlap_us / n_split as u128,
                        total_wait_us / n_split as u128,
                        (total_submit_us + total_overlap_us + total_wait_us) / n_split as u128
                    );

                    // Zero-copy bench: caller writes directly into the
                    // pre-mapped BO regions. Skips the per-call A copy
                    // (M*K*2 = 165 KiB) and the per-call B copy (576 B).
                    let _ = rt.matvec_i16_288x288_init().expect("matvec init");
                    // Pre-fill A and B once via the zero-copy buffer.
                    {
                        let abuf = rt.matvec_i16_288x288_a_buf().expect("a_buf");
                        for i in 0..(m * k) {
                            let r = i / k;
                            let kk = i % k;
                            let v = ((r + kk) as i16) & 0x7;
                            abuf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    {
                        let bbuf = rt.matvec_i16_288x288_b_buf().expect("b_buf");
                        for kk in 0..k {
                            let v = (kk as i16) & 0x7;
                            bbuf[kk * 2..kk * 2 + 2].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    let n_zc = 50u32;
                    let t = std::time::Instant::now();
                    for _ in 0..n_zc {
                        let seq = rt
                            .matvec_i16_288x288_submit_zero_copy()
                            .expect("zc submit");
                        rt.matvec_i16_288x288_wait(seq, &mut c).expect("zc wait");
                    }
                    let mean_us = t.elapsed().as_micros() / n_zc as u128;
                    let macs = 2.0 * m as f64 * k as f64;
                    let gops = macs / (mean_us as f64 / 1e6) / 1e9;
                    println!(
                        "  zero-copy ({n_zc} iters): {mean_us} us mean → {gops:.2} GOp/s"
                    );

                    // Pipelined dispatch: queue N matvecs, then wait on
                    // the last one. Tells us if the firmware/scheduler
                    // pipelines or serializes dispatches.
                    let depths = [2u32, 4, 8];
                    for &depth in &depths {
                        let total_iters = 32u32;
                        let batches = total_iters / depth;
                        let t = std::time::Instant::now();
                        for _ in 0..batches {
                            let mut last_seq = 0u64;
                            for _ in 0..depth {
                                last_seq = rt
                                    .matvec_i16_288x288_submit_zero_copy()
                                    .expect("pipe submit");
                            }
                            rt.matvec_i16_288x288_wait(last_seq, &mut c)
                                .expect("pipe wait");
                        }
                        let total_us = t.elapsed().as_micros();
                        let per_op_us = total_us / total_iters as u128;
                        let gops_pipe = macs / (per_op_us as f64 / 1e6) / 1e9;
                        println!(
                            "  pipeline depth={depth}: {per_op_us} us/op → {gops_pipe:.2} GOp/s ({total_iters} ops)"
                        );
                    }
                } else {
                    println!("  FAIL — {errors}/{m} matvec mismatches");
                }

        // 4-core matmul (the press-headline kernel — 1+ TOp/s INT16).
        println!("\n[hipfire-x] engine-API NPU dispatch (matmul_i16_512_4c):");
        let mm = 512usize;
        let kk = 512usize;
        let nn = 512usize;
        let mut a_mm = vec![0i16; mm * kk];
        let mut b_mm = vec![0i16; kk * nn];
        for r in 0..mm {
            for k in 0..kk {
                a_mm[r * kk + k] = ((r + k) as i16) & 0x7;
            }
        }
        for k in 0..kk {
            for c in 0..nn {
                b_mm[k * nn + c] = ((k + c) as i16) & 0x7;
            }
        }
        let mut c_mm = vec![0i32; mm * nn];
        let t0 = std::time::Instant::now();
        match rt.matmul_i16_512_4c(&a_mm, &b_mm, &mut c_mm) {
            Ok(()) => {
                let warm_us = t0.elapsed().as_micros();
                // verify first row
                let mut errs = 0;
                for c in 0..16 {
                    let mut want: i32 = 0;
                    for k in 0..kk {
                        want += a_mm[k] as i32 * b_mm[k * nn + c] as i32;
                    }
                    if c_mm[c] != want { errs += 1; }
                }
                if errs == 0 {
                    let n = 30u32;
                    let t = std::time::Instant::now();
                    for _ in 0..n {
                        rt.matmul_i16_512_4c(&a_mm, &b_mm, &mut c_mm).expect("mm");
                    }
                    let mean_us = t.elapsed().as_micros() / n as u128;
                    let macs = 2.0 * mm as f64 * kk as f64 * nn as f64;
                    let tops = macs / (mean_us as f64 / 1e6) / 1e12;
                    println!(
                        "  PASS — first {warm_us} us; warm mean {mean_us} us → {tops:.2} TOp/s INT16 ({n} iters, {mm}^3)"
                    );
                } else {
                    println!("  FAIL — {errs}/16 first-row mismatches");
                }
            }
            Err(e) => println!("  FAIL: {e}"),
        }

        // INT8 matmul — the natural max-throughput kernel.
        println!("\n[hipfire-x] engine-API NPU dispatch (matmul_i8_512_4c):");
        let mut a_i8 = vec![0i8; mm * kk];
        let mut b_i8 = vec![0i8; kk * nn];
        for r in 0..mm {
            for k in 0..kk {
                a_i8[r * kk + k] = ((r + k) as i8) & 0x7;
            }
        }
        for k in 0..kk {
            for c in 0..nn {
                b_i8[k * nn + c] = ((k + c) as i8) & 0x7;
            }
        }
        let mut c_i8 = vec![0i32; mm * nn];
        let t0 = std::time::Instant::now();
        match rt.matmul_i8_512_4c(&a_i8, &b_i8, &mut c_i8) {
            Ok(()) => {
                let warm_us = t0.elapsed().as_micros();
                let mut errs = 0;
                for c in 0..16 {
                    let mut want: i32 = 0;
                    for k in 0..kk {
                        want += a_i8[k] as i32 * b_i8[k * nn + c] as i32;
                    }
                    if c_i8[c] != want { errs += 1; }
                }
                if errs == 0 {
                    let n = 50u32;
                    let t = std::time::Instant::now();
                    for _ in 0..n {
                        rt.matmul_i8_512_4c(&a_i8, &b_i8, &mut c_i8).expect("mm8");
                    }
                    let mean_us = t.elapsed().as_micros() / n as u128;
                    let macs = 2.0 * mm as f64 * kk as f64 * nn as f64;
                    let tops = macs / (mean_us as f64 / 1e6) / 1e12;
                    println!(
                        "  PASS — first {warm_us} us; warm mean {mean_us} us → {tops:.2} TOp/s INT8 ({n} iters, {mm}^3)"
                    );

                    // Zero-copy bench: write A/B once via the mapped BO,
                    // then submit→wait in a tight loop. Recovers the
                    // engine API overhead from the per-call A/B refresh.
                    let _ = rt.matmul_i8_512_4c_init().expect("mm8 init");
                    {
                        let abuf = rt.matmul_i8_512_4c_a_buf().expect("a_buf");
                        for r in 0..mm {
                            for k in 0..kk {
                                abuf[r * kk + k] = (((r + k) as i8) & 0x7) as u8;
                            }
                        }
                    }
                    {
                        let bbuf = rt.matmul_i8_512_4c_b_buf().expect("b_buf");
                        for k in 0..kk {
                            for c in 0..nn {
                                bbuf[k * nn + c] = (((k + c) as i8) & 0x7) as u8;
                            }
                        }
                    }
                    let n_zc = 50u32;
                    let t = std::time::Instant::now();
                    for _ in 0..n_zc {
                        let seq = rt
                            .matmul_i8_512_4c_submit_zero_copy()
                            .expect("zc submit");
                        rt.matmul_i8_512_4c_wait(seq, &mut c_i8).expect("zc wait");
                    }
                    let mean_us = t.elapsed().as_micros() / n_zc as u128;
                    let tops = macs / (mean_us as f64 / 1e6) / 1e12;
                    println!(
                        "  zero-copy ({n_zc} iters): {mean_us} us mean → {tops:.2} TOp/s INT8"
                    );
                } else {
                    println!("  FAIL — {errs}/16 first-row mismatches");
                }
            }
            Err(e) => println!("  FAIL: {e}"),
        }
            }
            Err(e) => println!("  FAIL: {e}"),
        }
    }
}

#[cfg(not(feature = "npu"))]
fn run() {
    println!("[hipfire-x] not compiled in (build with --features npu)");
}

fn main() {
    run();
}
