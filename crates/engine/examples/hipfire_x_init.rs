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
                } else {
                    println!("  FAIL — {errors}/{m} matvec mismatches");
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
