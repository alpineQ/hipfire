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
    }
}

#[cfg(not(feature = "npu"))]
fn run() {
    println!("[hipfire-x] not compiled in (build with --features npu)");
}

fn main() {
    run();
}
