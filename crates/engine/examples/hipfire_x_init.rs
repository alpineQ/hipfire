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
}

#[cfg(not(feature = "npu"))]
fn run() {
    println!("[hipfire-x] not compiled in (build with --features npu)");
}

fn main() {
    run();
}
