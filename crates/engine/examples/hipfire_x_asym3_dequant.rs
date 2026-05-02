//! `hipfire_x_asym3_dequant` — stage 1.3a engine-API integration smoke.
//!
//! Calls `engine::npu::NpuRuntime::asym3_dequant_256` with a known
//! input pattern (cnorm = 1.0, packed all-idx = k for each k in 0..8)
//! and verifies the output matches the engine TURBO_C3_256 codebook
//! within the AIE-2P-shape envelope (max 4 bf16 ULP per element,
//! see docs/plans/aie2p-bf16-mul-shape.md).
//!
//! Build:
//!   cargo run -p engine --features npu --example hipfire_x_asym3_dequant
//!
//! This is the engine-API mirror of the more thorough
//! `crates/hipx/src/bin/verify_asym3_dequant.rs` 100-seed verifier.
//! The verifier dispatches via raw hipx; this example dispatches via
//! `NpuRuntime::asym3_dequant_256` to confirm the new wrapper hooks
//! up correctly.

#[cfg(feature = "npu")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use engine::npu::NpuRuntime;

    const TURBO_C3_256: [f32; 8] = [
        -0.134860, -0.083320, -0.046469, -0.015176,
         0.015176,  0.046469,  0.083320,  0.134860,
    ];

    fn f32_to_bf16_rtz(x: f32) -> u16 { (x.to_bits() >> 16) as u16 }
    /// RNE matches the kernel's `bfloat16(f32_lit)` compile-time
    /// constructor used to store codebook entries.
    fn f32_to_bf16_rne(x: f32) -> u16 {
        let xb = x.to_bits();
        let lsb = (xb >> 16) & 1;
        let bias = 0x7fff + lsb;
        ((xb.wrapping_add(bias)) >> 16) as u16
    }
    fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }
    fn ulp_distance(a: u16, b: u16) -> u32 {
        let sa = a & 0x8000;
        let sb = b & 0x8000;
        if sa == sb {
            ((a & 0x7fff) as i32 - (b & 0x7fff) as i32).unsigned_abs()
        } else {
            (a & 0x7fff) as u32 + (b & 0x7fff) as u32
        }
    }

    let mut rt = NpuRuntime::try_init().ok_or("no NPU on this system")?;
    println!("[hipfire-x] NPU init OK; cols={} TOPS_INT8={}",
             rt.cols(), rt.tops_int8());

    // Build an all-same-idx pattern for each k in 0..8.
    let mut max_ulp_global: u32 = 0;
    for k in 0..8u8 {
        let mut packed = [0u8; 96];
        let mut word: u32 = 0;
        for i in 0..8 { word |= (k as u32) << (i * 3); }
        for tid in 0..32usize {
            let base = tid * 3;
            packed[base] = (word & 0xff) as u8;
            packed[base + 1] = ((word >> 8) & 0xff) as u8;
            packed[base + 2] = ((word >> 16) & 0xff) as u8;
        }
        let cnorm = 1.0f32;
        let mut out = [0u8; 512];

        rt.asym3_dequant_256(&packed, cnorm, &mut out)?;

        let expected_bf16 = f32_to_bf16_rne(TURBO_C3_256[k as usize]);
        let mut max_ulp_k: u32 = 0;
        for d in 0..256 {
            let observed = (out[d * 2] as u16) | ((out[d * 2 + 1] as u16) << 8);
            let u = ulp_distance(observed, expected_bf16);
            if u > max_ulp_k { max_ulp_k = u; }
        }
        if max_ulp_k > max_ulp_global { max_ulp_global = max_ulp_k; }
        let observed_d0 = (out[0] as u16) | ((out[1] as u16) << 8);
        println!(
            "[hipfire-x] k={k}: dim 0 = 0x{observed_d0:04x} ({:.7}); expected 0x{expected_bf16:04x} ({:.7}); max_ulp_per_dim = {max_ulp_k}",
            bf16_to_f32(observed_d0), TURBO_C3_256[k as usize]
        );
    }

    if max_ulp_global > 4 {
        return Err(format!("max ULP {max_ulp_global} > bound 4 (AIE-2P-shape envelope)").into());
    }
    println!("\n=== engine-API asym3_dequant_256: PASS (max_ulp={max_ulp_global} <= 4) ===");
    Ok(())
}

#[cfg(not(feature = "npu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("Build with --features npu to exercise this example.");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
