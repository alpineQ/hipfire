//! `hipfire_x_asym3_dequant_layer` - stage 1.4 engine-API smoke for the
//! layer-batched dequant kernel.
//!
//! Calls `engine::npu::NpuRuntime::asym3_dequant_layer` with a known
//! input pattern: chunk i in 0..N_ITERS packs all 256 indices equal
//! to `i % 8` and uses cnorm = 1.0. Expected output for chunk i:
//! all 256 bf16 values equal `bf16(TURBO_C3_256[i % 8])` within the
//! AIE-2P-shape envelope.
//!
//! Validates the engine wrapper end-to-end: hwctx + bound CU + reused
//! BOs, multi-element BOs sized for N_ITERS, command packet shape.
//!
//! Build:
//!   cargo run -p engine --features npu --example hipfire_x_asym3_dequant_layer
//!
//! Companion to verify_asym3_dequant_layer.rs (raw hipx) and the 256
//! engine smoke test hipfire_x_asym3_dequant.rs.

#[cfg(feature = "npu")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use engine::npu::NpuRuntime;
    use hipx::kernels::{
        ASYM3_DEQUANT_LAYER_HEAD_DIM as HEAD_DIM,
        ASYM3_DEQUANT_LAYER_N_ITERS as N_ITERS,
        ASYM3_DEQUANT_LAYER_OUT_BYTES as OUT_BYTES,
        ASYM3_DEQUANT_LAYER_PACKED_BYTES as PACKED_BYTES,
    };

    const TURBO_C3_256: [f32; 8] = [
        -0.134860, -0.083320, -0.046469, -0.015176,
         0.015176,  0.046469,  0.083320,  0.134860,
    ];

    fn f32_to_bf16_rne(x: f32) -> u16 {
        let xb = x.to_bits();
        if (xb & 0x7fff_ffff) > 0x7f80_0000 {
            return ((xb >> 16) | 0x0040) as u16;
        }
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
    println!("[hipfire-x] N_ITERS={N_ITERS} HEAD_DIM={HEAD_DIM} \
              packed_bytes={PACKED_BYTES} out_bytes={OUT_BYTES}");

    // Build batched inputs: chunk i packs all-idx = (i % 8), cnorm = 1.0.
    let mut packed = vec![0u8; PACKED_BYTES];
    let cnorms = vec![1.0f32; N_ITERS];
    for i in 0..N_ITERS {
        let k = (i % 8) as u32;
        let mut word: u32 = 0;
        for j in 0..8 { word |= k << (j * 3); }
        let chunk_base = i * 96;
        for tid in 0..32usize {
            let base = chunk_base + tid * 3;
            packed[base] = (word & 0xff) as u8;
            packed[base + 1] = ((word >> 8) & 0xff) as u8;
            packed[base + 2] = ((word >> 16) & 0xff) as u8;
        }
    }

    let mut out = vec![0u8; OUT_BYTES];
    rt.asym3_dequant_layer(&packed, &cnorms, &mut out)?;

    // Verify chunk by chunk.
    let mut max_ulp_global: u32 = 0;
    let mut n_diff: usize = 0;
    let mut first_fail: Option<String> = None;
    for i in 0..N_ITERS {
        let k = i % 8;
        let expected_bf16 = f32_to_bf16_rne(TURBO_C3_256[k]);
        let chunk_base = i * HEAD_DIM * 2;
        for d in 0..HEAD_DIM {
            let lo = out[chunk_base + d * 2] as u16;
            let hi = out[chunk_base + d * 2 + 1] as u16;
            let observed = lo | (hi << 8);
            let u = ulp_distance(observed, expected_bf16);
            if observed != expected_bf16 { n_diff += 1; }
            if u > max_ulp_global { max_ulp_global = u; }
            if u > 4 && first_fail.is_none() {
                first_fail = Some(format!(
                    "chunk {i} dim {d}: observed 0x{observed:04x} ({:.7}) \
                     expected 0x{expected_bf16:04x} ({:.7}) ulp {u}",
                    bf16_to_f32(observed), TURBO_C3_256[k]
                ));
            }
        }
    }

    let total_elems = N_ITERS * HEAD_DIM;
    println!("[hipfire-x] {N_ITERS} chunks x {HEAD_DIM} dims = {total_elems} elements");
    println!("[hipfire-x] max_ulp = {max_ulp_global} (bound 4); n_diff = {n_diff}");
    if let Some(fail) = first_fail {
        return Err(format!("max ULP > bound 4: {fail}").into());
    }
    println!("\n=== engine-API asym3_dequant_layer: PASS (max_ulp={max_ulp_global} <= 4) ===");
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
