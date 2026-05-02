//! GPU parity test for `triattn_score_bf16` vs `triattn_score_asym3`.
//!
//! Both kernels do the same Givens + RoPE + score arithmetic; they only
//! differ in how they get the `v[i]` per-thread vector. asym3 reads the
//! asym3 K cache (cnorm + packed 3-bit indices) and dequants inline.
//! bf16 reads pre-dequanted bf16 values directly. This test:
//!
//!   1. Builds an asym3 K cache via the production fused writer.
//!   2. Runs `triattn_score_asym3` -> scores_asym3 (reference path).
//!   3. CPU-side dequants the asym3 K cache to a bf16 K cache that
//!      mirrors the kernel's `v[i]` values (cnorm * TURBO_C3_256[idx]
//!      truncated to bf16 RAZ to match the AIE-2P-shape output of
//!      NpuRuntime::asym3_dequant_layer).
//!   4. Runs `triattn_score_bf16` -> scores_bf16.
//!   5. Compares scores_asym3 vs scores_bf16 for tight Pearson r and
//!      bounded max relative delta.
//!
//! Discrepancy expected from the bf16 truncation step (~2 bf16 ULP per
//! `v[i]`); the score is a sum-product over 32*4 = 128 such values per
//! (head, pos), so the score's relative delta is bounded by O(1e-3).

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::llama::KvCache;
    use engine::triattn::{self, BandCenter, TriAttnCenters};
    use rdna_compute::{DType, Gpu};

    // Same shape as triattn_gpu_parity_asym3.
    let n_heads = 16usize;
    let n_kv_heads = 4usize;
    let head_dim = 256usize;
    let _kv_group = n_heads / n_kv_heads;
    let n_bands = head_dim / 2;
    let rope_theta = 10_000_000.0f32;
    let partial_rotary_factor = 1.0f32;
    let n_rot = (head_dim as f32 * partial_rotary_factor) as usize;
    let seq_len = 64usize;
    let p_q = (seq_len - 1) as f32;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*seed >> 40) as u32;
        let uniform = bits as f32 / (1u32 << 24) as f32;
        uniform * 2.0 - 1.0
    }
    fn f32_to_bf16_rtz(x: f32) -> u16 {
        let xb = x.to_bits();
        if (xb & 0x7fff_ffff) > 0x7f80_0000 {
            return ((xb >> 16) | 0x0040) as u16;
        }
        (xb >> 16) as u16
    }
    fn f32_to_bf16_rne(x: f32) -> u16 {
        let xb = x.to_bits();
        if (xb & 0x7fff_ffff) > 0x7f80_0000 {
            return ((xb >> 16) | 0x0040) as u16;
        }
        let lsb = (xb >> 16) & 1;
        let bias = 0x7fff + lsb;
        ((xb.wrapping_add(bias)) >> 16) as u16
    }
    fn f32_to_bf16_raz(x: f32) -> u16 {
        let xb = x.to_bits();
        if (xb & 0x7fff_ffff) > 0x7f80_0000 {
            return ((xb >> 16) | 0x0040) as u16;
        }
        let abs = xb & 0x7fff_ffff;
        let sign = xb & 0x8000_0000;
        let biased = abs.wrapping_add(0xffff);
        ((sign | (biased & 0x7fff_ffff)) >> 16) as u16
    }
    fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

    let mut centers = TriAttnCenters::new(1, n_heads, head_dim, rope_theta, partial_rotary_factor);
    let mut seed = 0xdeadbeefu64;
    for h in 0..n_heads {
        for f in 0..n_bands {
            centers.set(0, h, f, BandCenter {
                eq_re: 0.3 * lcg(&mut seed),
                eq_im: 0.3 * lcg(&mut seed),
                e_abs_q: 0.5 + 0.3 * lcg(&mut seed).abs(),
            });
        }
    }
    let mut centers_flat = Vec::with_capacity(n_heads * n_bands * 3);
    for h in 0..n_heads {
        for f in 0..n_bands {
            let c = centers.get(0, h, f);
            centers_flat.push(c.eq_re);
            centers_flat.push(c.eq_im);
            centers_flat.push(c.e_abs_q);
        }
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let kv = KvCache::new_gpu_asym3(&mut gpu, 1, n_kv_heads, head_dim, seq_len)
        .expect("asym3 kv cache");
    let cos_theta = kv.givens_cos.as_ref().expect("asym3 has cos table");
    let sin_theta = kv.givens_sin.as_ref().expect("asym3 has sin table");
    let k_cache = &kv.k_gpu[0];
    let v_cache = &kv.v_gpu[0];
    let pos_dev = gpu.hip.malloc(4).unwrap();
    let kv_dim = n_kv_heads * head_dim;

    for pos in 0..seq_len {
        let k_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let k_tmp = gpu.upload_f32(&k_row, &[kv_dim]).unwrap();
        let v_tmp = gpu.upload_f32(&v_row, &[kv_dim]).unwrap();
        let pos_bytes = (pos as i32).to_ne_bytes();
        gpu.hip.memcpy_htod(&pos_dev, &pos_bytes).unwrap();
        gpu.kv_cache_write_asym3_fused(
            k_cache, v_cache, &k_tmp, &v_tmp, &pos_dev,
            cos_theta, sin_theta, n_kv_heads, head_dim,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Path A: triattn_score_asym3 (reference).
    let scores_asym3 = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32).unwrap();
    let centers_dev = gpu.upload_f32(&centers_flat, &[n_heads * n_bands * 3]).unwrap();
    gpu.triattn_score_asym3(
        k_cache, &centers_dev, cos_theta, sin_theta, &scores_asym3,
        n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let asym3_vec = gpu.download_f32(&scores_asym3).unwrap();

    // Build the bf16 K cache mirroring NpuRuntime::asym3_dequant_layer
    // output: cnorm * codebook[idx], rounded RAZ to bf16.
    const LLOYD_C3_256: [f32; 8] = [
        -0.134860, -0.083320, -0.046469, -0.015176,
         0.015176,  0.046469,  0.083320,  0.134860,
    ];
    let k_cache_bytes = {
        let floats = gpu.download_f32(k_cache).unwrap();
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for v in &floats { bytes.extend_from_slice(&v.to_ne_bytes()); }
        bytes
    };
    let k_bytes_per_head = 4 + (head_dim * 3) / 8;
    let k_bytes_per_pos = n_kv_heads * k_bytes_per_head;
    let bf16_per_pos = n_kv_heads * head_dim;
    let mut bf16_k = vec![0u16; seq_len * bf16_per_pos];
    // AIE-2P-shape dequant model (matches NpuRuntime::asym3_dequant_layer):
    //   cnorm: f32 -> bf16 RTZ (kernel runtime (bfloat16)*float_ptr cast)
    //   codebook: f32 -> bf16 RNE (compile-time bfloat16 ctor)
    //   product: f32 mul of (cnorm_bf16 -> f32) * (cb_bf16 -> f32)
    //   output: f32 -> bf16 RAZ
    // See docs/plans/aie2p-bf16-mul-shape.md and
    // crates/hipx/src/bin/verify_asym3_dequant.rs::cpu_reference.
    let cb_bf16_f32: [f32; 8] = std::array::from_fn(|i|
        bf16_to_f32(f32_to_bf16_rne(LLOYD_C3_256[i]))
    );
    for pos in 0..seq_len {
        for h_kv in 0..n_kv_heads {
            let head_off = pos * k_bytes_per_pos + h_kv * k_bytes_per_head;
            let cnorm = f32::from_le_bytes([
                k_cache_bytes[head_off],
                k_cache_bytes[head_off + 1],
                k_cache_bytes[head_off + 2],
                k_cache_bytes[head_off + 3],
            ]);
            let cnorm_b = bf16_to_f32(f32_to_bf16_rtz(cnorm));
            for tid in 0..32usize {
                let base_off = head_off + 4 + tid * 3;
                let b0 = k_cache_bytes[base_off] as u32;
                let b1 = k_cache_bytes[base_off + 1] as u32;
                let b2 = k_cache_bytes[base_off + 2] as u32;
                let packed = b0 | (b1 << 8) | (b2 << 16);
                for i in 0..8 {
                    let idx = ((packed >> (i * 3)) & 7) as usize;
                    let v = cnorm_b * cb_bf16_f32[idx];
                    let dim = tid * 8 + i;
                    let dst = pos * bf16_per_pos + h_kv * head_dim + dim;
                    bf16_k[dst] = f32_to_bf16_raz(v);
                }
            }
        }
    }

    // Upload bf16 K as raw bytes; the kernel reads u16. rdna_compute
    // doesn't have DType::BF16, but DType::Raw is the right shape for
    // "byte buffer, no element interpretation."
    let bf16_k_bytes: Vec<u8> = bf16_k.iter()
        .flat_map(|v| v.to_le_bytes().to_vec())
        .collect();
    let bf16_k_dev = gpu.hip.malloc(bf16_k_bytes.len()).unwrap();
    gpu.hip.memcpy_htod(&bf16_k_dev, &bf16_k_bytes).unwrap();
    let bf16_k_tensor = rdna_compute::GpuTensor {
        buf: bf16_k_dev,
        shape: vec![bf16_k.len()],
        dtype: DType::Raw,
    };

    // Path B: triattn_score_bf16 with the dequanted-then-RAZ K.
    let scores_bf16 = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32).unwrap();
    gpu.triattn_score_bf16(
        &bf16_k_tensor, &centers_dev, cos_theta, sin_theta, &scores_bf16,
        n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let bf16_vec = gpu.download_f32(&scores_bf16).unwrap();

    // Compare.
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..asym3_vec.len() {
        let a = asym3_vec[i];
        let b = bf16_vec[i];
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let denom = a.abs().max(b.abs()).max(1e-6);
        let r = d / denom;
        if r > max_rel { max_rel = r; }
    }
    let r = triattn::pearson(&asym3_vec, &bf16_vec);
    eprintln!("triattn_score_bf16 vs triattn_score_asym3 over {} heads x {} positions = {} scores",
              n_heads, seq_len, n_heads * seq_len);
    eprintln!("  max |Δ|  = {max_abs:.2e}");
    eprintln!("  max rel  = {max_rel:.2e}");
    eprintln!("  Pearson r = {r:.6}");

    // bf16 truncation per element induces ~2 ULP * 128 elements per
    // score = O(1e-3) relative noise. Empirically observed:
    //   max |Δ|  ≈ 1e-1 (absolute)
    //   max rel  ≈ 5e-3
    //   Pearson  ≈ 0.99985
    // Tight enough to catch real correctness bugs (anything other
    // than bf16 truncation drives Pearson << 0.99 fast); loose enough
    // to accept the expected truncation envelope.
    assert!(r > 0.999, "score ranking correlation too low: {r}");
    assert!(max_rel < 1e-2, "max relative delta too high: {max_rel}");
    eprintln!("bf16 vs asym3 parity within tolerance");
}
