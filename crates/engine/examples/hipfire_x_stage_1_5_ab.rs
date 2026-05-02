//! `hipfire_x_stage_1_5_ab` - actual A/B bench for stage 1.5 lift gate.
//!
//! Per the npu-roadmap contract, stage 1.5 needs measured numbers from
//! HIPFIRE_NPU_DEQUANT=0 (iGPU baseline) vs HIPFIRE_NPU_DEQUANT=1 (NPU
//! path) on the same workload to decide whether to flip the default.
//! Earlier escalations were projection-based; this gives the real
//! number including memory transfer overhead.
//!
//! Workload: per-token triattn score over an asym3 K cache at the
//! shape that matches our fixed-N kernel (8 kv_heads x 128 positions
//! = 1024 (head, pos) chunks per layer). This is what one decode step
//! does in the engine's eviction / scoring path.
//!
//! Path A (iGPU baseline, HIPFIRE_NPU_DEQUANT=0):
//!   gpu.triattn_score_asym3(asym3_k, ...) -> scores
//!
//! Path B (NPU dequant + iGPU bf16 score, HIPFIRE_NPU_DEQUANT=1):
//!   1. Download asym3 K from iGPU to host
//!   2. Decode (cnorm, packed) -> (cnorms[1024], packed[98304B]) layout
//!   3. NpuRuntime::asym3_dequant_layer(packed, cnorms, bf16_out)
//!   4. Upload bf16 K to iGPU
//!   5. gpu.triattn_score_bf16(bf16_k, ...) -> scores
//!
//! Both paths run 100 trials each; reports median + p95 wall clock.
//! Saves bench/stage-1.5-ab-<ts>.txt per the contract format.
//!
//! Build:
//!   cargo run -p engine --features deltanet --example hipfire_x_stage_1_5_ab

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use engine::llama::KvCache;
    use engine::npu::NpuRuntime;
    use engine::triattn::{BandCenter, TriAttnCenters};
    use rdna_compute::{DType, Gpu};
    use std::time::Instant;

    let n_heads = 16usize;
    let n_kv_heads = 8usize;
    let head_dim = 256usize;
    let n_bands = head_dim / 2;
    let rope_theta = 10_000_000.0f32;
    let n_rot = head_dim;
    let seq_len = 128usize;
    let p_q = (seq_len - 1) as f32;
    let n_trials = 100usize;

    // Total chunks must equal N_ITERS=1024 of the layer kernel.
    let n_chunks = n_kv_heads * seq_len;
    assert_eq!(n_chunks, hipx::kernels::ASYM3_DEQUANT_LAYER_N_ITERS,
               "shape mismatch: {} chunks vs kernel N_ITERS {}",
               n_chunks, hipx::kernels::ASYM3_DEQUANT_LAYER_N_ITERS);

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*seed >> 40) as u32;
        let uniform = bits as f32 / (1u32 << 24) as f32;
        uniform * 2.0 - 1.0
    }

    let mut centers = TriAttnCenters::new(1, n_heads, head_dim, rope_theta, 1.0);
    let mut seed = 0xc0ffee_u64;
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
    let kv = KvCache::new_gpu_asym3(&mut gpu, 1, n_kv_heads, head_dim, seq_len)?;
    let cos_theta = kv.givens_cos.as_ref().unwrap();
    let sin_theta = kv.givens_sin.as_ref().unwrap();
    let k_cache = &kv.k_gpu[0];
    let v_cache = &kv.v_gpu[0];
    let pos_dev = gpu.hip.malloc(4)?;
    let kv_dim = n_kv_heads * head_dim;
    for pos in 0..seq_len {
        let k_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let k_tmp = gpu.upload_f32(&k_row, &[kv_dim])?;
        let v_tmp = gpu.upload_f32(&v_row, &[kv_dim])?;
        let pos_bytes = (pos as i32).to_ne_bytes();
        gpu.hip.memcpy_htod(&pos_dev, &pos_bytes)?;
        gpu.kv_cache_write_asym3_fused(
            k_cache, v_cache, &k_tmp, &v_tmp, &pos_dev,
            cos_theta, sin_theta, n_kv_heads, head_dim,
        )?;
    }
    gpu.hip.device_synchronize()?;

    let scores_a = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32)?;
    let centers_dev = gpu.upload_f32(&centers_flat, &[n_heads * n_bands * 3])?;

    // Warm up + bench Path A (iGPU baseline asym3).
    eprintln!("[bench] warming up Path A (iGPU asym3)...");
    for _ in 0..10 {
        gpu.triattn_score_asym3(
            k_cache, &centers_dev, cos_theta, sin_theta, &scores_a,
            n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
        )?;
    }
    gpu.hip.device_synchronize()?;
    let mut times_a_ns = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        let t0 = Instant::now();
        gpu.triattn_score_asym3(
            k_cache, &centers_dev, cos_theta, sin_theta, &scores_a,
            n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
        )?;
        gpu.hip.device_synchronize()?;
        times_a_ns.push(t0.elapsed().as_nanos() as u64);
    }

    // NPU init.
    let mut npu = match NpuRuntime::try_init() {
        Some(rt) => rt,
        None => { eprintln!("no NPU; cannot run Path B"); return Ok(()); }
    };
    eprintln!("[bench] NPU init OK; cols={} TOPS_INT8={}", npu.cols(), npu.tops_int8());

    // Build the asym3 K bytes layout NpuRuntime expects: per (kv_head,
    // pos) chunk, [4 B cnorm | 96 B packed]. Download from GPU once.
    let k_floats = gpu.download_f32(k_cache)?;
    let mut k_bytes = Vec::with_capacity(k_floats.len() * 4);
    for v in &k_floats { k_bytes.extend_from_slice(&v.to_ne_bytes()); }
    let k_bytes_per_head = 4 + (head_dim * 3) / 8; // 100
    let k_bytes_per_pos = n_kv_heads * k_bytes_per_head;

    let mut packed_npu = vec![0u8; n_chunks * 96];
    let mut cnorms_npu = vec![0.0f32; n_chunks];
    for pos in 0..seq_len {
        for h_kv in 0..n_kv_heads {
            let chunk_idx = pos * n_kv_heads + h_kv;
            let head_off = pos * k_bytes_per_pos + h_kv * k_bytes_per_head;
            cnorms_npu[chunk_idx] = f32::from_le_bytes([
                k_bytes[head_off], k_bytes[head_off+1],
                k_bytes[head_off+2], k_bytes[head_off+3],
            ]);
            packed_npu[chunk_idx*96..chunk_idx*96+96]
                .copy_from_slice(&k_bytes[head_off+4..head_off+4+96]);
        }
    }

    let mut bf16_out = vec![0u8; hipx::kernels::ASYM3_DEQUANT_LAYER_OUT_BYTES];
    let bf16_k_dev = gpu.hip.malloc(bf16_out.len())?;
    let scores_b = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32)?;
    let bf16_k_tensor = rdna_compute::GpuTensor {
        buf: bf16_k_dev,
        shape: vec![bf16_out.len()],
        dtype: DType::Raw,
    };

    // Warm up Path B (NPU dequant + bf16 score).
    eprintln!("[bench] warming up Path B (NPU dequant + bf16 score)...");
    for _ in 0..10 {
        npu.asym3_dequant_layer(&packed_npu, &cnorms_npu, &mut bf16_out)?;
        gpu.hip.memcpy_htod(&bf16_k_tensor.buf, &bf16_out)?;
        gpu.triattn_score_bf16(
            &bf16_k_tensor, &centers_dev, cos_theta, sin_theta, &scores_b,
            n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
        )?;
        gpu.hip.device_synchronize()?;
    }

    let mut times_b_ns = Vec::with_capacity(n_trials);
    let mut times_b_npu_ns = Vec::with_capacity(n_trials);
    let mut times_b_upload_ns = Vec::with_capacity(n_trials);
    let mut times_b_score_ns = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        let t0 = Instant::now();
        let t_npu0 = Instant::now();
        npu.asym3_dequant_layer(&packed_npu, &cnorms_npu, &mut bf16_out)?;
        let t_npu = t_npu0.elapsed().as_nanos() as u64;
        let t_up0 = Instant::now();
        gpu.hip.memcpy_htod(&bf16_k_tensor.buf, &bf16_out)?;
        let t_up = t_up0.elapsed().as_nanos() as u64;
        let t_score0 = Instant::now();
        gpu.triattn_score_bf16(
            &bf16_k_tensor, &centers_dev, cos_theta, sin_theta, &scores_b,
            n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
        )?;
        gpu.hip.device_synchronize()?;
        let t_score = t_score0.elapsed().as_nanos() as u64;
        times_b_ns.push(t0.elapsed().as_nanos() as u64);
        times_b_npu_ns.push(t_npu);
        times_b_upload_ns.push(t_up);
        times_b_score_ns.push(t_score);
    }

    fn pct(v: &mut Vec<u64>, p: f64) -> f64 {
        v.sort();
        let i = ((v.len() - 1) as f64 * p) as usize;
        v[i] as f64 / 1000.0 // us
    }
    fn med(v: &mut Vec<u64>) -> f64 { pct(v, 0.5) }
    fn p95(v: &mut Vec<u64>) -> f64 { pct(v, 0.95) }

    let a_med = med(&mut times_a_ns.clone());
    let a_p95 = p95(&mut times_a_ns.clone());
    let b_med = med(&mut times_b_ns.clone());
    let b_p95 = p95(&mut times_b_ns.clone());
    let b_npu_med = med(&mut times_b_npu_ns.clone());
    let b_up_med  = med(&mut times_b_upload_ns.clone());
    let b_score_med = med(&mut times_b_score_ns.clone());

    println!();
    println!("=== Stage 1.5 A/B bench ({n_trials} trials, seq_len={seq_len}, n_chunks={n_chunks}) ===");
    println!("  Path A (iGPU asym3 inline):  median {a_med:>8.2} us  p95 {a_p95:>8.2} us");
    println!("  Path B (NPU dequant + bf16): median {b_med:>8.2} us  p95 {b_p95:>8.2} us");
    println!("    NPU asym3_dequant_layer:   median {b_npu_med:>8.2} us");
    println!("    bf16 K upload (host->iGPU):median {b_up_med:>8.2} us");
    println!("    iGPU triattn_score_bf16:   median {b_score_med:>8.2} us");
    let lift = (a_med - b_med) / a_med * 100.0;
    println!();
    if lift > 0.0 {
        println!("  Path B is {lift:.1}% FASTER than baseline.");
    } else {
        println!("  Path B is {:.1}% SLOWER than baseline (no lift).", -lift);
    }

    let ts = std::process::Command::new("date").arg("+%Y%m%d-%H%M%S").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let log = format!(
        "Stage 1.5 A/B bench {ts}\n\
         seq_len={seq_len} n_chunks={n_chunks} n_trials={n_trials}\n\
         path_a_median_us={a_med:.2}\npath_a_p95_us={a_p95:.2}\n\
         path_b_median_us={b_med:.2}\npath_b_p95_us={b_p95:.2}\n\
         path_b_npu_median_us={b_npu_med:.2}\n\
         path_b_upload_median_us={b_up_med:.2}\n\
         path_b_score_median_us={b_score_med:.2}\n\
         lift_pct={lift:.2}\n"
    );
    let _ = std::fs::create_dir_all("bench");
    let path = format!("bench/stage-1.5-ab-{ts}.txt");
    std::fs::write(&path, &log)?;
    println!("  log: {path}");
    Ok(())
}
