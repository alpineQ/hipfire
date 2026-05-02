//! `hipfire_x_asym3_shadow` — stage 1.3b heavy-load shadow harness.
//!
//! Mirrors the engine's per-(layer, head, position) calling pattern
//! against `NpuRuntime::asym3_dequant_256`. For each simulated
//! (layer, head, position), generates an asym3 K-cache slice with
//! realistic statistics and dispatches the kernel; compares the
//! output element-by-element against the engine TURBO_C3_256
//! codebook with the same gates as `verify_asym3_dequant`:
//!
//!   - max bf16 ULP <= 4 per element
//!   - |mean signed ULP| <= 1
//!   - deterministic across two consecutive dispatches (sampled)
//!
//! Logs per-layer summary to `bench/shadow-<timestamp>.tsv`. Catches
//! lazy-init steady-state regressions, BO reuse drift, and any per-
//! layer state leakage that the 100-seed verifier would miss because
//! it only does ~100 dispatches in tight succession.
//!
//! Why not instrument `cask.rs::eviction_step` directly: the iGPU
//! `kv_fold_asym3` and `triattn_score_asym3` paths consume `(cnorm,
//! packed)` from inside the K-cache buffer and produce different
//! outputs (folded weights / scores), neither of which is bf16 K
//! that aligns with the NPU dequant output. A standalone harness
//! that compares both paths against the engine codebook reference
//! is structurally simpler and avoids putting an opt-in branch in
//! the decode hot path until 1.3a + 1.5 ship.
//!
//! Run on hipx:
//!   cargo run -p engine --features npu --release --example hipfire_x_asym3_shadow

#[cfg(feature = "npu")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use engine::npu::NpuRuntime;
    use std::time::Instant;

    const TURBO_C3_256: [f32; 8] = [
        -0.134860, -0.083320, -0.046469, -0.015176,
         0.015176,  0.046469,  0.083320,  0.134860,
    ];
    // Configurable via env vars so the harness can scale up/down
    // without recompile. Defaults model 27B Gemma decode for 16
    // layers x 8 heads x 8 positions = 1024 dispatches.
    fn env_or<T: std::str::FromStr>(k: &str, def: T) -> T {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
    }
    let n_layers: usize = env_or("ASYM3_SHADOW_LAYERS", 16);
    let n_heads: usize = env_or("ASYM3_SHADOW_HEADS", 8);
    let n_positions: usize = env_or("ASYM3_SHADOW_POSITIONS", 8);
    let max_ulp_bound: u32 = env_or("ASYM3_SHADOW_MAX_ULP", 4);
    let mean_bias_bound: f64 = env_or("ASYM3_SHADOW_MEAN_BIAS", 1.0);

    fn f32_to_bf16_rtz(x: f32) -> u16 { (x.to_bits() >> 16) as u16 }
    fn f32_to_bf16_raz(x: f32) -> u16 {
        let xb = x.to_bits();
        let abs = xb & 0x7fff_ffff;
        let sign = xb & 0x8000_0000;
        let biased = abs.wrapping_add(0xffff);
        ((sign | (biased & 0x7fff_ffff)) >> 16) as u16
    }
    fn bf16_bits_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }
    fn ulp_distance(a: u16, b: u16) -> u32 {
        let sa = a & 0x8000;
        let sb = b & 0x8000;
        if sa == sb {
            ((a & 0x7fff) as i32 - (b & 0x7fff) as i32).unsigned_abs()
        } else {
            (a & 0x7fff) as u32 + (b & 0x7fff) as u32
        }
    }
    fn signed_ulp(cpu: u16, npu: u16) -> i32 {
        let sc = cpu & 0x8000;
        let sn = npu & 0x8000;
        let mc = (cpu & 0x7fff) as i32;
        let mn = (npu & 0x7fff) as i32;
        if sc == sn { mn - mc } else { mn + mc }
    }

    // CPU-side reference using the AIE-2P-shape model (RTZ cnorm,
    // RAZ output). Mirrors crates/hipx/src/bin/verify_asym3_dequant
    // exactly so the shadow harness applies the same gate.
    fn cpu_reference(packed: &[u8; 96], cnorm: f32, out: &mut [u16; 256]) {
        let cnorm_b = bf16_bits_to_f32(f32_to_bf16_rtz(cnorm));
        let cb_b: [f32; 8] = std::array::from_fn(|i|
            bf16_bits_to_f32(f32_to_bf16_rtz(TURBO_C3_256[i]))
        );
        for tid in 0..32usize {
            let base = tid * 3;
            let word = (packed[base] as u32)
                | ((packed[base + 1] as u32) << 8)
                | ((packed[base + 2] as u32) << 16);
            for i in 0..8 {
                let idx = ((word >> (i * 3)) & 7) as usize;
                let dim = tid * 8 + i;
                out[dim] = f32_to_bf16_raz(cnorm_b * cb_b[idx]);
            }
        }
    }

    // xorshift64 deterministic PRNG.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
        fn unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32) / (1u32 << 24) as f32
        }
    }

    let mut rt = NpuRuntime::try_init().ok_or("no NPU on this system")?;
    println!("[shadow] NPU init OK; cols={} TOPS_INT8={}",
             rt.cols(), rt.tops_int8());
    println!(
        "[shadow] {} layers x {} heads x {} positions = {} dispatches",
        n_layers, n_heads, n_positions, n_layers * n_heads * n_positions
    );
    println!(
        "[shadow] gate: max_ulp <= {} mean_bias <= {:.2}",
        max_ulp_bound, mean_bias_bound
    );

    // Bench dir.
    std::fs::create_dir_all("bench").ok();
    let ts = std::process::Command::new("date")
        .arg("+%Y%m%d-%H%M%S").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let log_path = format!("bench/shadow-{}.tsv", ts);
    let mut log = String::new();
    log.push_str("layer\tn_dispatches\tmax_ulp\tmean_signed_ulp\tn_diff_total\tdeterm_ok\n");

    let mut grand_max_ulp: u32 = 0;
    let mut grand_sum_signed: i64 = 0;
    let mut grand_n_diff: usize = 0;
    let mut grand_determ_ok = true;
    let mut grand_dispatches: usize = 0;
    let t_start = Instant::now();

    for layer in 0..n_layers {
        let mut layer_max_ulp: u32 = 0;
        let mut layer_sum_signed: i64 = 0;
        let mut layer_n_diff: usize = 0;
        let mut layer_determ_ok = true;
        let mut rng = Rng(0xC001D00D ^ (layer as u64).wrapping_mul(0x9E3779B97F4A7C15));

        for h in 0..n_heads {
            for p in 0..n_positions {
                let mut packed = [0u8; 96];
                for b in packed.iter_mut() { *b = rng.byte(); }
                // cnorm in [-2, 2) realistic for asym3 magnitude factor
                let cnorm: f32 = (rng.unit() - 0.5) * 4.0;

                let mut out = [0u8; 512];
                rt.asym3_dequant_256(&packed, cnorm, &mut out)?;

                // Sample determinism: re-dispatch every 64th call.
                let determ_check = (h * n_positions + p) % 64 == 0;
                if determ_check {
                    let mut out2 = [0u8; 512];
                    rt.asym3_dequant_256(&packed, cnorm, &mut out2)?;
                    if out != out2 {
                        layer_determ_ok = false;
                        eprintln!("[shadow] WARN layer {layer} h {h} p {p} non-deterministic");
                    }
                }

                let mut cpu_out = [0u16; 256];
                cpu_reference(&packed, cnorm, &mut cpu_out);

                for d in 0..256usize {
                    let npu_bits = (out[d * 2] as u16) | ((out[d * 2 + 1] as u16) << 8);
                    let cpu_bits = cpu_out[d];
                    if npu_bits != cpu_bits {
                        let u = ulp_distance(cpu_bits, npu_bits);
                        let s = signed_ulp(cpu_bits, npu_bits) as i64;
                        if u > layer_max_ulp { layer_max_ulp = u; }
                        layer_sum_signed += s;
                        layer_n_diff += 1;
                    }
                }
                grand_dispatches += 1;
            }
        }

        let layer_mean = if layer_n_diff > 0 {
            layer_sum_signed as f64 / layer_n_diff as f64
        } else { 0.0 };
        log.push_str(&format!(
            "{}\t{}\t{}\t{:.4}\t{}\t{}\n",
            layer, n_heads * n_positions, layer_max_ulp,
            layer_mean, layer_n_diff,
            if layer_determ_ok { "ok" } else { "FAIL" }
        ));
        if layer < 4 || layer == n_layers - 1 {
            println!(
                "  layer {:2}: dispatches={} max_ulp={} mean_signed={:.4} determ={}",
                layer, n_heads * n_positions, layer_max_ulp,
                layer_mean,
                if layer_determ_ok { "ok" } else { "FAIL" }
            );
        }
        if layer_max_ulp > grand_max_ulp { grand_max_ulp = layer_max_ulp; }
        grand_sum_signed += layer_sum_signed;
        grand_n_diff += layer_n_diff;
        if !layer_determ_ok { grand_determ_ok = false; }
    }

    let elapsed = t_start.elapsed();
    let grand_mean = if grand_n_diff > 0 {
        grand_sum_signed as f64 / grand_n_diff as f64
    } else { 0.0 };
    log.push_str(&format!(
        "TOTAL\t{}\t{}\t{:.4}\t{}\t{}\n",
        grand_dispatches, grand_max_ulp, grand_mean, grand_n_diff,
        if grand_determ_ok { "ok" } else { "FAIL" }
    ));
    std::fs::write(&log_path, &log)?;

    println!(
        "\n=== Stage 1.3b shadow harness ({} dispatches in {:.3}s) ===",
        grand_dispatches,
        elapsed.as_secs_f64()
    );
    println!("  determinism:  {}", if grand_determ_ok { "PASS" } else { "FAIL" });
    println!("  max ULP:      observed {} <= bound {} ({})",
             grand_max_ulp, max_ulp_bound,
             if grand_max_ulp <= max_ulp_bound { "PASS" } else { "FAIL" });
    println!("  mean signed:  {:.4} <= bound {:.2} ({})",
             grand_mean, mean_bias_bound,
             if grand_mean.abs() <= mean_bias_bound { "PASS" } else { "FAIL" });
    println!("  log:          {log_path}");

    let pass = grand_determ_ok
        && grand_max_ulp <= max_ulp_bound
        && grand_mean.abs() <= mean_bias_bound;
    if pass {
        println!("\n=== STAGE 1.3b PASS ===");
        Ok(())
    } else {
        Err("shadow harness gates failed; see log".into())
    }
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
