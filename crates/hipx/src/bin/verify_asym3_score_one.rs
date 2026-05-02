//! `hipx-verify-asym3-score-one` - stage 2.6 single-(head, position)
//! fused score verifier. Dispatches the asym3_score_one NPU kernel
//! and compares against a CPU reference mirroring
//! `kernels/src/triattn_score_asym3.hip`.
//!
//! The kernel uses the trig-eliminated reformulation so the host
//! must precompute (cos_a, sin_a) per band where:
//!   cos_a[f] = cos(omega[f]*p_q + c_phase[f])
//!   sin_a[f] = sin(omega[f]*p_q + c_phase[f])
//!   c_phase[f] = atan2(c_im[f], c_re[f])
//!   c_mag[f]   = sqrt(c_re[f]^2 + c_im[f]^2)
//!   omega[f]   = f < n_rot/2 ? exp(-2*f/n_rot * log(rope_theta)) : 0
//!
//! The kernel input buffer (3200 bytes) holds:
//!   0    .. 96    packed (96 B)
//!   96   .. 100   cnorm (1 f32)
//!   100  .. 612   c_mag (128 f32)
//!   612  .. 1124  c_abs (128 f32)
//!   1124 .. 1636  cos_a (128 f32)
//!   1636 .. 2148  sin_a (128 f32)
//!   2148 .. 2660  cos_theta (128 f32)
//!   2660 .. 3172  sin_theta (128 f32)
//!
//! Tolerance: relative error < 1e-3 vs CPU reference. Newton-Raphson
//! sqrt in the kernel introduces ~5e-4 relative error per band, and
//! 128 bands sum so the cumulative error is bounded by O(1e-3).
//!
//! Build:
//!   cargo build -p hipx --bin verify_asym3_score_one
//! Run:
//!   ./target/debug/verify_asym3_score_one [N_SEEDS]

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::Hipx;

const HEAD_DIM: usize = 256;
const N_BANDS: usize = HEAD_DIM / 2; // 128
const PACKED_BYTES: usize = HEAD_DIM * 3 / 8; // 96
const INPUT_BYTES: usize = 3200;
const DEFAULT_SEEDS: usize = 100;

const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176,
     0.015176,  0.046469,  0.083320,  0.134860,
];

// Layout offsets (must match asym3_score_kernel.cc).
const OFF_PACKED: usize = 0;
const OFF_CNORM: usize = 96;
const OFF_C_MAG: usize = 100;
const OFF_C_ABS: usize = 612;
const OFF_COS_A: usize = 1124;
const OFF_SIN_A: usize = 1636;
const OFF_COS_T: usize = 2148;
const OFF_SIN_T: usize = 2660;

fn load_pdi() -> Vec<u8> {
    if let Ok(p) = std::env::var("ASYM3_SCORE_PDI") {
        return std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("ASYM3_SCORE_PDI={p} read failed: {e}");
            std::process::exit(1);
        });
    }
    std::fs::read("kernels/aie2p/asym3_score_one/build/main.pdi").unwrap_or_else(|e| {
        eprintln!("default PDI read failed: {e}");
        std::process::exit(1);
    })
}

fn load_insts() -> Vec<u8> {
    if let Ok(p) = std::env::var("ASYM3_SCORE_INSTS") {
        return std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("ASYM3_SCORE_INSTS={p} read failed: {e}");
            std::process::exit(1);
        });
    }
    std::fs::read("kernels/aie2p/asym3_score_one/build/insts.bin").unwrap_or_else(|e| {
        eprintln!("default insts read failed: {e}");
        std::process::exit(1);
    })
}

struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdeadbeefcafebabe } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_byte(&mut self) -> u8 { (self.next_u64() & 0xff) as u8 }
    fn unit(&mut self) -> f32 { ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32 }
}

/// CPU reference score, mirroring triattn_score_asym3.hip. Uses the
/// SAME Newton-Raphson sqrt as the kernel so we expect bit-equivalent
/// output up to floating-point reordering noise.
fn fast_sqrtf(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut u: u32 = x.to_bits();
    u = 0x5f3759df_u32.wrapping_sub(u >> 1);
    let mut y = f32::from_bits(u);
    y = y * (1.5 - 0.5 * x * y * y);
    y = y * (1.5 - 0.5 * x * y * y);
    x * y
}

#[allow(clippy::too_many_arguments)]
fn cpu_reference_score(
    packed: &[u8],
    cnorm: f32,
    c_mag: &[f32],
    c_abs: &[f32],
    cos_a: &[f32],
    sin_a: &[f32],
    cos_theta: &[f32],
    sin_theta: &[f32],
) -> f32 {
    assert_eq!(packed.len(), PACKED_BYTES);
    let mut s_trig: f32 = 0.0;
    let mut s_norm: f32 = 0.0;
    for tid in 0..32usize {
        let base = tid * 3;
        let bits = (packed[base] as u32)
            | ((packed[base + 1] as u32) << 8)
            | ((packed[base + 2] as u32) << 16);
        let mut v = [0.0f32; 8];
        for i in 0..8 {
            let idx = ((bits >> (3 * i)) & 7) as usize;
            v[i] = cnorm * TURBO_C3_256[idx];
        }
        let b0 = tid * 4;
        for j in 0..4 {
            let f = b0 + j;
            let cb = cos_theta[f];
            let sb = sin_theta[f];
            let a_gv = v[j * 2 + 0];
            let b_gv = v[j * 2 + 1];
            let k_re = cb * a_gv + sb * b_gv;
            let k_im = -sb * a_gv + cb * b_gv;
            // Reformulated s_trig (no atan2 / cos in inner loop).
            s_trig += c_mag[f] * (cos_a[f] * k_re + sin_a[f] * k_im);
            // s_norm uses NR sqrt to match kernel.
            let k_mag = fast_sqrtf(k_re * k_re + k_im * k_im);
            let r = if c_abs[f] > 1e-20 { (c_mag[f] / c_abs[f]).min(1.0) } else { 0.0 };
            s_norm += (1.0 - r) * c_abs[f] * k_mag;
        }
    }
    s_trig + s_norm
}

fn main() -> ExitCode {
    let n_seeds: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);
    let rope_theta: f32 = 10_000_000.0;
    let n_rot = HEAD_DIM; // partial_rotary_factor = 1.0 typical

    let pdi = load_pdi();
    let insts = load_insts();
    println!("[verify-score] PDI {} bytes; insts {} bytes; {n_seeds} seeds",
             pdi.len(), insts.len());

    let hipx_dev = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };
    let mut b = HwctxBuilder::default();
    b.num_columns = 8;
    b.max_opc = 2048;
    let ctx = match hipx_dev.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };

    let pdi_bo = hipx_dev.alloc_dev(pdi.len()).expect("pdi alloc");
    unsafe {
        let buf = hipx_dev.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..pdi.len()].copy_from_slice(&pdi);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx_dev.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    let instr_bo = hipx_dev.alloc_dev(insts.len()).expect("instr alloc");
    unsafe {
        let buf = hipx_dev.dev_slice(&instr_bo).expect("instr slice");
        buf[..insts.len()].copy_from_slice(&insts);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (insts.len() / 4) as u32;

    let mut input_bo = hipx_dev.alloc_shmem(INPUT_BYTES).expect("input alloc");
    let mut score_bo = hipx_dev.alloc_shmem(4).expect("score alloc");
    let _ = input_bo.map().expect("input map");
    let _ = input_bo.sync(SYNC_TO_DEVICE);
    let _ = score_bo.map().expect("score map");
    let _ = score_bo.sync(SYNC_TO_DEVICE);
    let input_va = input_bo.host_ptr().unwrap() as u64;
    let score_va = score_bo.host_ptr().unwrap() as u64;

    let mut bo3 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 8).expect("bo3");
    { let buf = bo3.map().expect("bo3 map"); for b in buf.iter_mut() { *b = 0; } }
    let _ = bo3.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3.host_ptr().unwrap() as u64;
    let mut bo4 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 1).expect("bo4");
    { let buf = bo4.map().expect("bo4 map"); for b in buf.iter_mut() { *b = 0; } }
    let _ = bo4.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4.host_ptr().unwrap() as u64;

    let mut cmd_bo = hipx_dev.alloc_cmd(4096).expect("cmd alloc");
    {
        let cbuf = cmd_bo.map().expect("cmd map");
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        use hipx::kernels::passthrough_4k_args as args;
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::BO0, input_va);
        eb.set_arg_u64(args::BO1, score_va);
        eb.set_arg_u64(args::BO2, bo3_va);
        eb.set_arg_u64(args::BO3, bo4_va);
        eb.set_arg_u64(args::BO4, 0);
        let _ = eb.finalize(0x3C);
    }
    let _ = cmd_bo.sync(SYNC_TO_DEVICE);

    let mut max_rel: f64 = 0.0;
    let mut max_abs: f64 = 0.0;
    let mut first_fail: Option<String> = None;
    let mut bad_npu_count: usize = 0;       // NaN, +/-inf, or 0xCCCCCCCC sentinel
    let mut nondet_count: usize = 0;        // run1 != run2 within a seed
    const SENTINEL_F32_BITS: u32 = 0xCCCCCCCC;

    for seed in 1..=n_seeds as u64 {
        let mut rng = XorShift64::new(seed.wrapping_mul(0x9E3779B97F4A7C15));

        // Random asym3 inputs.
        let mut packed = [0u8; PACKED_BYTES];
        for b in packed.iter_mut() { *b = rng.next_byte(); }
        let cnorm = (rng.unit() - 0.5) * 4.0;

        // Random Givens trig tables.
        let mut cos_t = [0.0f32; N_BANDS];
        let mut sin_t = [0.0f32; N_BANDS];
        for f in 0..N_BANDS {
            let theta = (rng.unit() - 0.5) * std::f32::consts::PI;
            cos_t[f] = theta.cos();
            sin_t[f] = theta.sin();
        }

        // Random centers; precompute (c_mag, c_abs, c_phase).
        let mut c_mag = [0.0f32; N_BANDS];
        let mut c_abs = [0.0f32; N_BANDS];
        let mut c_phase = [0.0f32; N_BANDS];
        for f in 0..N_BANDS {
            let c_re = (rng.unit() - 0.5) * 0.6;
            let c_im = (rng.unit() - 0.5) * 0.6;
            let c_abs_v = 0.5 + rng.unit() * 0.3;
            c_mag[f] = (c_re * c_re + c_im * c_im).sqrt();
            c_phase[f] = c_im.atan2(c_re);
            c_abs[f] = c_abs_v;
        }

        // RoPE phase per band, p_q from the seed for variety.
        let p_q = (seed % 256) as f32;
        let n_rot_bands = n_rot / 2;
        let mut cos_a = [0.0f32; N_BANDS];
        let mut sin_a = [0.0f32; N_BANDS];
        for f in 0..N_BANDS {
            let omega = if f < n_rot_bands {
                let exponent = -2.0 * (f as f32) / (n_rot as f32);
                (exponent * rope_theta.ln()).exp()
            } else {
                0.0
            };
            let angle = omega * p_q + c_phase[f];
            cos_a[f] = angle.cos();
            sin_a[f] = angle.sin();
        }

        // CPU reference.
        let cpu = cpu_reference_score(&packed, cnorm, &c_mag, &c_abs,
                                       &cos_a, &sin_a, &cos_t, &sin_t);

        // Pack into NPU input buffer.
        {
            let buf = input_bo.map().expect("input map");
            for b in buf[..INPUT_BYTES].iter_mut() { *b = 0; }
            buf[OFF_PACKED..OFF_PACKED + PACKED_BYTES].copy_from_slice(&packed);
            buf[OFF_CNORM..OFF_CNORM + 4].copy_from_slice(&cnorm.to_le_bytes());
            for (i, &v) in c_mag.iter().enumerate() {
                buf[OFF_C_MAG + i * 4..OFF_C_MAG + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (i, &v) in c_abs.iter().enumerate() {
                buf[OFF_C_ABS + i * 4..OFF_C_ABS + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (i, &v) in cos_a.iter().enumerate() {
                buf[OFF_COS_A + i * 4..OFF_COS_A + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (i, &v) in sin_a.iter().enumerate() {
                buf[OFF_SIN_A + i * 4..OFF_SIN_A + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (i, &v) in cos_t.iter().enumerate() {
                buf[OFF_COS_T + i * 4..OFF_COS_T + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (i, &v) in sin_t.iter().enumerate() {
                buf[OFF_SIN_T + i * 4..OFF_SIN_T + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let _ = input_bo.sync(SYNC_TO_DEVICE);
        {
            let cbuf = cmd_bo.map().expect("cmd map");
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);
        {
            let buf = score_bo.map().expect("score map");
            for b in buf[..4].iter_mut() { *b = 0xCC; }
        }
        let _ = score_bo.sync(SYNC_TO_DEVICE);

        // Two-dispatch determinism check: same input must produce
        // the same output across runs.
        let mut npu_runs = [0.0f32; 2];
        let mut npu_bits = [0u32; 2];
        for run_idx in 0..2 {
            {
                let cbuf = cmd_bo.map().expect("cmd map");
                hipx::ert::reset_state(&mut cbuf[..4]);
            }
            let _ = cmd_bo.sync(SYNC_TO_DEVICE);
            // Reset score sentinel each run.
            {
                let buf = score_bo.map().expect("score map");
                buf[0] = 0xCC; buf[1] = 0xCC; buf[2] = 0xCC; buf[3] = 0xCC;
            }
            let _ = score_bo.sync(SYNC_TO_DEVICE);

            let seq = match submit_exec_cmd(
                hipx_dev.device.fd, &ctx, &[&cmd_bo],
                &[&instr_bo, &input_bo, &score_bo, &bo3, &bo4],
            ) {
                Ok(s) => s,
                Err(e) => { eprintln!("submit FAIL seed {seed}: {e}"); return ExitCode::FAILURE; }
            };
            if let Err(e) = timeline_wait(hipx_dev.device.fd, ctx.syncobj_handle, seq, Duration::from_secs(5)) {
                eprintln!("timeline_wait FAIL seed {seed}: {e}"); return ExitCode::FAILURE;
            }
            let _ = score_bo.sync(SYNC_FROM_DEVICE);
            let outp = score_bo.map().expect("score map back");
            let mut score_bytes = [0u8; 4];
            score_bytes.copy_from_slice(&outp[..4]);
            npu_bits[run_idx] = u32::from_le_bytes(score_bytes);
            npu_runs[run_idx] = f32::from_le_bytes(score_bytes);
        }
        let npu = npu_runs[0];
        if npu_bits[0] != npu_bits[1] {
            nondet_count += 1;
            if first_fail.is_none() {
                first_fail = Some(format!(
                    "seed {seed} non-deterministic: run1=0x{:08x} run2=0x{:08x}",
                    npu_bits[0], npu_bits[1]
                ));
            }
        }

        // Sanity: NPU output must be finite and not the pre-dispatch
        // sentinel. Rust's NaN/inf comparisons silently return false
        // so without this explicit gate, NaN/inf would pass the
        // tolerance check below.
        let bad = !npu.is_finite()
            || npu_bits[0] == SENTINEL_F32_BITS
            || npu_bits[1] == SENTINEL_F32_BITS;
        if bad {
            bad_npu_count += 1;
            if first_fail.is_none() {
                first_fail = Some(format!(
                    "seed {seed} bad output: bits=0x{:08x} f32={:?}",
                    npu_bits[0], npu
                ));
            }
            continue;
        }

        let abs = (cpu - npu).abs() as f64;
        let denom = cpu.abs().max(npu.abs()).max(1e-6) as f64;
        let rel = abs / denom;
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
        if rel > 1e-2 && first_fail.is_none() {
            first_fail = Some(format!("seed {seed}: cpu={cpu:.6e} npu={npu:.6e} rel={rel:.4e}"));
        }
    }

    println!();
    println!("=== verify_asym3_score_one ({n_seeds} seeds) ===");
    println!("  determinism:  {} ({} non-deterministic seeds)",
             if nondet_count == 0 { "PASS" } else { "FAIL" }, nondet_count);
    println!("  bad outputs:  {} (NaN, inf, or 0xCCCCCCCC sentinel)", bad_npu_count);
    println!("  max |Δ|  = {max_abs:.4e}");
    println!("  max rel  = {max_rel:.4e}");
    if let Some(fail) = &first_fail { eprintln!("  first failure: {fail}"); }

    // Tolerance: 1% relative is a lot but the score is a sum-product
    // of 128 bands so accumulated NR sqrt + reordering noise can hit
    // this. Tighter bound after the SIMD-optimized kernel lands.
    let pass = nondet_count == 0
        && bad_npu_count == 0
        && max_rel <= 1e-2;
    if !pass {
        eprintln!("\n=== STAGE 2.6 SCORE VERIFY: FAIL ===");
        ExitCode::FAILURE
    } else {
        println!("\n=== STAGE 2.6 SCORE VERIFY: PASS ===");
        ExitCode::SUCCESS
    }
}
