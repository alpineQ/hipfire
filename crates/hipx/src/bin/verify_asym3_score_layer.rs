//! `hipx-verify-asym3-score-layer` - stage 2.6 multi-iter (N_ITERS=128)
//! fused score verifier. Same per-iter math as verify_asym3_score_one
//! (same kernel C++); the layer variant batches 128 (head, position)
//! iterations in a single dispatch.
//!
//! Validates against both references:
//!   1. matching ref (NR sqrt + trig-eliminated)
//!   2. iGPU-shape ref (libm sqrtf + atan2 + cos directly)
//!
//! Build:
//!   cargo build -p hipx --bin verify_asym3_score_layer --release
//! Run:
//!   ./target/release/verify_asym3_score_layer [N_SEEDS]

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
const PER_ITER_INPUT: usize = 3200;
const N_ITERS: usize = 128;
const INPUT_BYTES: usize = N_ITERS * PER_ITER_INPUT;
const SCORE_BYTES: usize = N_ITERS * 4;
const DEFAULT_SEEDS: usize = 100;

const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176, 0.015176, 0.046469, 0.083320, 0.134860,
];

const OFF_PACKED: usize = 0;
const OFF_CNORM: usize = 96;
const OFF_C_MAG: usize = 100;
const OFF_C_ABS: usize = 612;
const OFF_COS_A: usize = 1124;
const OFF_SIN_A: usize = 1636;
const OFF_COS_T: usize = 2148;
const OFF_SIN_T: usize = 2660;

fn load_pdi() -> Vec<u8> {
    std::fs::read("kernels/aie2p/asym3_score_layer/build/main.pdi").unwrap_or_else(|e| {
        eprintln!("default PDI read failed: {e}");
        std::process::exit(1);
    })
}
fn load_insts() -> Vec<u8> {
    std::fs::read("kernels/aie2p/asym3_score_layer/build/insts.bin").unwrap_or_else(|e| {
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
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

fn fast_sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut u: u32 = x.to_bits();
    u = 0x5f3759df_u32.wrapping_sub(u >> 1);
    let mut y = f32::from_bits(u);
    y = y * (1.5 - 0.5 * x * y * y);
    y = y * (1.5 - 0.5 * x * y * y);
    x * y
}

#[allow(clippy::too_many_arguments)]
fn ref_match(
    packed: &[u8],
    cnorm: f32,
    c_mag: &[f32],
    c_abs: &[f32],
    cos_a: &[f32],
    sin_a: &[f32],
    cos_theta: &[f32],
    sin_theta: &[f32],
) -> f32 {
    let mut s_trig = 0.0f32;
    let mut s_norm = 0.0f32;
    for tid in 0..32usize {
        let base = tid * 3;
        let bits = (packed[base] as u32)
            | ((packed[base + 1] as u32) << 8)
            | ((packed[base + 2] as u32) << 16);
        let mut v = [0.0f32; 8];
        for i in 0..8 {
            v[i] = cnorm * TURBO_C3_256[((bits >> (3 * i)) & 7) as usize];
        }
        let b0 = tid * 4;
        for j in 0..4 {
            let f = b0 + j;
            let a_gv = v[j * 2];
            let b_gv = v[j * 2 + 1];
            let k_re = cos_theta[f] * a_gv + sin_theta[f] * b_gv;
            let k_im = -sin_theta[f] * a_gv + cos_theta[f] * b_gv;
            s_trig += c_mag[f] * (cos_a[f] * k_re + sin_a[f] * k_im);
            let k_mag = fast_sqrtf(k_re * k_re + k_im * k_im);
            let r = if c_abs[f] > 1e-20 {
                (c_mag[f] / c_abs[f]).min(1.0)
            } else {
                0.0
            };
            s_norm += (1.0 - r) * c_abs[f] * k_mag;
        }
    }
    s_trig + s_norm
}

#[allow(clippy::too_many_arguments)]
fn ref_igpu(
    packed: &[u8],
    cnorm: f32,
    c_re: &[f32],
    c_im: &[f32],
    c_abs: &[f32],
    omega: &[f32],
    p_q: f32,
    cos_theta: &[f32],
    sin_theta: &[f32],
) -> f32 {
    let mut s_trig = 0.0f32;
    let mut s_norm = 0.0f32;
    for tid in 0..32usize {
        let base = tid * 3;
        let bits = (packed[base] as u32)
            | ((packed[base + 1] as u32) << 8)
            | ((packed[base + 2] as u32) << 16);
        let mut v = [0.0f32; 8];
        for i in 0..8 {
            v[i] = cnorm * TURBO_C3_256[((bits >> (3 * i)) & 7) as usize];
        }
        let b0 = tid * 4;
        for j in 0..4 {
            let f = b0 + j;
            let a_gv = v[j * 2];
            let b_gv = v[j * 2 + 1];
            let k_re = cos_theta[f] * a_gv + sin_theta[f] * b_gv;
            let k_im = -sin_theta[f] * a_gv + cos_theta[f] * b_gv;
            let k_mag = (k_re * k_re + k_im * k_im).sqrt();
            let k_phase = k_im.atan2(k_re);
            let cm = (c_re[f] * c_re[f] + c_im[f] * c_im[f]).sqrt();
            let cp = c_im[f].atan2(c_re[f]);
            let angle = omega[f] * p_q + cp - k_phase;
            s_trig += cm * k_mag * angle.cos();
            let r = if c_abs[f] > 1e-20 {
                (cm / c_abs[f]).min(1.0)
            } else {
                0.0
            };
            s_norm += (1.0 - r) * c_abs[f] * k_mag;
        }
    }
    s_trig + s_norm
}

fn main() -> ExitCode {
    let n_seeds: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);
    let rope_theta: f32 = 10_000_000.0;
    let n_rot = HEAD_DIM;

    let pdi = load_pdi();
    let insts = load_insts();
    println!(
        "[verify-score-layer] PDI {} bytes; insts {} bytes; N_ITERS={N_ITERS}; {n_seeds} seeds",
        pdi.len(),
        insts.len()
    );

    let hipx_dev = match Hipx::open() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Hipx::open: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut b = HwctxBuilder::default();
    b.num_columns = 8;
    b.max_opc = 2048;
    let ctx = match hipx_dev.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create_hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };

    let pdi_bo = hipx_dev.alloc_dev(pdi.len()).expect("pdi alloc");
    unsafe {
        let buf = hipx_dev.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..pdi.len()].copy_from_slice(&pdi);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx_dev.device.fd, &ctx, vec![pdi_bo], &[0u8]).expect("config_cus");

    let instr_bo = hipx_dev.alloc_dev(insts.len()).expect("instr alloc");
    unsafe {
        let buf = hipx_dev.dev_slice(&instr_bo).expect("instr slice");
        buf[..insts.len()].copy_from_slice(&insts);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (insts.len() / 4) as u32;

    let mut input_bo = hipx_dev.alloc_shmem(INPUT_BYTES).expect("input alloc");
    let mut score_bo = hipx_dev.alloc_shmem(SCORE_BYTES).expect("score alloc");
    let _ = input_bo.map().expect("input map prime");
    let _ = input_bo.sync(SYNC_TO_DEVICE);
    let _ = score_bo.map().expect("score map prime");
    let _ = score_bo.sync(SYNC_TO_DEVICE);
    let input_va = input_bo.host_ptr().unwrap() as u64;
    let score_va = score_bo.host_ptr().unwrap() as u64;

    let mut bo3 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 8).expect("bo3");
    {
        let buf = bo3.map().expect("bo3 map");
        for b in buf.iter_mut() {
            *b = 0;
        }
    }
    let _ = bo3.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3.host_ptr().unwrap() as u64;
    let mut bo4 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 1).expect("bo4");
    {
        let buf = bo4.map().expect("bo4 map");
        for b in buf.iter_mut() {
            *b = 0;
        }
    }
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

    let t_start = std::time::Instant::now();

    let mut max_rel_match: f64 = 0.0;
    let mut max_rel_igpu: f64 = 0.0;
    let mut bad: usize = 0;
    let mut nondet: usize = 0;
    let mut n_compared: usize = 0;
    let mut first_fail: Option<String> = None;
    const SENTINEL: u32 = 0xCCCCCCCC;

    for seed in 1..=n_seeds as u64 {
        let mut rng = XorShift64::new(seed.wrapping_mul(0x9E3779B97F4A7C15));

        // Per-seed: same trig tables (constant per model/layer) + N_ITERS
        // independent (head, pos) pairs (each with own packed + cnorm).
        let mut cos_t = [0.0f32; N_BANDS];
        let mut sin_t = [0.0f32; N_BANDS];
        for f in 0..N_BANDS {
            let theta = (rng.unit() - 0.5) * std::f32::consts::PI;
            cos_t[f] = theta.cos();
            sin_t[f] = theta.sin();
        }
        let mut c_re = [0.0f32; N_BANDS];
        let mut c_im = [0.0f32; N_BANDS];
        let mut c_mag = [0.0f32; N_BANDS];
        let mut c_abs = [0.0f32; N_BANDS];
        let mut c_phase = [0.0f32; N_BANDS];
        for f in 0..N_BANDS {
            let r1 = (rng.unit() - 0.5) * 0.6;
            let r2 = (rng.unit() - 0.5) * 0.6;
            c_re[f] = r1;
            c_im[f] = r2;
            c_mag[f] = (r1 * r1 + r2 * r2).sqrt();
            c_phase[f] = r2.atan2(r1);
            c_abs[f] = 0.5 + rng.unit() * 0.3;
        }
        let p_q = (seed % 256) as f32;
        let mut omega = [0.0f32; N_BANDS];
        let mut cos_a = [0.0f32; N_BANDS];
        let mut sin_a = [0.0f32; N_BANDS];
        for f in 0..N_BANDS {
            omega[f] = if f < n_rot / 2 {
                (-2.0 * (f as f32) / (n_rot as f32) * rope_theta.ln()).exp()
            } else {
                0.0
            };
            let angle = omega[f] * p_q + c_phase[f];
            cos_a[f] = angle.cos();
            sin_a[f] = angle.sin();
        }

        // Build N_ITERS unique (packed, cnorm) pairs and pack into the
        // batched input buffer.
        let mut packed_per_iter: Vec<[u8; PACKED_BYTES]> = Vec::with_capacity(N_ITERS);
        let mut cnorm_per_iter: Vec<f32> = Vec::with_capacity(N_ITERS);
        for _it in 0..N_ITERS {
            let mut p = [0u8; PACKED_BYTES];
            for b in p.iter_mut() {
                *b = rng.next_byte();
            }
            packed_per_iter.push(p);
            cnorm_per_iter.push((rng.unit() - 0.5) * 4.0);
        }

        {
            let buf = input_bo.map().expect("input map");
            for b in buf[..INPUT_BYTES].iter_mut() {
                *b = 0;
            }
            for it in 0..N_ITERS {
                let base = it * PER_ITER_INPUT;
                buf[base + OFF_PACKED..base + OFF_PACKED + PACKED_BYTES]
                    .copy_from_slice(&packed_per_iter[it]);
                buf[base + OFF_CNORM..base + OFF_CNORM + 4]
                    .copy_from_slice(&cnorm_per_iter[it].to_le_bytes());
                for (i, &v) in c_mag.iter().enumerate() {
                    buf[base + OFF_C_MAG + i * 4..base + OFF_C_MAG + i * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
                for (i, &v) in c_abs.iter().enumerate() {
                    buf[base + OFF_C_ABS + i * 4..base + OFF_C_ABS + i * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
                for (i, &v) in cos_a.iter().enumerate() {
                    buf[base + OFF_COS_A + i * 4..base + OFF_COS_A + i * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
                for (i, &v) in sin_a.iter().enumerate() {
                    buf[base + OFF_SIN_A + i * 4..base + OFF_SIN_A + i * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
                for (i, &v) in cos_t.iter().enumerate() {
                    buf[base + OFF_COS_T + i * 4..base + OFF_COS_T + i * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
                for (i, &v) in sin_t.iter().enumerate() {
                    buf[base + OFF_SIN_T + i * 4..base + OFF_SIN_T + i * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        let _ = input_bo.sync(SYNC_TO_DEVICE);

        // Two-dispatch determinism check.
        let mut runs = [vec![0u32; N_ITERS], vec![0u32; N_ITERS]];
        for run_idx in 0..2 {
            {
                let cbuf = cmd_bo.map().expect("cmd map");
                hipx::ert::reset_state(&mut cbuf[..4]);
            }
            let _ = cmd_bo.sync(SYNC_TO_DEVICE);
            {
                let buf = score_bo.map().expect("score map");
                for b in buf[..SCORE_BYTES].iter_mut() {
                    *b = 0xCC;
                }
            }
            let _ = score_bo.sync(SYNC_TO_DEVICE);
            let seq = match submit_exec_cmd(
                hipx_dev.device.fd,
                &ctx,
                &[&cmd_bo],
                &[&instr_bo, &input_bo, &score_bo, &bo3, &bo4],
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("submit FAIL seed {seed}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(e) = timeline_wait(
                hipx_dev.device.fd,
                ctx.syncobj_handle,
                seq,
                Duration::from_secs(10),
            ) {
                eprintln!("timeline_wait FAIL seed {seed}: {e}");
                return ExitCode::FAILURE;
            }
            let _ = score_bo.sync(SYNC_FROM_DEVICE);
            let outp = score_bo.map().expect("score map back");
            for it in 0..N_ITERS {
                let mut sb = [0u8; 4];
                sb.copy_from_slice(&outp[it * 4..it * 4 + 4]);
                runs[run_idx][it] = u32::from_le_bytes(sb);
            }
        }

        for it in 0..N_ITERS {
            let bits = runs[0][it];
            if bits != runs[1][it] {
                nondet += 1;
                if first_fail.is_none() {
                    first_fail = Some(format!("seed {seed} iter {it} non-deterministic"));
                }
                continue;
            }
            let npu = f32::from_bits(bits);
            if !npu.is_finite() || bits == SENTINEL {
                bad += 1;
                if first_fail.is_none() {
                    first_fail = Some(format!("seed {seed} iter {it} bad npu=0x{bits:08x}"));
                }
                continue;
            }
            let m = ref_match(
                &packed_per_iter[it],
                cnorm_per_iter[it],
                &c_mag,
                &c_abs,
                &cos_a,
                &sin_a,
                &cos_t,
                &sin_t,
            );
            let g = ref_igpu(
                &packed_per_iter[it],
                cnorm_per_iter[it],
                &c_re,
                &c_im,
                &c_abs,
                &omega,
                p_q,
                &cos_t,
                &sin_t,
            );
            let denom_m = (m.abs() as f64).max(npu.abs() as f64).max(1e-6);
            let denom_g = (g.abs() as f64).max(npu.abs() as f64).max(1e-6);
            let rel_m = ((m - npu).abs() as f64) / denom_m;
            let rel_g = ((g - npu).abs() as f64) / denom_g;
            if rel_m > max_rel_match {
                max_rel_match = rel_m;
            }
            if rel_g > max_rel_igpu {
                max_rel_igpu = rel_g;
            }
            if rel_m > 1e-2 && first_fail.is_none() {
                first_fail = Some(format!("seed {seed} iter {it} match rel={rel_m:.4e}"));
            }
            if rel_g > 5e-2 && first_fail.is_none() {
                first_fail = Some(format!("seed {seed} iter {it} igpu rel={rel_g:.4e}"));
            }
            n_compared += 1;
        }
    }

    let elapsed = t_start.elapsed();
    let dispatches = n_seeds * 2;
    let per_dispatch = elapsed.as_secs_f64() * 1000.0 / dispatches as f64;
    let per_iter = per_dispatch * 1000.0 / N_ITERS as f64;

    println!();
    println!(
        "=== verify_asym3_score_layer ({n_seeds} seeds x {N_ITERS} iters = {} compared) ===",
        n_compared
    );
    println!(
        "  determinism:    {} ({} non-det iters)",
        if nondet == 0 { "PASS" } else { "FAIL" },
        nondet
    );
    println!("  bad outputs:    {}", bad);
    println!("  max rel match:  {max_rel_match:.4e}  (bound 1e-2)");
    println!("  max rel iGPU:   {max_rel_igpu:.4e}  (bound 5e-2)");
    println!(
        "  wall clock:     {:.3}s for {dispatches} dispatches = {per_dispatch:.3} ms/dispatch",
        elapsed.as_secs_f64()
    );
    println!("  per-iter:       {per_iter:.2} us / (head, pos)");
    if let Some(fail) = &first_fail {
        eprintln!("  first failure: {fail}");
    }

    let pass = nondet == 0 && bad == 0 && max_rel_match <= 1e-2 && max_rel_igpu <= 5e-2;
    if pass {
        println!("\n=== STAGE 2.6 SCORE LAYER VERIFY: PASS ===");
        ExitCode::SUCCESS
    } else {
        eprintln!("\n=== STAGE 2.6 SCORE LAYER VERIFY: FAIL ===");
        ExitCode::FAILURE
    }
}
