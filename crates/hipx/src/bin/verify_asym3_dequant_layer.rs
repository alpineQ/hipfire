//! `hipx-verify-asym3-dequant-layer` - stage 1.4 layer-batched verifier.
//!
//! Validates the asym3_dequant_layer kernel (per-layer batched MVP at
//! N_ITERS=32) by dispatching a single batched call and comparing
//! every iteration's output against the same CPU reference used by
//! verify_asym3_dequant.rs (AIE-2P-shape: RTZ cnorm + RNE codebook
//! + RAZ output).
//!
//! Why a separate binary: the layer kernel's PDI / insts are different
//! artifacts; the BO sizes and runtime sequence DMA strides are
//! different. Reusing the 256-variant verifier with a kernel switch
//! adds branching across most of its body. Cleaner to start a focused
//! N_ITERS-shape verifier and let the two grow independently. Shared
//! helper code (ULP, conversions, CPU reference) is duplicated for
//! now; if it grows we factor a module.
//!
//! Same 3-gate acceptance as verify_asym3_dequant: determinism,
//! max bf16 ULP <= 4 per element, |mean signed| <= 1.0 ULP. Iteration
//! mechanism (DMA stride + scf.for) is the new thing being tested.
//!
//! Build:
//!   cargo build -p hipx --bin verify_asym3_dequant_layer
//! Run:
//!   ./target/debug/verify_asym3_dequant_layer [N_SEEDS]

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::Hipx;

const N_ITERS: usize = hipx::kernels::ASYM3_DEQUANT_LAYER_N_ITERS;
const HEAD_DIM: usize = hipx::kernels::ASYM3_DEQUANT_LAYER_HEAD_DIM;
const PACKED_PER_ITER: usize = HEAD_DIM * 3 / 8; // 96
const PACKED_BYTES: usize = hipx::kernels::ASYM3_DEQUANT_LAYER_PACKED_BYTES; // 3072
const CNORM_BYTES: usize = hipx::kernels::ASYM3_DEQUANT_LAYER_CNORM_BYTES;   // 128
const OUT_BYTES: usize = hipx::kernels::ASYM3_DEQUANT_LAYER_OUT_BYTES;       // 16384
const DEFAULT_SEEDS: usize = 100;

const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176,
     0.015176,  0.046469,  0.083320,  0.134860,
];

fn load_pdi() -> Vec<u8> {
    if let Ok(p) = std::env::var("ASYM3_LAYER_PDI") {
        std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("ASYM3_LAYER_PDI={p} read failed: {e}");
            std::process::exit(1);
        })
    } else {
        hipx::kernels::ASYM3_DEQUANT_LAYER_PDI.to_vec()
    }
}

fn load_insts() -> Vec<u8> {
    if let Ok(p) = std::env::var("ASYM3_LAYER_INSTS") {
        std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("ASYM3_LAYER_INSTS={p} read failed: {e}");
            std::process::exit(1);
        })
    } else {
        hipx::kernels::ASYM3_DEQUANT_LAYER_INSTS.to_vec()
    }
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
    fn next_byte(&mut self) -> u8 { (self.next_u64() & 0xff) as u8 }
    fn next_f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

fn f32_to_bf16_bits_rtz(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    (xb >> 16) as u16
}

fn f32_to_bf16_bits_raz(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    let abs = xb & 0x7fff_ffff;
    let sign = xb & 0x8000_0000;
    let biased = abs.wrapping_add(0xffff);
    ((sign | (biased & 0x7fff_ffff)) >> 16) as u16
}

fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    let lsb = (xb >> 16) & 1;
    let bias = 0x7fff + lsb;
    ((xb.wrapping_add(bias)) >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// CPU reference for one (head, position): same AIE-2P-shape model as
/// verify_asym3_dequant.rs::cpu_reference.
fn cpu_reference_one(packed: &[u8], cnorm: f32, out_bf16: &mut [u16]) {
    assert_eq!(packed.len(), PACKED_PER_ITER);
    assert_eq!(out_bf16.len(), HEAD_DIM);
    let cnorm_b = bf16_bits_to_f32(f32_to_bf16_bits_rtz(cnorm));
    let cb_b: [f32; 8] = std::array::from_fn(|i|
        bf16_bits_to_f32(f32_to_bf16_bits_rne(TURBO_C3_256[i]))
    );
    for tid in 0..32 {
        let base = tid * 3;
        let word = (packed[base] as u32)
            | ((packed[base + 1] as u32) << 8)
            | ((packed[base + 2] as u32) << 16);
        for i in 0..8 {
            let idx = ((word >> (i * 3)) & 7) as usize;
            let dim = tid * 8 + i;
            out_bf16[dim] = f32_to_bf16_bits_raz(cnorm_b * cb_b[idx]);
        }
    }
}

fn ulp_distance(a: u16, b: u16) -> u32 {
    let sa = a & 0x8000;
    let sb = b & 0x8000;
    if sa == sb {
        let mag_a = (a & 0x7fff) as i32;
        let mag_b = (b & 0x7fff) as i32;
        (mag_a - mag_b).unsigned_abs()
    } else {
        let mag_a = (a & 0x7fff) as u32;
        let mag_b = (b & 0x7fff) as u32;
        mag_a + mag_b
    }
}

fn signed_ulp_delta(cpu: u16, npu: u16) -> i32 {
    let sc = cpu & 0x8000;
    let sn = npu & 0x8000;
    let mag_c = (cpu & 0x7fff) as i32;
    let mag_n = (npu & 0x7fff) as i32;
    if sc == sn { mag_n - mag_c } else { mag_n + mag_c }
}

fn main() -> ExitCode {
    let n_seeds: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);
    let max_ulp_bound: u32 = std::env::var("ASYM3_LAYER_MAX_ULP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let mean_bias_bound: f64 = std::env::var("ASYM3_LAYER_MEAN_BIAS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);

    let pdi = load_pdi();
    let insts = load_insts();
    println!("[verify-layer] PDI {} bytes; insts {} bytes; N_ITERS={N_ITERS}; {n_seeds} seeds",
             pdi.len(), insts.len());
    println!("[verify-layer] gates: max_ulp <= {max_ulp_bound}, |mean signed| <= {mean_bias_bound:.2}, determ");

    let hipx_dev = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };

    let mut b = HwctxBuilder::default();
    b.num_columns = hipx::kernels::ASYM3_DEQUANT_LAYER_COLUMNS;
    b.max_opc = hipx::kernels::ASYM3_DEQUANT_LAYER_OPS_PER_CYCLE;
    let ctx = match hipx_dev.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };

    let pdi_bo = match hipx_dev.alloc_dev(pdi.len()) {
        Ok(b) => b,
        Err(e) => { eprintln!("pdi alloc: {e}"); return ExitCode::FAILURE; }
    };
    unsafe {
        let buf = hipx_dev.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..pdi.len()].copy_from_slice(&pdi);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx_dev.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    let instr_bo = match hipx_dev.alloc_dev(insts.len()) {
        Ok(b) => b,
        Err(e) => { eprintln!("instr alloc: {e}"); return ExitCode::FAILURE; }
    };
    unsafe {
        let buf = hipx_dev.dev_slice(&instr_bo).expect("instr slice");
        buf[..insts.len()].copy_from_slice(&insts);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);

    let mut packed_bo = hipx_dev.alloc_shmem(PACKED_BYTES).expect("packed alloc");
    let mut cnorm_bo = hipx_dev.alloc_shmem(CNORM_BYTES).expect("cnorm alloc");
    let mut out_bo = hipx_dev.alloc_shmem(OUT_BYTES).expect("out alloc");
    let packed_va = packed_bo.host_ptr().unwrap() as u64;
    let cnorm_va = cnorm_bo.host_ptr().unwrap() as u64;
    let out_va = out_bo.host_ptr().unwrap() as u64;

    let mut bo3 = match hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 8) {
        Ok(b) => b, Err(e) => { eprintln!("bo3 alloc: {e}"); return ExitCode::FAILURE; }
    };
    { let buf = bo3.map().expect("bo3 map"); for b in buf.iter_mut() { *b = 0; } }
    let _ = bo3.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3.host_ptr().unwrap() as u64;
    let mut bo4 = match hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 1) {
        Ok(b) => b, Err(e) => { eprintln!("bo4 alloc: {e}"); return ExitCode::FAILURE; }
    };
    { let buf = bo4.map().expect("bo4 map"); for b in buf.iter_mut() { *b = 0; } }
    let _ = bo4.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4.host_ptr().unwrap() as u64;

    let mut cmd_bo = hipx_dev.alloc_cmd(4096).expect("cmd alloc");
    let ninstr_dwords = (insts.len() / 4) as u32;

    // Per-seed iteration: refill packed/cnorm/out, dispatch twice
    // (determinism), compare every iteration against cpu_reference_one.
    let mut grand_max_ulp: u32 = 0;
    let mut grand_sum_signed: i64 = 0;
    let mut grand_n_diff: usize = 0;
    let mut grand_determ_ok = true;
    let mut grand_iters_checked: usize = 0;
    let mut first_fail: Option<String> = None;

    for seed in 1..=n_seeds as u64 {
        let mut rng = XorShift64::new(seed.wrapping_mul(0x9E3779B97F4A7C15));

        // Generate N_ITERS (packed, cnorm) pairs and lay out in batched
        // buffers.
        let mut packed_all = vec![0u8; PACKED_BYTES];
        let mut cnorm_all = vec![0f32; N_ITERS];
        for it in 0..N_ITERS {
            let base = it * PACKED_PER_ITER;
            for b in packed_all[base..base + PACKED_PER_ITER].iter_mut() {
                *b = rng.next_byte();
            }
            cnorm_all[it] = (rng.next_f32_unit() - 0.5) * 4.0; // cnorm in [-2, 2)
        }

        // CPU expected: N_ITERS sequential calls.
        let mut cpu_all = vec![0u16; N_ITERS * HEAD_DIM];
        for it in 0..N_ITERS {
            let p_base = it * PACKED_PER_ITER;
            let o_base = it * HEAD_DIM;
            cpu_reference_one(
                &packed_all[p_base..p_base + PACKED_PER_ITER],
                cnorm_all[it],
                &mut cpu_all[o_base..o_base + HEAD_DIM],
            );
        }

        // Stage inputs into BOs.
        {
            let buf = packed_bo.map().expect("packed map");
            buf[..PACKED_BYTES].copy_from_slice(&packed_all);
        }
        let _ = packed_bo.sync(SYNC_TO_DEVICE);
        {
            let buf = cnorm_bo.map().expect("cnorm map");
            for (i, &c) in cnorm_all.iter().enumerate() {
                let bytes = c.to_le_bytes();
                buf[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
        }
        let _ = cnorm_bo.sync(SYNC_TO_DEVICE);

        // Build command packet once per seed; reset state nibble each
        // run.
        {
            let cbuf = cmd_bo.map().expect("cmd map");
            let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
            use hipx::kernels::passthrough_4k_args as args;
            eb.set_cu_mask(0x1);
            eb.set_arg_u64(args::OPCODE, 3);
            eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
            eb.set_arg_u32(args::NINSTR, ninstr_dwords);
            eb.set_arg_u64(args::BO0, packed_va);
            eb.set_arg_u64(args::BO1, cnorm_va);
            eb.set_arg_u64(args::BO2, out_va);
            eb.set_arg_u64(args::BO3, bo3_va);
            eb.set_arg_u64(args::BO4, bo4_va);
            let _ = eb.finalize(0x3C);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);

        // Two-dispatch determinism check.
        let mut npu_run1 = vec![0u16; N_ITERS * HEAD_DIM];
        let mut npu_run2 = vec![0u16; N_ITERS * HEAD_DIM];
        for run_idx in 0..2 {
            {
                let cbuf = cmd_bo.map().expect("cmd map");
                hipx::ert::reset_state(&mut cbuf[..4]);
            }
            let _ = cmd_bo.sync(SYNC_TO_DEVICE);
            {
                let buf = out_bo.map().expect("out map");
                for b in buf[..OUT_BYTES].iter_mut() { *b = 0xCC; }
            }
            let _ = out_bo.sync(SYNC_TO_DEVICE);

            let seq = match submit_exec_cmd(
                hipx_dev.device.fd,
                &ctx,
                &[&cmd_bo],
                &[&instr_bo, &packed_bo, &cnorm_bo, &out_bo, &bo3, &bo4],
            ) {
                Ok(s) => s,
                Err(e) => { eprintln!("submit FAIL seed {seed}: {e}"); return ExitCode::FAILURE; }
            };
            if let Err(e) = timeline_wait(hipx_dev.device.fd, ctx.syncobj_handle, seq, Duration::from_secs(10)) {
                eprintln!("timeline_wait FAIL seed {seed}: {e}"); return ExitCode::FAILURE;
            }
            let _ = out_bo.sync(SYNC_FROM_DEVICE);
            let outp = out_bo.map().expect("out map back");
            let dst = if run_idx == 0 { &mut npu_run1 } else { &mut npu_run2 };
            for d in 0..N_ITERS * HEAD_DIM {
                let lo = outp[d * 2] as u16;
                let hi = outp[d * 2 + 1] as u16;
                dst[d] = lo | (hi << 8);
            }
        }
        let determ_ok = npu_run1 == npu_run2;
        if !determ_ok {
            grand_determ_ok = false;
            if first_fail.is_none() {
                first_fail = Some(format!("seed {seed} determinism FAIL"));
            }
        }

        // Compare run1 vs cpu_all element-by-element across all N_ITERS.
        for d in 0..N_ITERS * HEAD_DIM {
            let cpu_bits = cpu_all[d];
            let npu_bits = npu_run1[d];
            grand_iters_checked += 1;
            if cpu_bits != npu_bits {
                let u = ulp_distance(cpu_bits, npu_bits);
                let s = signed_ulp_delta(cpu_bits, npu_bits) as i64;
                if u > grand_max_ulp { grand_max_ulp = u; }
                grand_sum_signed += s;
                grand_n_diff += 1;
                if u > max_ulp_bound && first_fail.is_none() {
                    let it = d / HEAD_DIM;
                    let dim = d % HEAD_DIM;
                    first_fail = Some(format!(
                        "seed {seed} iter {it} dim {dim}: cpu 0x{cpu_bits:04x} vs npu 0x{npu_bits:04x} ulp {u} > bound {max_ulp_bound}"
                    ));
                }
            }
        }
    }

    let mean_signed = if grand_n_diff > 0 {
        grand_sum_signed as f64 / grand_n_diff as f64
    } else { 0.0 };

    println!();
    println!("=== verify_asym3_dequant_layer ({n_seeds} seeds, {} elements/seed) ===",
             N_ITERS * HEAD_DIM);
    println!("  determinism:  {} (run1 == run2 across seeds)",
             if grand_determ_ok { "PASS" } else { "FAIL" });
    println!("  max ULP:      observed {} <= bound {} ({})",
             grand_max_ulp, max_ulp_bound,
             if grand_max_ulp <= max_ulp_bound { "PASS" } else { "FAIL" });
    println!("  mean signed:  {:.4} ({} diffs / {} elements; bound |{:.2}|) ({})",
             mean_signed, grand_n_diff, grand_iters_checked, mean_bias_bound,
             if mean_signed.abs() <= mean_bias_bound { "PASS" } else { "FAIL" });
    if let Some(fail) = &first_fail {
        eprintln!("  first failure: {fail}");
    }

    let pass = grand_determ_ok
        && grand_max_ulp <= max_ulp_bound
        && mean_signed.abs() <= mean_bias_bound;
    if pass {
        println!("\n=== STAGE 1.4 LAYER VERIFY: PASS ===");
        ExitCode::SUCCESS
    } else {
        eprintln!("\n=== STAGE 1.4 LAYER VERIFY: FAIL ===");
        ExitCode::FAILURE
    }
}
