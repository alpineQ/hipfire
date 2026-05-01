//! `hipx-matmul-bf16` — BF16 4-core whole-array 512×512×512 matmul.
//! C[f32] = A[bf16] · B[bf16]. The natural-precision form for LLM
//! attention scoring and FP16 hidden-state projections; the most
//! directly applicable shape for engine integration since the iGPU
//! already operates predominantly on FP16 in the hot path.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::kernels::{
    matmul_bf16_512_4c_args as args, MATMUL_BF16_512_4C_COLUMNS, MATMUL_BF16_512_4C_INSTS,
    MATMUL_BF16_512_4C_K, MATMUL_BF16_512_4C_M, MATMUL_BF16_512_4C_N,
    MATMUL_BF16_512_4C_OPS_PER_CYCLE, MATMUL_BF16_512_4C_PDI,
};
use hipx::Hipx;

const A_BYTES: usize = MATMUL_BF16_512_4C_M * MATMUL_BF16_512_4C_K * 2; // bf16
const B_BYTES: usize = MATMUL_BF16_512_4C_K * MATMUL_BF16_512_4C_N * 2; // bf16
const C_BYTES: usize = MATMUL_BF16_512_4C_M * MATMUL_BF16_512_4C_N * 4; // f32

/// Pack an f32 to bf16 (round-toward-zero — sufficient for bench).
fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };
    println!("[mmbf] device: AIE {}.{}, fw {}.{}.{} build {}",
             hipx.info.aie_version.0, hipx.info.aie_version.1,
             hipx.info.firmware_version.0, hipx.info.firmware_version.1,
             hipx.info.firmware_version.2, hipx.info.firmware_version.3);

    let mut b = HwctxBuilder::default();
    b.num_columns = MATMUL_BF16_512_4C_COLUMNS;
    b.max_opc = MATMUL_BF16_512_4C_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };
    println!("[mmbf] hwctx={} cols={} M=K=N={} (4 cores, BF16)",
             ctx.handle, MATMUL_BF16_512_4C_COLUMNS, MATMUL_BF16_512_4C_M);

    let pdi_bo = hipx.alloc_dev(MATMUL_BF16_512_4C_PDI.len()).expect("pdi alloc");
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..MATMUL_BF16_512_4C_PDI.len()].copy_from_slice(MATMUL_BF16_512_4C_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    let instr_bo = hipx.alloc_dev(MATMUL_BF16_512_4C_INSTS.len()).expect("instr alloc");
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..MATMUL_BF16_512_4C_INSTS.len()].copy_from_slice(MATMUL_BF16_512_4C_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (MATMUL_BF16_512_4C_INSTS.len() / 4) as u32;

    // Small deterministic values so the f32 sum stays clean for verification.
    let mut a_host = vec![0f32; MATMUL_BF16_512_4C_M * MATMUL_BF16_512_4C_K];
    for r in 0..MATMUL_BF16_512_4C_M {
        for kk in 0..MATMUL_BF16_512_4C_K {
            a_host[r * MATMUL_BF16_512_4C_K + kk] = ((r + kk) % 8) as f32 * 0.125;
        }
    }
    let mut b_host = vec![0f32; MATMUL_BF16_512_4C_K * MATMUL_BF16_512_4C_N];
    for kk in 0..MATMUL_BF16_512_4C_K {
        for c in 0..MATMUL_BF16_512_4C_N {
            b_host[kk * MATMUL_BF16_512_4C_N + c] = ((kk + c) % 8) as f32 * 0.125;
        }
    }

    let mut a_bo = hipx.alloc_shmem(A_BYTES).expect("A alloc");
    {
        let buf = a_bo.map().expect("A map");
        for (i, &v) in a_host.iter().enumerate() {
            let bf = f32_to_bf16(v);
            buf[i * 2..i * 2 + 2].copy_from_slice(&bf.to_le_bytes());
        }
    }
    let _ = a_bo.sync(SYNC_TO_DEVICE);
    let a_va = a_bo.host_ptr().unwrap() as u64;

    let mut b_bo = hipx.alloc_shmem(B_BYTES).expect("B alloc");
    {
        let buf = b_bo.map().expect("B map");
        for (i, &v) in b_host.iter().enumerate() {
            let bf = f32_to_bf16(v);
            buf[i * 2..i * 2 + 2].copy_from_slice(&bf.to_le_bytes());
        }
    }
    let _ = b_bo.sync(SYNC_TO_DEVICE);
    let b_va = b_bo.host_ptr().unwrap() as u64;

    let mut c_bo = hipx.alloc_shmem(C_BYTES).expect("C alloc");
    {
        let buf = c_bo.map().expect("C map");
        for byte in buf[..C_BYTES].iter_mut() { *byte = 0xCC; }
    }
    let _ = c_bo.sync(SYNC_TO_DEVICE);
    let c_va = c_bo.host_ptr().unwrap() as u64;

    let mut bo3_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 1).expect("bo3 alloc");
    { let buf = bo3_bo.map().expect("bo3 map"); for b in buf.iter_mut() { *b = 0; } }
    let _ = bo3_bo.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3_bo.host_ptr().unwrap() as u64;
    let mut bo4_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 4).expect("bo4 alloc");
    { let buf = bo4_bo.map().expect("bo4 map"); for b in buf.iter_mut() { *b = 0; } }
    let _ = bo4_bo.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4_bo.host_ptr().unwrap() as u64;

    println!("[mmbf] BOs: A={A_BYTES} B B={B_BYTES} B C={C_BYTES} B");

    let mut cmd_bo = hipx.alloc_cmd(4096).expect("cmd alloc");
    {
        let cbuf = cmd_bo.map().expect("cmd map");
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::A, a_va);
        eb.set_arg_u64(args::B, b_va);
        eb.set_arg_u64(args::C, c_va);
        eb.set_arg_u64(args::BO3, bo3_va);
        eb.set_arg_u64(args::BO4, bo4_va);
        let _ = eb.finalize(0x3C);
    }
    let _ = cmd_bo.sync(SYNC_TO_DEVICE);

    let t0 = Instant::now();
    let seq = match submit_exec_cmd(
        hipx.device.fd, &ctx, &[&cmd_bo],
        &[&instr_bo, &a_bo, &b_bo, &c_bo, &bo3_bo, &bo4_bo],
    ) {
        Ok(s) => s, Err(e) => { eprintln!("submit: {e}"); return ExitCode::FAILURE; }
    };
    if let Err(e) = timeline_wait(hipx.device.fd, ctx.syncobj_handle, seq,
                                  Duration::from_secs(10)) {
        eprintln!("timeline_wait: {e}");
        return ExitCode::FAILURE;
    }
    let single_us = t0.elapsed().as_micros();
    let _ = c_bo.sync(SYNC_FROM_DEVICE);

    let outp = c_bo.map().expect("C re-map");
    let mut errors = 0usize;
    let mut max_err: f32 = 0.0;
    let mut first_bad = None;
    let n_check = 16;
    for r in 0..n_check {
        for c in 0..n_check {
            let mut want: f32 = 0.0;
            for kk in 0..MATMUL_BF16_512_4C_K {
                want += a_host[r * MATMUL_BF16_512_4C_K + kk] *
                        b_host[kk * MATMUL_BF16_512_4C_N + c];
            }
            let off = (r * MATMUL_BF16_512_4C_N + c) * 4;
            let bytes: [u8; 4] = outp[off..off + 4].try_into().unwrap();
            let got = f32::from_le_bytes(bytes);
            let err = (got - want).abs();
            if err > max_err { max_err = err; }
            // bf16 has 7-bit mantissa; with 512 accumulations the
            // expected error is around 1e-2 * |result|.
            let rel_err = if want.abs() > 1e-6 { err / want.abs() } else { err };
            if rel_err > 0.05 {
                if first_bad.is_none() { first_bad = Some((r, c, want, got)); }
                errors += 1;
            }
        }
    }
    if errors == 0 {
        let macs = 2.0 * MATMUL_BF16_512_4C_M as f64 * MATMUL_BF16_512_4C_K as f64 *
                   MATMUL_BF16_512_4C_N as f64;
        let gops = macs / (single_us as f64 / 1e6) / 1e9;
        println!("[mmbf] CORRECTNESS PASS — first {n_check}×{n_check} block within 5% rel-err (max abs err={max_err})");
        println!("[mmbf] single dispatch: {single_us} us → {gops:.2} GOp/s");
    } else {
        println!("[mmbf] CORRECTNESS FAIL: {errors}/{} mismatches >5% rel-err; first {first_bad:?}; max abs err={max_err}",
                 n_check * n_check);
        for c in 0..8 {
            let off = c * 4;
            let bytes: [u8; 4] = outp[off..off + 4].try_into().unwrap();
            let got = f32::from_le_bytes(bytes);
            println!("    C[0,{c}] = {got}");
        }
        return ExitCode::FAILURE;
    }

    let n_iter: u32 = 50;
    let mut total_us: u128 = 0;
    let mut max_us: u128 = 0;
    for _ in 0..n_iter {
        {
            let cbuf = cmd_bo.map().expect("cmd reset");
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);
        let t = Instant::now();
        let seq = submit_exec_cmd(hipx.device.fd, &ctx, &[&cmd_bo],
            &[&instr_bo, &a_bo, &b_bo, &c_bo, &bo3_bo, &bo4_bo]).expect("submit");
        timeline_wait(hipx.device.fd, ctx.syncobj_handle, seq,
                      Duration::from_secs(10)).expect("wait");
        let us = t.elapsed().as_micros();
        total_us += us;
        if us > max_us { max_us = us; }
    }
    let mean_us = total_us / n_iter as u128;
    let macs = 2.0 * MATMUL_BF16_512_4C_M as f64 * MATMUL_BF16_512_4C_K as f64 *
               MATMUL_BF16_512_4C_N as f64;
    let gops = macs / (mean_us as f64 / 1e6) / 1e9;
    println!("[mmbf] perf: mean={mean_us}us max={max_us}us → {gops:.2} GOp/s ({n_iter} iters, {} MMACs)",
             (macs / 2.0 / 1e6) as u64);

    ExitCode::SUCCESS
}
