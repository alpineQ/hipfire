//! `hipx-matmul-512` — 4-core whole-array i16×i16→i32 matmul at
//! 512×512×512. Builds on the matvec proof and lights up all 4 AIE
//! columns. Useful both as a hardware sanity-check that multi-core
//! dispatch works and as a perf reference for what the AIE array can
//! deliver when the compute fraction is large.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::kernels::{
    matmul_512_4c_args as args, MATMUL_512_4C_COLUMNS, MATMUL_512_4C_INSTS,
    MATMUL_512_4C_K, MATMUL_512_4C_M, MATMUL_512_4C_N,
    MATMUL_512_4C_OPS_PER_CYCLE, MATMUL_512_4C_PDI,
};
use hipx::Hipx;

const A_BYTES: usize = MATMUL_512_4C_M * MATMUL_512_4C_K * 2; // i16
const B_BYTES: usize = MATMUL_512_4C_K * MATMUL_512_4C_N * 2; // i16
const C_BYTES: usize = MATMUL_512_4C_M * MATMUL_512_4C_N * 4; // i32

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };
    println!("[mm] device: AIE {}.{}, fw {}.{}.{} build {}",
             hipx.info.aie_version.0, hipx.info.aie_version.1,
             hipx.info.firmware_version.0, hipx.info.firmware_version.1,
             hipx.info.firmware_version.2, hipx.info.firmware_version.3);

    let mut b = HwctxBuilder::default();
    b.num_columns = MATMUL_512_4C_COLUMNS;
    b.max_opc = MATMUL_512_4C_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };
    println!("[mm] hwctx={} cols={} M=K=N={} (4 cores)",
             ctx.handle, MATMUL_512_4C_COLUMNS, MATMUL_512_4C_M);

    // PDI BO (DEV)
    let pdi_bo = hipx.alloc_dev(MATMUL_512_4C_PDI.len()).expect("pdi alloc");
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..MATMUL_512_4C_PDI.len()].copy_from_slice(MATMUL_512_4C_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    // KEEP the CuBinding alive — when config_cus's return value is
    // dropped, the pdi_bo handle is closed, the firmware loses its
    // PDI reference, and subsequent dispatches see "completed but
    // empty output" because the CU never re-loads. matvec accidentally
    // worked because its PDI was small enough to share a heap page
    // with instr (instr was actually overlapping the freed PDI region
    // and pointing at the right bytes by luck).
    let _cu = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    // No pad — DEV_HEAP allocates BOs sequentially, so instr lands at
    // heap_base + sizeof(pdi). Worker-class kernels that depended on
    // the old vec_scalar_mul "pad to 0x8000" workaround were really
    // just covering for the dropped-PDI bug.

    // Instruction stream (DEV)
    let instr_bo = hipx.alloc_dev(MATMUL_512_4C_INSTS.len()).expect("instr alloc");
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..MATMUL_512_4C_INSTS.len()].copy_from_slice(MATMUL_512_4C_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (MATMUL_512_4C_INSTS.len() / 4) as u32;

    // A, B — fill with small deterministic values (i*j mod 8) so the
    // accumulated product fits in i32.
    let mut a_host = vec![0i16; MATMUL_512_4C_M * MATMUL_512_4C_K];
    for r in 0..MATMUL_512_4C_M {
        for kk in 0..MATMUL_512_4C_K {
            a_host[r * MATMUL_512_4C_K + kk] = ((r + kk) as i16) & 0x7;
        }
    }
    let mut b_host = vec![0i16; MATMUL_512_4C_K * MATMUL_512_4C_N];
    for kk in 0..MATMUL_512_4C_K {
        for c in 0..MATMUL_512_4C_N {
            b_host[kk * MATMUL_512_4C_N + c] = ((kk + c) as i16) & 0x7;
        }
    }

    let mut a_bo = hipx.alloc_shmem(A_BYTES).expect("A alloc");
    {
        let buf = a_bo.map().expect("A map");
        for (i, &v) in a_host.iter().enumerate() {
            buf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    let _ = a_bo.sync(SYNC_TO_DEVICE);
    let a_va = a_bo.host_ptr().unwrap() as u64;

    let mut b_bo = hipx.alloc_shmem(B_BYTES).expect("B alloc");
    {
        let buf = b_bo.map().expect("B map");
        for (i, &v) in b_host.iter().enumerate() {
            buf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
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

    // whole_array test.cpp uses bo_tmp1 = 1 byte (group 6) and
    // bo_trace = trace_size*4 (or 4 bytes when trace_size=0) (group 7).
    // Match exactly — vec_scalar_mul's 8B/1B sizes were specific to
    // its kernel; whole_array expects different sizes.
    let mut bo3_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 1).expect("bo3 alloc");
    {
        let buf = bo3_bo.map().expect("bo3 map");
        for b in buf.iter_mut() { *b = 0; }
    }
    let _ = bo3_bo.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3_bo.host_ptr().unwrap() as u64;
    let mut bo4_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 4).expect("bo4 alloc");
    {
        let buf = bo4_bo.map().expect("bo4 map");
        for b in buf.iter_mut() { *b = 0; }
    }
    let _ = bo4_bo.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4_bo.host_ptr().unwrap() as u64;

    println!("[mm] xdna: pdi_size={} instr.xdna_addr={:#x} instr_size={}",
             MATMUL_512_4C_PDI.len(), instr_bo.xdna_addr, MATMUL_512_4C_INSTS.len());
    println!("[mm] BOs: A={a_va:#x} ({A_BYTES} B = {} KiB)", A_BYTES / 1024);
    println!("[mm]      B={b_va:#x} ({B_BYTES} B = {} KiB)", B_BYTES / 1024);
    println!("[mm]      C={c_va:#x} ({C_BYTES} B = {} KiB)", C_BYTES / 1024);

    // Cmd packet
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

    // Single dispatch
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

    // Verify against host scalar matmul (sample only — full O(MNK) host
    // matmul on 512^3 is ~5 sec in scalar). Check first 32 outputs.
    let outp = c_bo.map().expect("C re-map");
    let mut errors = 0usize;
    let mut first_bad = None;
    let n_check = 32;
    for r in 0..n_check {
        for c in 0..n_check {
            let mut want: i32 = 0;
            for kk in 0..MATMUL_512_4C_K {
                want += a_host[r * MATMUL_512_4C_K + kk] as i32 *
                        b_host[kk * MATMUL_512_4C_N + c] as i32;
            }
            let off = (r * MATMUL_512_4C_N + c) * 4;
            let bytes: [u8; 4] = outp[off..off + 4].try_into().unwrap();
            let got = i32::from_le_bytes(bytes);
            if got != want {
                if first_bad.is_none() { first_bad = Some((r, c, want, got)); }
                errors += 1;
            }
        }
    }
    if errors == 0 {
        let macs = 2.0 * MATMUL_512_4C_M as f64 * MATMUL_512_4C_K as f64 *
                   MATMUL_512_4C_N as f64;
        let gops = macs / (single_us as f64 / 1e6) / 1e9;
        println!("[mm] CORRECTNESS PASS — first {n_check}×{n_check} block matches host ref");
        println!("[mm] single dispatch: {single_us} us → {gops:.2} GOp/s on {} MACs",
                 macs as u64);
    } else {
        println!("[mm] CORRECTNESS FAIL: {errors}/{} mismatches in first {n_check}×{n_check}; first {first_bad:?}",
                 n_check * n_check);
        // dump first row 8 outputs
        for c in 0..8 {
            let off = c * 4;
            let bytes: [u8; 4] = outp[off..off + 4].try_into().unwrap();
            println!("    C[0,{c}] = {}", i32::from_le_bytes(bytes));
        }
        return ExitCode::FAILURE;
    }

    // Perf microbench: 50 iters with state reset between
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
    let macs = 2.0 * MATMUL_512_4C_M as f64 * MATMUL_512_4C_K as f64 *
               MATMUL_512_4C_N as f64;
    let gops = macs / (mean_us as f64 / 1e6) / 1e9;
    println!("[mm] perf: mean={mean_us}us max={max_us}us → {gops:.2} GOp/s ({n_iter} iters, {} MMACs)",
             (macs / 2.0 / 1e6) as u64);

    ExitCode::SUCCESS
}
