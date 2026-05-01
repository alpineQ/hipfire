//! `hipx-matvec` — first GEMM-class kernel proof. Dispatches an
//! i16×i16→i32 matrix-vector multiply (288×288 × 288 → 288) to the
//! AIE-2P NPU and verifies output against a host scalar matvec.
//!
//! Architecturally this is the first kernel that is a real building
//! block for inference workloads: speculative-decode draft heads and
//! the asym KV-codec fold both decompose into GEMV calls of this
//! shape (or scaled versions of it).

use std::process::ExitCode;
use std::time::{Duration, Instant};

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::kernels::{
    matvec_288x288_args as args, MATVEC_288X288_COLUMNS, MATVEC_288X288_INSTS,
    MATVEC_288X288_K, MATVEC_288X288_M, MATVEC_288X288_OPS_PER_CYCLE, MATVEC_288X288_PDI,
};
use hipx::Hipx;

const A_BYTES: usize = MATVEC_288X288_M * MATVEC_288X288_K * 2; // i16
const B_BYTES: usize = MATVEC_288X288_K * 2;                    // i16
const C_BYTES: usize = MATVEC_288X288_M * 4;                    // i32

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };
    println!("[matvec] device: AIE {}.{}, fw {}.{}.{} build {}",
             hipx.info.aie_version.0, hipx.info.aie_version.1,
             hipx.info.firmware_version.0, hipx.info.firmware_version.1,
             hipx.info.firmware_version.2, hipx.info.firmware_version.3);

    let mut b = HwctxBuilder::default();
    b.num_columns = MATVEC_288X288_COLUMNS;
    b.max_opc = MATVEC_288X288_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };
    println!("[matvec] hwctx={} cols={} M={} K={}",
             ctx.handle, MATVEC_288X288_COLUMNS, MATVEC_288X288_M, MATVEC_288X288_K);

    // PDI BO (DEV)
    let pdi_bo = hipx.alloc_dev(MATVEC_288X288_PDI.len()).expect("pdi alloc");
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..MATVEC_288X288_PDI.len()].copy_from_slice(MATVEC_288X288_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    // Hold the CuBinding alive — dropping it closes the PDI BO and
    // the firmware loses its CU reference. (See matmul_512 for the
    // root cause writeup — `let _ = config_cus(...)` was the latent
    // bug across all our binaries; the only ones that worked did so
    // because their PDI was small enough that the subsequent instr
    // BO landed inside the freed page and the firmware got valid
    // bytes by accident.)
    let _cu = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    // Instruction stream (DEV)
    let instr_bo = hipx.alloc_dev(MATVEC_288X288_INSTS.len()).expect("instr alloc");
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..MATVEC_288X288_INSTS.len()].copy_from_slice(MATVEC_288X288_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (MATVEC_288X288_INSTS.len() / 4) as u32;

    // A — M×K i16. Matrix is what aie2 mv.cc expects: 32-bit-word
    // transposed at granularity of 4 bytes (see mv.cc comment block).
    // For the host reference we'll compute the equivalent transposed
    // form, then write it. To keep the bench simple, fill A and B with
    // small deterministic values whose product fits in i32 cleanly.
    let mut a_bo = hipx.alloc_shmem(A_BYTES).expect("A alloc");
    let mut a_host = vec![0i16; MATVEC_288X288_M * MATVEC_288X288_K];
    for r in 0..MATVEC_288X288_M {
        for k in 0..MATVEC_288X288_K {
            // small values so M·K accumulation stays in i32 range
            let v = ((r + k) as i16) & 0x7;
            a_host[r * MATVEC_288X288_K + k] = v;
        }
    }
    {
        // Transpose A row-major → 32-bit-word transposed layout.
        // For i16 (2 bytes), 32-bit word = 2 elements. mv.cc wants
        // pairs of (row, row+1) interleaved at 4-byte granularity.
        // See mv.cc comment: "1 2 9 10 17 18 / 3 4 11 12 19 / ..."
        // Layout: groups of 8 rows × 2 cols = 16 i16 = 32 bytes,
        // sweeping K by 2.
        let buf = a_bo.map().expect("A map");
        // Naive write: row-major. We'll see if the kernel accepts
        // it directly first; if not, transpose per the comment.
        for r in 0..MATVEC_288X288_M {
            for k in 0..MATVEC_288X288_K {
                let v = a_host[r * MATVEC_288X288_K + k];
                let off = (r * MATVEC_288X288_K + k) * 2;
                buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    let _ = a_bo.sync(SYNC_TO_DEVICE);
    let a_va = a_bo.host_ptr().unwrap() as u64;

    // B — K i16
    let mut b_bo = hipx.alloc_shmem(B_BYTES).expect("B alloc");
    let mut b_host = vec![0i16; MATVEC_288X288_K];
    for k in 0..MATVEC_288X288_K {
        b_host[k] = (k as i16) & 0x7;
    }
    {
        let buf = b_bo.map().expect("B map");
        for k in 0..MATVEC_288X288_K {
            buf[k * 2..k * 2 + 2].copy_from_slice(&b_host[k].to_le_bytes());
        }
    }
    let _ = b_bo.sync(SYNC_TO_DEVICE);
    let b_va = b_bo.host_ptr().unwrap() as u64;

    // C — output, sentinel
    let mut c_bo = hipx.alloc_shmem(C_BYTES).expect("C alloc");
    {
        let buf = c_bo.map().expect("C map");
        for byte in buf[..C_BYTES].iter_mut() { *byte = 0xCC; }
    }
    let _ = c_bo.sync(SYNC_TO_DEVICE);
    let c_va = c_bo.host_ptr().unwrap() as u64;

    // bo3, bo4 placeholders — Worker-class needs all 5 slots filled.
    let mut bo3_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 8).expect("bo3 alloc");
    {
        let buf = bo3_bo.map().expect("bo3 map");
        for b in buf.iter_mut() { *b = 0; }
    }
    let _ = bo3_bo.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3_bo.host_ptr().unwrap() as u64;
    let mut bo4_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 1).expect("bo4 alloc");
    {
        let buf = bo4_bo.map().expect("bo4 map");
        for b in buf.iter_mut() { *b = 0; }
    }
    let _ = bo4_bo.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4_bo.host_ptr().unwrap() as u64;

    println!("[matvec] BOs: A={a_va:#x} ({A_BYTES} B) B={b_va:#x} ({B_BYTES} B) C={c_va:#x} ({C_BYTES} B)");

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

    // Single dispatch first to verify correctness, then a perf loop.
    let t0 = Instant::now();
    let seq = match submit_exec_cmd(
        hipx.device.fd, &ctx, &[&cmd_bo],
        &[&instr_bo, &a_bo, &b_bo, &c_bo, &bo3_bo, &bo4_bo],
    ) {
        Ok(s) => s, Err(e) => { eprintln!("submit: {e}"); return ExitCode::FAILURE; }
    };
    if let Err(e) = timeline_wait(hipx.device.fd, ctx.syncobj_handle, seq,
                                  Duration::from_secs(5)) {
        eprintln!("timeline_wait: {e}");
        return ExitCode::FAILURE;
    }
    let single_us = t0.elapsed().as_micros();
    let _ = c_bo.sync(SYNC_FROM_DEVICE);

    // Verify
    let outp = c_bo.map().expect("C re-map");
    let mut errors = 0usize;
    let mut first_bad = None;
    for r in 0..MATVEC_288X288_M {
        let mut want: i32 = 0;
        for k in 0..MATVEC_288X288_K {
            want += a_host[r * MATVEC_288X288_K + k] as i32 * b_host[k] as i32;
        }
        let got_bytes: [u8; 4] = outp[r * 4..r * 4 + 4].try_into().unwrap();
        let got = i32::from_le_bytes(got_bytes);
        if got != want {
            if first_bad.is_none() { first_bad = Some((r, want, got)); }
            errors += 1;
        }
    }
    if errors == 0 {
        println!("[matvec] CORRECTNESS PASS — 288 dot-products match host reference");
        println!("[matvec] single dispatch latency: {single_us} us");
    } else {
        println!("[matvec] CORRECTNESS FAIL: {errors}/{} mismatches; first {first_bad:?}",
                 MATVEC_288X288_M);
        // dump first 8 outputs for sanity
        for r in 0..8 {
            let bytes: [u8; 4] = outp[r * 4..r * 4 + 4].try_into().unwrap();
            let got = i32::from_le_bytes(bytes);
            println!("    [{r}] = {got}");
        }
        return ExitCode::FAILURE;
    }

    // Perf microbench: 100 iters back-to-back, single-kernel.
    let n_iter: u32 = 100;
    // reset cmd state on first iter; we have to do it each iter
    // because firmware writes COMPLETED into the packet.
    let mut total_us: u128 = 0;
    let mut max_us: u128 = 0;
    for _ in 0..n_iter {
        // cmd state reset
        {
            let cbuf = cmd_bo.map().expect("cmd reset");
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);
        let t = Instant::now();
        let seq = submit_exec_cmd(hipx.device.fd, &ctx, &[&cmd_bo],
            &[&instr_bo, &a_bo, &b_bo, &c_bo, &bo3_bo, &bo4_bo]).expect("submit");
        timeline_wait(hipx.device.fd, ctx.syncobj_handle, seq,
                      Duration::from_secs(5)).expect("wait");
        let us = t.elapsed().as_micros();
        total_us += us;
        if us > max_us { max_us = us; }
    }
    let mean_us = total_us / n_iter as u128;
    // 2 * M * K MACs per matvec
    let macs = 2.0 * MATVEC_288X288_M as f64 * MATVEC_288X288_K as f64;
    let mean_s = mean_us as f64 / 1e6;
    let gops = macs / mean_s / 1e9;
    println!("[matvec] perf: mean={mean_us}us max={max_us}us → {gops:.2} GOp/s ({n_iter} iters)");

    ExitCode::SUCCESS
}
