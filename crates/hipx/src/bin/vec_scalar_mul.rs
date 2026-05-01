//! `hipx-vec-scalar-mul` — second-kernel smoke test demonstrating
//! real NPU compute. Multiplies 4096 × i16 by an i32 scalar via the
//! AIE-2P-compiled MLIR-AIE example kernel; verifies output[i] ==
//! input[i] * scale[0] on the host.

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::kernels::{
    vec_scalar_mul_args as args, VEC_SCALAR_MUL_COLUMNS, VEC_SCALAR_MUL_INSTS,
    VEC_SCALAR_MUL_OPS_PER_CYCLE, VEC_SCALAR_MUL_PDI,
};
use hipx::Hipx;

const N_ELEMS: usize = 4096;
const INPUT_BYTES: usize = N_ELEMS * 2; // i16
const SCALE_BYTES: usize = 4; // i32
const OUTPUT_BYTES: usize = N_ELEMS * 2;
const SCALE: i32 = 7;

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };
    println!("[vsm] device: AIE {}.{}, fw {}.{}.{} build {}",
             hipx.info.aie_version.0, hipx.info.aie_version.1,
             hipx.info.firmware_version.0, hipx.info.firmware_version.1,
             hipx.info.firmware_version.2, hipx.info.firmware_version.3);

    let mut b = HwctxBuilder::default();
    b.num_columns = VEC_SCALAR_MUL_COLUMNS;
    b.max_opc = VEC_SCALAR_MUL_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };
    println!("[vsm] hwctx={} syncobj={}", ctx.handle, ctx.syncobj_handle);

    // PDI BO (DEV) — copy via heap mmap
    let pdi_bo = match hipx.alloc_dev(VEC_SCALAR_MUL_PDI.len()) {
        Ok(b) => b, Err(e) => { eprintln!("pdi alloc: {e}"); return ExitCode::FAILURE; }
    };
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..VEC_SCALAR_MUL_PDI.len()].copy_from_slice(VEC_SCALAR_MUL_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    // Hold the CuBinding alive — dropping it closes the PDI BO, which
    // is fatal for kernels whose PDI > one heap page (matmul_512). For
    // small PDIs (this kernel, passthrough_dmas) it just happens to
    // work because instr lands inside the freed region.
    let _cu = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");
    println!("[vsm] CU bound");

    // (No pad BO — earlier "instr must land at heap+0x8000" theory
    // was a misdiagnosis of the dropped-CuBinding bug fixed in 918c3b7.
    // Holding _cu alive keeps the PDI BO open; instr lands wherever
    // DEV_HEAP next allocates and the firmware reads it correctly.)

    // Instruction stream (DEV)
    let instr_bo = match hipx.alloc_dev(VEC_SCALAR_MUL_INSTS.len()) {
        Ok(b) => b, Err(e) => { eprintln!("instr alloc: {e}"); return ExitCode::FAILURE; }
    };
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..VEC_SCALAR_MUL_INSTS.len()].copy_from_slice(VEC_SCALAR_MUL_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (VEC_SCALAR_MUL_INSTS.len() / 4) as u32;

    // Input — 4096 × i16, fill with i (mod 256) so output is predictable
    let mut input_bo = hipx.alloc_shmem(INPUT_BYTES).expect("input alloc");
    {
        let buf = input_bo.map().expect("input map");
        for i in 0..N_ELEMS {
            let v = (i & 0xFF) as i16;
            buf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    let _ = input_bo.sync(SYNC_TO_DEVICE);
    let input_va = input_bo.host_ptr().unwrap() as u64;

    // Scale — 1 × i32
    let mut scale_bo = hipx.alloc_shmem(SCALE_BYTES).expect("scale alloc");
    {
        let buf = scale_bo.map().expect("scale map");
        buf[..4].copy_from_slice(&SCALE.to_le_bytes());
    }
    let _ = scale_bo.sync(SYNC_TO_DEVICE);
    let scale_va = scale_bo.host_ptr().unwrap() as u64;

    // Output — sentinel pattern so we can detect "NPU wrote nothing"
    let mut output_bo = hipx.alloc_shmem(OUTPUT_BYTES).expect("output alloc");
    {
        let buf = output_bo.map().expect("output map");
        for b in buf[..OUTPUT_BYTES].iter_mut() { *b = 0xCC; }
    }
    let _ = output_bo.sync(SYNC_TO_DEVICE);
    let output_va = output_bo.host_ptr().unwrap() as u64;

    // CRITICAL: AMD's vec_scalar_mul XRT-test cmd packet has all 5 BO
    // slots (bo0..bo4) filled with real host VAs — even bo3/bo4
    // (ctrlpkts/trace) get placeholder buffers. Leaving them at 0 is
    // why our previous attempt failed: the firmware completes the job
    // but the worker tile never writes output. Verified by capturing
    // AMD's working cmd_bo via a kernel printk in amdxdna_cmd_submit.
    // AMD strace shows XRT requests EXACT sizes for ctrlpkts/trace:
    // ctrlpkts = 8 bytes, trace = 1 byte (kernel still page-aligns
    // the allocation). The firmware may validate against requested
    // size, not aligned size. Use alloc_shmem_exact to pass through
    // the raw size in CREATE_BO.
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

    println!("[vsm] BOs: input={input_va:#x} scale={scale_va:#x} output={output_va:#x} bo3={bo3_va:#x} bo4={bo4_va:#x}");

    // Cmd packet
    let mut cmd_bo = hipx.alloc_cmd(4096).expect("cmd alloc");
    {
        let cbuf = cmd_bo.map().expect("cmd map");
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::INPUT, input_va);
        eb.set_arg_u64(args::SCALE, scale_va);
        eb.set_arg_u64(args::OUTPUT, output_va);
        eb.set_arg_u64(args::BO3, bo3_va);
        eb.set_arg_u64(args::BO4, bo4_va);
        let _ = eb.finalize(0x3C);
    }

    // Run 50 iterations IN-PROCESS to test stability — engine integration
    // creates ONE NpuRuntime per process and reuses it across many ops.
    let mut total_pass = 0;
    let mut total_fail = 0;
    for iter in 0..50 {
        // Reset output to sentinel
        {
            let buf = output_bo.map().expect("output reset");
            for b in buf[..OUTPUT_BYTES].iter_mut() { *b = 0xCC; }
        }
        let _ = output_bo.sync(SYNC_TO_DEVICE);
        // Reset cmd packet state nibble to NEW — firmware skips re-execution
        // of a packet still marked COMPLETED from the prior submit.
        {
            let cbuf = cmd_bo.map().expect("cmd reset");
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);
        let seq = match submit_exec_cmd(
            hipx.device.fd, &ctx, &[&cmd_bo],
            &[&instr_bo, &input_bo, &scale_bo, &output_bo, &bo3_bo, &bo4_bo],
        ) {
            Ok(s) => s, Err(e) => { eprintln!("submit iter {iter}: {e}"); return ExitCode::FAILURE; }
        };
        // amdxdna registers fence at point=seq directly (kernel: aie2_ctx.c
        // does `drm_syncobj_add_point(syncobj, chain, fence, *seq)` where
        // *seq is the same value EXEC_CMD returns to userspace).
        if let Err(e) = timeline_wait(hipx.device.fd, ctx.syncobj_handle, seq,
                                      Duration::from_secs(5)) {
            eprintln!("[vsm] iter {iter} seq={seq} timeline_wait: {e}");
            return ExitCode::FAILURE;
        }

        let _ = output_bo.sync(SYNC_FROM_DEVICE);
        let outp = output_bo.map().expect("re-map output");
        let mut errors = 0;
        for i in 0..N_ELEMS {
            let want = ((i & 0xFF) as i16).wrapping_mul(SCALE as i16);
            let got_bytes: [u8; 2] = outp[i * 2..i * 2 + 2].try_into().unwrap();
            let got = i16::from_le_bytes(got_bytes);
            if got != want { errors += 1; }
        }
        if errors == 0 {
            total_pass += 1;
        } else {
            println!("[vsm] iter {iter} seq={seq} FAIL ({errors}/{N_ELEMS} mismatches)");
            total_fail += 1;
        }
    }
    println!("[vsm] in-process: {total_pass} PASS / {total_fail} FAIL");
    if total_fail == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
