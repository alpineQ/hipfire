//! Phase 1.3b — pure-hipx passthrough end-to-end test.
//!
//! Replicates AMD's `passthrough_pykernel` test through hipx ioctls:
//!  1. Hipx::open (also allocates the per-client DEV_HEAP)
//!  2. CREATE_HWCTX (8 cols, max_opc=2048, log_buf=none)
//!  3. Alloc DEV BO for the embedded PDI bytes; copy in.
//!  4. CONFIG_HWCTX(CU) with the PDI BO at CU index 0.
//!  5. Alloc DEV BO for npu_insts; copy in (75 dwords).
//!  6. Alloc SHMEM input BO (4096 B); fill `(i & 0xff)` pattern.
//!  7. Alloc SHMEM output BO (4096 B); zero.
//!  8. Alloc CMD BO (4096 B). Build ERT_START_NPU packet:
//!       cu_mask = 0x1
//!       npu_data: instr_addr = npu_insts.xdna_addr
//!                 instr_size = bytes
//!                 prop_count = 0
//!       arg 0x00 opcode = 3
//!       arg 0x08 instr_ptr = npu_insts.xdna_addr  (some flows duplicate)
//!       arg 0x10 ninstr (dwords)
//!       arg 0x14 bo0 = input.xdna_addr
//!       arg 0x1C bo1 = output.xdna_addr
//!       (bo2..bo4 = 0)
//!  9. EXEC_CMD; receive `seq` (firmware sequence).
//! 10. SYNCOBJ_TIMELINE_WAIT(syncobj=hwctx.syncobj_handle, point=seq).
//! 11. Read output bytes, compare to input. PASS if all match.

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::SYNC_TO_DEVICE;
use hipx::kernels::{
    passthrough_4k_args as args, PASSTHROUGH_4K_COLUMNS, PASSTHROUGH_4K_INSTS,
    PASSTHROUGH_4K_OPS_PER_CYCLE, PASSTHROUGH_4K_PDI,
};
use hipx::Hipx;

const SIZE: usize = 4096;

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Hipx::open: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[passthrough] device: AIE {}.{}, {} cols, fw {}.{}.{} build {}",
        hipx.info.aie_version.0,
        hipx.info.aie_version.1,
        hipx.info.aie_cols,
        hipx.info.firmware_version.0,
        hipx.info.firmware_version.1,
        hipx.info.firmware_version.2,
        hipx.info.firmware_version.3
    );
    println!(
        "[passthrough] DEV_HEAP allocated: handle={} size={}MB xdna_addr={:#x}",
        hipx.heap.handle,
        hipx.heap.size / (1024 * 1024),
        hipx.heap.xdna_addr
    );

    // hwctx — full 8-column partition, ops/cycle from kernel metadata.
    let mut b = HwctxBuilder::default();
    b.num_columns = PASSTHROUGH_4K_COLUMNS;
    b.max_opc = PASSTHROUGH_4K_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create_hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[passthrough] hwctx: handle={} syncobj={} doorbell={:#x}",
        ctx.handle, ctx.syncobj_handle, ctx.umq_doorbell
    );

    // (3) PDI BO — DEV-typed (lives in the heap; access via heap mmap)
    let pdi_bo = match hipx.alloc_dev(PASSTHROUGH_4K_PDI.len()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_dev(pdi): {e}");
            return ExitCode::FAILURE;
        }
    };
    unsafe {
        let buf = match hipx.dev_slice(&pdi_bo) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("dev_slice(pdi): {e}");
                return ExitCode::FAILURE;
            }
        };
        buf[..PASSTHROUGH_4K_PDI.len()].copy_from_slice(PASSTHROUGH_4K_PDI);
    }
    if let Err(e) = pdi_bo.sync(SYNC_TO_DEVICE) {
        eprintln!("sync(pdi): {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "[passthrough] PDI BO: handle={} size={} xdna_addr={:#x}",
        pdi_bo.handle, pdi_bo.size, pdi_bo.xdna_addr
    );

    // (4) Bind the PDI as CU index 0 on this hwctx
    let _cu_binding = match config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("config_cus: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[passthrough] CONFIG_HWCTX(CU) ok — 1 CU bound");

    // (5) NPU instruction stream — DEV-typed (via heap)
    let instr_bo = match hipx.alloc_dev(PASSTHROUGH_4K_INSTS.len()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_dev(instr): {e}");
            return ExitCode::FAILURE;
        }
    };
    unsafe {
        let buf = match hipx.dev_slice(&instr_bo) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("dev_slice(instr): {e}");
                return ExitCode::FAILURE;
            }
        };
        buf[..PASSTHROUGH_4K_INSTS.len()].copy_from_slice(PASSTHROUGH_4K_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (PASSTHROUGH_4K_INSTS.len() / 4) as u32;
    println!(
        "[passthrough] instr BO: handle={} size={} xdna_addr={:#x} ndwords={ninstr_dwords}",
        instr_bo.handle, instr_bo.size, instr_bo.xdna_addr
    );

    // (6,7) input/output BOs — SHMEM (host-pinned)
    let mut input_bo = match hipx.alloc_shmem(SIZE) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_shmem(input): {e}");
            return ExitCode::FAILURE;
        }
    };
    {
        let buf = input_bo.map().expect("map(input)");
        for (i, b) in buf[..SIZE].iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
    }
    let _ = input_bo.sync(SYNC_TO_DEVICE);

    let mut output_bo = match hipx.alloc_shmem(SIZE) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_shmem(output): {e}");
            return ExitCode::FAILURE;
        }
    };
    {
        let buf = output_bo.map().expect("map(output)");
        // Sentinel to detect "NPU wrote nothing" vs "NPU wrote zeros"
        for b in buf[..SIZE].iter_mut() {
            *b = 0xAB;
        }
    }
    let _ = output_bo.sync(SYNC_TO_DEVICE);

    let input_dev_addr = input_bo.host_ptr().expect("input mapped") as u64;
    let output_dev_addr = output_bo.host_ptr().expect("output mapped") as u64;
    println!(
        "[passthrough] input BO: handle={} host_va={:#x} (xdna_addr={:#x})",
        input_bo.handle, input_dev_addr, input_bo.xdna_addr
    );
    println!(
        "[passthrough] output BO: handle={} host_va={:#x} (xdna_addr={:#x})",
        output_bo.handle, output_dev_addr, output_bo.xdna_addr
    );

    // (8) CMD BO — build the ert_start_kernel_cmd packet
    let mut cmd_bo = match hipx.alloc_cmd(4096) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_cmd: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cmd_size = {
        let cbuf = cmd_bo.map().expect("map(cmd)");
        // AMD test (verified via in-kernel printk of cmd_bo) uses
        // ERT_START_CU, NOT ERT_START_NPU — no npu_data prefix; args
        // begin immediately after cu_mask.
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        // SHMEM BOs are HOST-resident; with PASID the NPU sees the
        // same VA as the host. Pass the user-VA (mmap pointer), not
        // xdna_addr (which is INVALID for unattached SHMEM).
        eb.set_arg_u64(args::BO0, input_dev_addr);
        eb.set_arg_u64(args::BO1, output_dev_addr);
        // bo2..bo4 left zero — the kernel JSON declares 5 BOs but
        // the passthrough flow only uses 2.
        let total = eb.finalize(0x3C); // 0x34 + 8 bytes for last u64
        // Dump packet in hex for debugging
        eprint!("[passthrough] cmd packet ({total} bytes):");
        for (i, dword) in cbuf[..total].chunks(4).enumerate() {
            if i % 4 == 0 {
                eprint!("\n  {:02x}:", i * 4);
            }
            let v = u32::from_le_bytes(dword.try_into().unwrap_or([0; 4]));
            eprint!(" {v:08x}");
        }
        eprintln!();
        total
    };
    let _ = cmd_size;
    let _ = cmd_bo.sync(SYNC_TO_DEVICE);

    // (9) Submit. args[] is the BO list firmware needs to pin.
    let seq = match submit_exec_cmd(
        hipx.device.fd,
        &ctx,
        &[&cmd_bo],
        &[&instr_bo, &input_bo, &output_bo],
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("submit_exec_cmd: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[passthrough] EXEC_CMD seq={seq}");

    // (10) Timeline wait at the returned sequence (point=seq exactly,
    // matching aie2_ctx.c's drm_syncobj_add_point(syncobj, chain, fence, seq)).
    timeline_wait(hipx.device.fd, ctx.syncobj_handle, seq,
                  Duration::from_secs(5)).expect("timeline_wait");
    println!("[passthrough] syncobj signaled at point={seq}");

    // (11) verify
    let _ = output_bo.sync(hipx::ioctl::SYNC_FROM_DEVICE);
    let inp = input_bo.map().expect("re-map(input)").to_vec();
    let outp = output_bo.map().expect("re-map(output)");

    let mut hist = [0u32; 256];
    for &b in &outp[..SIZE] {
        hist[b as usize] += 1;
    }
    let mut top: Vec<(u8, u32)> = (0u8..=255u8)
        .map(|v| (v, hist[v as usize]))
        .filter(|(_, c)| *c > 0)
        .collect();
    top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    eprint!("[passthrough] output histogram (top 5): ");
    for (v, c) in top.iter().take(5) {
        eprint!("{v:#04x}={c} ");
    }
    eprintln!();
    eprintln!(
        "[passthrough] output first 16 bytes: {:02x?}",
        &outp[..16]
    );

    let mut errors = 0;
    let mut first_bad = None;
    for i in 0..SIZE {
        if outp[i] != inp[i] {
            if first_bad.is_none() {
                first_bad = Some((i, inp[i], outp[i]));
            }
            errors += 1;
        }
    }
    if errors == 0 {
        println!("[passthrough] PASS ({SIZE} bytes round-tripped)");
        ExitCode::SUCCESS
    } else {
        println!(
            "[passthrough] FAIL: {errors} mismatches; first {:?}",
            first_bad
        );
        ExitCode::FAILURE
    }
}
