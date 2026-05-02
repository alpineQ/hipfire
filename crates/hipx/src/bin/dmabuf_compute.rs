//! `dmabuf_compute` — exercise a real NPU kernel with a dmabuf-imported
//! input BO. Proves the full path: amdgpu GTT alloc → dmabuf → amdxdna
//! PRIME import → mmap → use as EXEC_CMD argument → AIE-2P kernel reads
//! it via PASID-translated CPU VA → correct output.
//!
//! Mirrors `vec_scalar_mul` but the input vector lives in an amdgpu
//! GTT BO (the path the engine's K cache will take). All other BOs
//! (scale, output, bo3/bo4 placeholders, cmd, instr, pdi) stay
//! amdxdna-native — we're isolating the dmabuf input variable.
//!
//! If this passes, the asym3 codec kernel can consume engine-resident
//! K cache via the same path: HSA-allocate the K cache (or amdgpu-
//! allocate directly), export dmabuf, import on NPU, pass the imported
//! handle in EXEC_CMD args + the imported BO's CPU VA in the cmd packet.

use std::os::fd::AsRawFd;
use std::process::ExitCode;
use std::time::Duration;

use hipx::agpu;
use hipx::cmd::config_cus;
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
const SCALE_BYTES: usize = 4;
const OUTPUT_BYTES: usize = N_ELEMS * 2;
const SCALE: i32 = 7;

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Hipx::open: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[dbc] device: AIE {}.{}, fw {}.{}.{} build {}",
        hipx.info.aie_version.0,
        hipx.info.aie_version.1,
        hipx.info.firmware_version.0,
        hipx.info.firmware_version.1,
        hipx.info.firmware_version.2,
        hipx.info.firmware_version.3
    );

    // 1. Allocate input via amdgpu (the iGPU side), export dmabuf, import
    //    on amdxdna.
    let agpu_fd_owned = match agpu::open_render_node(0) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open render node: {e}");
            return ExitCode::FAILURE;
        }
    };
    let agpu_fd = agpu_fd_owned.as_raw_fd();
    let (agpu_handle, dmabuf, agpu_ptr) =
        agpu::alloc_and_export(agpu_fd, INPUT_BYTES as u64).unwrap_or_else(|e| {
            eprintln!("amdgpu alloc/export: {e}");
            std::process::exit(1);
        });
    let dmabuf_fd = dmabuf.as_raw_fd();

    // Fill input via the amdgpu CPU mapping.
    let agpu_input = unsafe { std::slice::from_raw_parts_mut(agpu_ptr, INPUT_BYTES) };
    for i in 0..N_ELEMS {
        let v = (i & 0xFF) as i16;
        agpu_input[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
    }

    // Import the dmabuf into amdxdna using the first-class wrapper.
    // Bo::from_imported_dmabuf does PRIME_FD_TO_HANDLE + GET_BO_INFO;
    // .map() does the amdxdna-side mmap. Drop closes the handle.
    let mut input_bo =
        hipx::Bo::from_imported_dmabuf(hipx.device.fd, dmabuf_fd, INPUT_BYTES)
            .unwrap_or_else(|e| {
                eprintln!("Bo::from_imported_dmabuf: {e}");
                std::process::exit(1);
            });
    let _ = input_bo.map().unwrap_or_else(|e| {
        eprintln!("input_bo.map: {e}");
        std::process::exit(1);
    });
    let input_va_npu = input_bo.host_ptr().unwrap() as u64;
    println!(
        "[dbc] amdgpu GTT input: handle={agpu_handle} cpu_va={agpu_ptr:?}; \
         dmabuf imported on NPU: handle={} cpu_va={input_va_npu:#x}",
        input_bo.handle
    );

    // 2. Standard hwctx + PDI + instr + scale + output + placeholder BOs
    //    (mirrors vec_scalar_mul exactly).
    let mut b = HwctxBuilder::default();
    b.num_columns = VEC_SCALAR_MUL_COLUMNS;
    b.max_opc = VEC_SCALAR_MUL_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create_hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };

    let pdi_bo = match hipx.alloc_dev(VEC_SCALAR_MUL_PDI.len()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("pdi alloc: {e}");
            return ExitCode::FAILURE;
        }
    };
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..VEC_SCALAR_MUL_PDI.len()].copy_from_slice(VEC_SCALAR_MUL_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    let instr_bo = match hipx.alloc_dev(VEC_SCALAR_MUL_INSTS.len()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("instr alloc: {e}");
            return ExitCode::FAILURE;
        }
    };
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..VEC_SCALAR_MUL_INSTS.len()].copy_from_slice(VEC_SCALAR_MUL_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (VEC_SCALAR_MUL_INSTS.len() / 4) as u32;

    let mut scale_bo = hipx.alloc_shmem(SCALE_BYTES).expect("scale alloc");
    {
        let buf = scale_bo.map().expect("scale map");
        buf[..4].copy_from_slice(&SCALE.to_le_bytes());
    }
    let _ = scale_bo.sync(SYNC_TO_DEVICE);
    let scale_va = scale_bo.host_ptr().unwrap() as u64;

    let mut output_bo = hipx.alloc_shmem(OUTPUT_BYTES).expect("output alloc");
    {
        let buf = output_bo.map().expect("output map");
        for b in buf[..OUTPUT_BYTES].iter_mut() {
            *b = 0xCC;
        }
    }
    let _ = output_bo.sync(SYNC_TO_DEVICE);
    let output_va = output_bo.host_ptr().unwrap() as u64;

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

    let mut cmd_bo = hipx.alloc_cmd(4096).expect("cmd alloc");
    {
        let cbuf = cmd_bo.map().expect("cmd map");
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::INPUT, input_va_npu); // ← from dmabuf-imported BO
        eb.set_arg_u64(args::SCALE, scale_va);
        eb.set_arg_u64(args::OUTPUT, output_va);
        eb.set_arg_u64(args::BO3, bo3_va);
        eb.set_arg_u64(args::BO4, bo4_va);
        let _ = eb.finalize(0x3C);
    }

    // The imported BO is now a hipx::Bo, so it goes through the
    // standard submit_exec_cmd path. Drop semantics: Bo::Drop does
    // GEM_CLOSE on the imported handle.

    // Test which input-VA flavour the firmware accepts. Both should
    // work (same physical pages); kept as a regression check.
    let amdgpu_va = agpu_ptr as u64;
    let npu_va = input_va_npu;

    println!("[dbc] testing both VA flavours through Bo wrapper...");

    let mut total_pass = 0;
    let mut total_fail = 0;
    for iter in 0..10 {
        // Variation: even iters use amdgpu CPU VA, odd use amdxdna mmap VA.
        // First pass also tests SYNC_BO TO_DEVICE on the imported handle.
        let use_amdgpu_va = iter % 2 == 0;
        let chosen_va = if use_amdgpu_va { amdgpu_va } else { npu_va };
        // Re-pack the cmd packet's INPUT field for this iteration.
        {
            let cbuf = cmd_bo.map().expect("cmd remap");
            let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
            eb.set_cu_mask(0x1);
            eb.set_arg_u64(args::OPCODE, 3);
            eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
            eb.set_arg_u32(args::NINSTR, ninstr_dwords);
            eb.set_arg_u64(args::INPUT, chosen_va);
            eb.set_arg_u64(args::SCALE, scale_va);
            eb.set_arg_u64(args::OUTPUT, output_va);
            eb.set_arg_u64(args::BO3, bo3_va);
            eb.set_arg_u64(args::BO4, bo4_va);
            let _ = eb.finalize(0x3C);
        }

        // Re-write the input pattern via amdgpu mapping every iter so
        // we know it's fresh.
        for i in 0..N_ELEMS {
            let v = (i & 0xFF) as i16;
            agpu_input[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        let sync_ret = match input_bo.sync(SYNC_TO_DEVICE) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("input_bo.sync failed: {e}");
                return ExitCode::FAILURE;
            }
        };

        {
            let buf = output_bo.map().expect("output reset");
            for b in buf[..OUTPUT_BYTES].iter_mut() {
                *b = 0xCC;
            }
        }
        let _ = output_bo.sync(SYNC_TO_DEVICE);
        {
            let cbuf = cmd_bo.map().expect("cmd reset");
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);

        let seq = match hipx::cmd::submit_exec_cmd(
            hipx.device.fd,
            &ctx,
            &[&cmd_bo],
            &[&instr_bo, &input_bo, &scale_bo, &output_bo, &bo3_bo, &bo4_bo],
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("iter {iter} submit: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = timeline_wait(
            hipx.device.fd,
            ctx.syncobj_handle,
            seq,
            Duration::from_secs(5),
        ) {
            eprintln!("iter {iter} seq={seq} wait: {e}");
            return ExitCode::FAILURE;
        }

        let _ = output_bo.sync(SYNC_FROM_DEVICE);
        let outp = output_bo.map().expect("re-map output");
        let mut errors = 0;
        for i in 0..N_ELEMS {
            let want = ((i & 0xFF) as i16).wrapping_mul(SCALE as i16);
            let got_bytes: [u8; 2] = outp[i * 2..i * 2 + 2].try_into().unwrap();
            let got = i16::from_le_bytes(got_bytes);
            if got != want {
                errors += 1;
            }
        }
        let va_label = if use_amdgpu_va { "amdgpu_va" } else { "npu_va" };
        if errors == 0 {
            total_pass += 1;
            println!(
                "[dbc] iter {iter} seq={seq} PASS  ({va_label}, sync_ret={sync_ret})"
            );
        } else {
            total_fail += 1;
            println!(
                "[dbc] iter {iter} seq={seq} FAIL  ({va_label}, sync_ret={sync_ret}, {errors}/{N_ELEMS} mismatches)"
            );
        }
    }

    // Cleanup. input_bo's Drop closes the imported handle and munmaps
    // its amdxdna-side mapping. Manually unmap + close the amdgpu side.
    drop(input_bo);
    unsafe {
        libc::munmap(agpu_ptr as *mut libc::c_void, INPUT_BYTES);
    }
    let _ = agpu::gem_close(agpu_fd, agpu_handle);

    println!(
        "[dbc] {} passed / {} failed across 10 iterations",
        total_pass, total_fail
    );
    if total_fail == 0 {
        println!("[dbc] === DMABUF AS KERNEL ARG: PASS ===");
        ExitCode::SUCCESS
    } else {
        println!("[dbc] === DMABUF AS KERNEL ARG: FAIL ===");
        ExitCode::FAILURE
    }
}
