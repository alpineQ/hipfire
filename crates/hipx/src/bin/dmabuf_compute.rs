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
use hipx::ioctl::{
    drm_ioctl_amdxdna_get_bo_info, drm_ioctl_amdxdna_sync_bo, DrmGetBoInfo, DrmSyncBo,
    SYNC_FROM_DEVICE, SYNC_TO_DEVICE,
};
use hipx::kernels::{
    vec_scalar_mul_args as args, VEC_SCALAR_MUL_COLUMNS, VEC_SCALAR_MUL_INSTS,
    VEC_SCALAR_MUL_OPS_PER_CYCLE, VEC_SCALAR_MUL_PDI,
};
use hipx::prime::import_fd_to_handle;
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

    // Import the dmabuf into amdxdna and get an mmap'd CPU view of the
    // *same* physical pages. We need a CPU VA that's valid in this
    // process (it is: amdgpu_ptr is process-local already), but going
    // through the amdxdna fd's mapping ensures the kernel records the
    // PASID enrollment for this BO from the NPU side too.
    let npu_handle = match import_fd_to_handle(hipx.device.fd, dmabuf_fd) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("PRIME_FD_TO_HANDLE: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut info = DrmGetBoInfo {
        handle: npu_handle,
        ..Default::default()
    };
    let ret = unsafe {
        libc::ioctl(
            hipx.device.fd,
            drm_ioctl_amdxdna_get_bo_info(),
            &mut info as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        eprintln!("GET_BO_INFO on imported handle failed: errno={}",
                  std::io::Error::last_os_error().raw_os_error().unwrap_or(0));
        return ExitCode::FAILURE;
    }
    let npu_input_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            INPUT_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            hipx.device.fd,
            info.map_offset as libc::off_t,
        )
    };
    if npu_input_ptr == libc::MAP_FAILED {
        eprintln!("mmap NPU side at imported map_offset: {}",
                  std::io::Error::last_os_error());
        return ExitCode::FAILURE;
    }
    let input_va_npu = npu_input_ptr as u64;
    println!(
        "[dbc] amdgpu GTT input: handle={agpu_handle} cpu_va={agpu_ptr:?}; \
         dmabuf imported on NPU: handle={npu_handle} cpu_va={npu_input_ptr:?}"
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

    // We need an `&Bo` for the EXEC_CMD args list, but the imported BO
    // isn't a hipx::Bo (it has no Drop semantics tied to amdxdna's
    // CREATE_BO path). Build a minimal proxy: a struct with the right
    // shape — but the cleaner path is to just wrap the imported handle
    // in something that implements the .handle field that submit_exec_cmd
    // reads. For this PoC we'll inline the EXEC_CMD here.
    //
    // (Alternative is to add an `import_dmabuf_as_bo` helper to the Bo
    // type. We'll do that once dmabuf-as-arg is proven to work.)

    use hipx::ioctl::{drm_ioctl_amdxdna_exec_cmd, DrmExecCmd, CMD_SUBMIT_EXEC_BUF};
    let cmd_handles = vec![cmd_bo.handle];
    let arg_handles = vec![
        instr_bo.handle,
        npu_handle, // ← imported dmabuf BO handle
        scale_bo.handle,
        output_bo.handle,
        bo3_bo.handle,
        bo4_bo.handle,
    ];

    // Helper: SYNC_BO on the dmabuf-imported handle. Existing
    // hipx::Bo::sync requires a hipx::Bo; we do it inline here.
    let sync_imported = |dir: u32| -> i32 {
        let mut req = DrmSyncBo {
            handle: npu_handle,
            direction: dir,
            offset: 0,
            size: INPUT_BYTES as u64,
        };
        unsafe {
            libc::ioctl(
                hipx.device.fd,
                drm_ioctl_amdxdna_sync_bo(),
                &mut req as *mut _ as *mut libc::c_void,
            )
        }
    };

    // Test which input-VA flavour the firmware accepts. Run all
    // iterations with each, report.
    let amdgpu_va = agpu_ptr as u64;
    let npu_va = input_va_npu;

    println!("[dbc] testing two input-VA candidates and SYNC_BO variations...");

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
        let sync_ret = sync_imported(SYNC_TO_DEVICE);

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

        let cmd_handles_field: u64 = cmd_handles[0] as u64;
        let args_field: u64 = arg_handles.as_ptr() as u64;
        let mut req = DrmExecCmd {
            ext: 0,
            ext_flags: 0,
            hwctx: ctx.handle,
            ty: CMD_SUBMIT_EXEC_BUF,
            cmd_handles: cmd_handles_field,
            args: args_field,
            cmd_count: cmd_handles.len() as u32,
            arg_count: arg_handles.len() as u32,
            seq: 0,
        };
        let ret = unsafe {
            libc::ioctl(
                hipx.device.fd,
                drm_ioctl_amdxdna_exec_cmd(),
                &mut req as *mut _ as *mut libc::c_void,
            )
        };
        if ret != 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            eprintln!(
                "iter {iter} EXEC_CMD failed errno={errno} ({})",
                std::io::Error::from_raw_os_error(errno)
            );
            return ExitCode::FAILURE;
        }
        let seq = req.seq;
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

    // Cleanup
    unsafe {
        libc::munmap(npu_input_ptr, INPUT_BYTES);
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
