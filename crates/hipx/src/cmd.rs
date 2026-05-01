//! Command submission — CONFIG_HWCTX + EXEC_CMD.
//!
//! The full dispatch protocol once a CU artifact exists:
//!
//! 1. Compile a CU offline (MLIR-AIE → PDI). Embed bytes in
//!    `kernels::*_PDI`.
//! 2. Allocate a SHMEM (or DEV) BO and copy the PDI bytes in. The
//!    BO's `xdna_addr` is the device-side pointer.
//! 3. Build a `CuConfig` referencing that BO + a function index. Pack
//!    a `HwctxParamConfigCuHeader` followed by `n` CuConfig entries
//!    into a buffer. Pass via `CONFIG_HWCTX(HWCTX_CONFIG_CU)`.
//! 4. Allocate a CMD-type BO; write a firmware command packet
//!    (descriptor + arg pointers). Submit via `EXEC_CMD`.
//! 5. Wait on the hwctx's syncobj_handle via `fence::wait`.
//!
//! This module wraps steps 3-4. Step 1 is in MLIR-AIE; step 2 is just
//! `Bo::alloc_shmem` + `bo.map()` + memcpy.

use crate::bo::Bo;
use crate::hwctx::Hwctx;
use crate::ioctl::*;
use crate::{Result, XdnaError};

/// A bound CU on a hwctx. Holds onto the CU BO so it lives at least
/// as long as the hwctx that references it.
pub struct CuBinding {
    /// CU BOs we asked the firmware to load. Must outlive the hwctx
    /// because the firmware reads them lazily.
    _cu_bos: Vec<Bo>,
    /// Records the function indices the firmware assigned. The index
    /// the user passes to `submit_exec_cmd` matches the order we
    /// passed them in.
    pub cu_count: u16,
}

/// Bind one or more CU BOs to a hwctx. The CUs become available as
/// indices 0..cu_bos.len() for subsequent EXEC_CMD calls. `cu_func`
/// is a per-CU function selector — typically 0 unless the PDI ships
/// multiple entrypoints.
pub fn config_cus(fd: i32, ctx: &Hwctx, cu_bos: Vec<Bo>, cu_funcs: &[u8]) -> Result<CuBinding> {
    if cu_bos.is_empty() {
        return Err(XdnaError {
            code: 0,
            message: "config_cus: empty cu_bos".into(),
        });
    }
    if cu_funcs.len() != cu_bos.len() {
        return Err(XdnaError {
            code: 0,
            message: format!(
                "config_cus: cu_funcs len {} != cu_bos len {}",
                cu_funcs.len(),
                cu_bos.len()
            ),
        });
    }
    let num_cus = cu_bos.len() as u16;

    // Pack header + array into a single buffer (driver enforces ≤ 4 KiB).
    let header_size = std::mem::size_of::<HwctxParamConfigCuHeader>();
    let entry_size = std::mem::size_of::<CuConfig>();
    let total_size = header_size + entry_size * num_cus as usize;
    if total_size > 4096 {
        return Err(XdnaError {
            code: 0,
            message: format!("config_cus: payload {total_size} > 4096 byte limit"),
        });
    }
    let mut buf = vec![0u8; total_size];
    unsafe {
        let header_ptr = buf.as_mut_ptr() as *mut HwctxParamConfigCuHeader;
        (*header_ptr).num_cus = num_cus;

        let entries_ptr = buf.as_mut_ptr().add(header_size) as *mut CuConfig;
        for (i, bo) in cu_bos.iter().enumerate() {
            (*entries_ptr.add(i)) = CuConfig {
                cu_bo: bo.handle,
                cu_func: cu_funcs[i],
                pad: [0; 3],
            };
        }
    }

    let mut req = DrmConfigHwctx {
        handle: ctx.handle,
        param_type: HWCTX_CONFIG_CU,
        param_val: buf.as_ptr() as u64,
        param_val_size: total_size as u32,
        pad: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_amdxdna_config_hwctx(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!(
                "CONFIG_HWCTX(CU, hwctx={}, num_cus={num_cus}) failed (errno={errno})",
                ctx.handle
            ),
        });
    }
    Ok(CuBinding {
        _cu_bos: cu_bos,
        cu_count: num_cus,
    })
}

/// Submit one or more command BOs to a hwctx. Returns the firmware
/// sequence number so the caller can correlate completion. The
/// hwctx's syncobj will signal once all submitted commands complete.
pub fn submit_exec_cmd(fd: i32, ctx: &Hwctx, cmd_bos: &[&Bo], arg_bos: &[&Bo]) -> Result<u64> {
    let cmd_handles: Vec<u32> = cmd_bos.iter().map(|b| b.handle).collect();
    let arg_handles: Vec<u32> = arg_bos.iter().map(|b| b.handle).collect();

    let mut req = DrmExecCmd {
        ext: 0,
        ext_flags: 0,
        hwctx: ctx.handle,
        ty: CMD_SUBMIT_EXEC_BUF,
        cmd_handles: cmd_handles.as_ptr() as u64,
        args: arg_handles.as_ptr() as u64,
        cmd_count: cmd_handles.len() as u32,
        arg_count: arg_handles.len() as u32,
        seq: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_amdxdna_exec_cmd(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!(
                "EXEC_CMD(hwctx={}, cmds={}, args={}) failed (errno={errno})",
                ctx.handle,
                cmd_handles.len(),
                arg_handles.len()
            ),
        });
    }
    Ok(req.seq)
}
