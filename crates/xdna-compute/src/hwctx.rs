//! Hardware context lifecycle.
//!
//! An hwctx is a slice of the AIE column array reserved for one
//! process. CUs (compute units) are bound to it via CONFIG_HWCTX
//! before commands can run. The kernel returns a syncobj_handle we
//! use to wait on command completion.
//!
//! On AIE-2P the only device type is KMQ (kernel-managed queue) — we
//! don't allocate the UMQ BO ourselves; passing umq_bo = 0 lets the
//! driver create internal queue resources.

use crate::ioctl::*;
use crate::{Result, XdnaError};

/// Hardware context request — 1 column for first dispatch.
pub struct HwctxBuilder {
    pub num_columns: u32,
    pub max_opc: u32,
    pub mem_size: u32,
    pub priority: u32,
    pub gops: u32,
    pub fps: u32,
    pub log_buf_bo: u32,
}

impl Default for HwctxBuilder {
    fn default() -> Self {
        Self {
            num_columns: 1,
            max_opc: 0,
            mem_size: 0,
            priority: QOS_NORMAL_PRIORITY,
            gops: 0,
            fps: 0,
            log_buf_bo: INVALID_BO_HANDLE,
        }
    }
}

pub struct Hwctx {
    fd: i32,
    pub handle: u32,
    pub syncobj_handle: u32,
    pub umq_doorbell: u32,
}

impl Hwctx {
    /// Create the hwctx. `core_rows` is core.row_count from
    /// QUERY_AIE_METADATA (4 on AIE-2P) — driver expects num_tiles =
    /// num_columns × core_rows.
    pub fn create(fd: i32, core_rows: u32, b: &HwctxBuilder) -> Result<Self> {
        let qos = QosInfo {
            gops: b.gops,
            fps: b.fps,
            dma_bandwidth: 0,
            latency: 0,
            frame_exec_time: 0,
            priority: b.priority,
        };
        let mut req = DrmCreateHwctx {
            ext: 0,
            ext_flags: 0,
            qos_p: &qos as *const _ as u64,
            umq_bo: INVALID_BO_HANDLE,
            log_buf_bo: b.log_buf_bo,
            max_opc: b.max_opc,
            num_tiles: b.num_columns * core_rows,
            mem_size: b.mem_size,
            umq_doorbell: 0,
            handle: 0,
            syncobj_handle: 0,
        };
        let ret = unsafe {
            libc::ioctl(
                fd,
                drm_ioctl_amdxdna_create_hwctx(),
                &mut req as *mut _ as *mut libc::c_void,
            )
        };
        if ret != 0 {
            return Err(XdnaError {
                code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                message: format!(
                    "CREATE_HWCTX(num_tiles={}, max_opc={}, mem_size={}) failed",
                    req.num_tiles, req.max_opc, req.mem_size
                ),
            });
        }
        Ok(Self {
            fd,
            handle: req.handle,
            syncobj_handle: req.syncobj_handle,
            umq_doorbell: req.umq_doorbell,
        })
    }
}

impl Drop for Hwctx {
    fn drop(&mut self) {
        let mut req = DrmDestroyHwctx {
            handle: self.handle,
            pad: 0,
        };
        unsafe {
            libc::ioctl(
                self.fd,
                drm_ioctl_amdxdna_destroy_hwctx(),
                &mut req as *mut _ as *mut libc::c_void,
            );
        }
    }
}
