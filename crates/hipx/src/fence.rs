//! DRM syncobj fence wait. Used to block on EXEC_CMD completion.
//!
//! `Hwctx::create()` returns a syncobj_handle from the kernel driver.
//! After EXEC_CMD signals it (via the firmware writing to the syncobj
//! after the CU runs), we call DRM_IOCTL_SYNCOBJ_WAIT to block.

use crate::ioctl::*;
use crate::{Result, XdnaError};
use std::time::Duration;

/// Wait for a single syncobj to be signaled.
///
/// `timeout`: relative timeout. Internally translated to an absolute
/// CLOCK_MONOTONIC ns timestamp the kernel expects.
pub fn wait(fd: i32, syncobj_handle: u32, timeout: Duration) -> Result<()> {
    wait_many(fd, &[syncobj_handle], timeout, true)
}

/// Wait for multiple syncobjs. `wait_all = true` waits for all to be
/// signaled; `false` returns when any one signals.
pub fn wait_many(
    fd: i32,
    handles: &[u32],
    timeout: Duration,
    wait_all: bool,
) -> Result<()> {
    if handles.is_empty() {
        return Ok(());
    }
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let abs_ns = unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
        let now_ns = (ts.tv_sec as i64) * 1_000_000_000 + (ts.tv_nsec as i64);
        let to_ns = timeout.as_nanos().min(i64::MAX as u128) as i64;
        now_ns.saturating_add(to_ns)
    };

    let flags = if wait_all {
        SYNCOBJ_WAIT_FLAGS_WAIT_ALL | SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT
    } else {
        SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT
    };

    let mut req = DrmSyncobjWait {
        handles: handles.as_ptr() as u64,
        timeout_nsec: abs_ns,
        count_handles: handles.len() as u32,
        flags,
        first_signaled: 0,
        pad: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_syncobj_wait(),
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
                "SYNCOBJ_WAIT(handles={}, abs_ns={abs_ns}) failed (errno={errno})",
                handles.len()
            ),
        });
    }
    Ok(())
}
