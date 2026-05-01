//! DRM PRIME — dmabuf-based BO sharing across drivers.
//!
//! On Strix Halo this is the architectural moat: a BO allocated on
//! `/dev/dri/renderD128` (gfx1151 iGPU via amdgpu) can be exported as
//! a dmabuf fd and imported on `/dev/accel/accel0` (NPU via amdxdna)
//! without copying — both engines pull from the same UMA pages.
//!
//! These ioctls are generic across DRM drivers (nr 0x2D / 0x2E,
//! below the DRM_COMMAND_BASE per-driver region). They work on any
//! DRM render node fd.

use crate::ioctl::*;
use crate::{Result, XdnaError};

/// Export a BO handle as a dmabuf fd. Caller is responsible for
/// closing the fd when done.
pub fn export_handle_to_fd(fd: i32, handle: u32) -> Result<i32> {
    let mut req = DrmPrimeHandle {
        handle,
        flags: PRIME_FLAG_RDWR | PRIME_FLAG_CLOEXEC,
        fd: -1,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_prime_handle_to_fd(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!("PRIME_HANDLE_TO_FD(handle={handle}) failed (errno={errno})"),
        });
    }
    Ok(req.fd)
}

/// Import a dmabuf fd as a BO handle on this fd's driver. The
/// returned handle is opaque to us; pass it back to the driver via
/// other ioctls (GET_BO_INFO, GEM_CLOSE).
pub fn import_fd_to_handle(fd: i32, dmabuf_fd: i32) -> Result<u32> {
    let mut req = DrmPrimeHandle {
        handle: 0,
        flags: 0,
        fd: dmabuf_fd,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_prime_fd_to_handle(),
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
                "PRIME_FD_TO_HANDLE(dmabuf_fd={dmabuf_fd}) failed (errno={errno})"
            ),
        });
    }
    Ok(req.handle)
}
