//! xdna-compute — direct-KMD NPU compute for AMD AIE-2P (Ryzen AI / XDNA).
//!
//! Bypasses XRT and the AMD Ryzen AI userspace runtime entirely. Talks to
//! `/dev/accel/accel0` via the 9 `DRM_AMDXDNA_*` ioctls defined in
//! `<drm/amdxdna_accel.h>` (kernel uapi, GPL-2.0 WITH Linux-syscall-note).
//!
//! Sister crate to `redline` (which does the same thing for amdgpu / RDNA).
//!
//! # Phase 0 scope
//!
//! Read-only metadata probes only:
//! - open `/dev/accel/accel0`
//! - `DRM_IOCTL_AMDXDNA_GET_INFO` for AIE version, metadata, clocks,
//!   firmware version, resource info
//!
//! No BO allocation, no hwctx, no command submission. Those land in
//! Phase 1 once the ABI is verified.

pub mod ioctl;
pub mod device;
pub mod bo;
pub mod hwctx;

#[derive(Debug)]
pub struct XdnaError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for XdnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "xdna error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for XdnaError {}

pub type Result<T> = std::result::Result<T, XdnaError>;
