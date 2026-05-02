//! `hipx` — Rust-native AMD XDNA / AIE-2P NPU compute.
//!
//! Direct ioctl bypass of XRT and the AMD Ryzen AI userspace runtime,
//! analogous to how `redline` bypasses libdrm_amdgpu / HIP for the
//! iGPU side. Talks to `/dev/accel/accel0` via the 9 `DRM_AMDXDNA_*`
//! ioctls defined in `<drm/amdxdna_accel.h>`.
//!
//! # Layout
//!
//! ```text
//! ioctl     — raw struct layouts + ioctl numbers (the ABI surface)
//! device    — open + GET_INFO probes (read-only metadata)
//! bo        — buffer-object lifecycle (SHMEM, DEV_HEAP, DEV, CMD)
//! hwctx     — hardware-context create + destroy
//! runtime   — high-level `Hipx` wrapper combining all of the above
//! dispatch  — predicates engine code uses to pick kernels per SoC
//! kernels   — embedded PDI artifacts (placeholder until Phase 1.3)
//! ```
//!
//! # Engine usage
//!
//! ```ignore
//! use hipx::{Hipx, HwctxBuilder};
//!
//! let hipx = Hipx::open()?;
//! if !hipx::dispatch::has_npu(&hipx.info) { return Ok(()); }
//!
//! let mut b = HwctxBuilder::default();
//! b.num_columns = (hipx::dispatch::cols_available(&hipx.info) / 2) as u32;
//! let ctx = hipx.create_hwctx(&b)?;
//! // ... configure CUs, submit cmds, wait on syncobj
//! ```

pub mod ioctl;
pub mod device;
pub mod bo;
pub mod hwctx;
pub mod runtime;
pub mod dispatch;
pub mod kernels;
pub mod cmd;
pub mod fence;
pub mod prime;
pub mod ert;
pub mod agpu;
pub mod hsa_dmabuf;

#[derive(Debug)]
pub struct XdnaError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for XdnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hipx error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for XdnaError {}

pub type Result<T> = std::result::Result<T, XdnaError>;

pub use bo::Bo;
pub use device::{Device, NpuInfo};
pub use hwctx::{Hwctx, HwctxBuilder};
pub use runtime::Hipx;
