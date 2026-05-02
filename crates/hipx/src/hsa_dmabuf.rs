//! Minimal dlopen wrapper for `hsa_amd_portable_export_dmabuf`.
//!
//! HSA-allocated memory (which includes everything HIP/HSA hands out
//! via `hipMalloc` since HIP wraps HSA) can be exported as a dmabuf
//! fd through a single ROCm runtime call. That fd then plugs into
//! `hipx::Bo::from_imported_dmabuf` and reaches the NPU with no
//! copy.
//!
//! This is the engine-side hookup point: the engine already
//! allocates K cache, weights, and scratch through HIP, so any
//! HIP `*mut c_void` device pointer becomes NPU-visible by going
//! through this export → import path.
//!
//! On Strix Halo (UMA), the underlying physical pages are the same
//! whether allocated via amdgpu DRM or HSA's GTT pool, so the
//! resulting NPU access has identical performance characteristics
//! to the amdgpu round-trip validated in `dmabuf_compute`.
//!
//! Library is loaded lazily — absence of libhsa-runtime64 is a
//! recoverable error (caller falls back to iGPU-only path).

use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::os::fd::OwnedFd;
use std::os::unix::io::FromRawFd;
use std::sync::OnceLock;

/// HSA status code (0 = success).
pub type HsaStatus = u32;
pub const HSA_STATUS_SUCCESS: HsaStatus = 0;

type FnExportDmabuf = unsafe extern "C" fn(
    ptr: *const c_void,
    size: usize,
    dmabuf_fd: *mut i32,
    offset: *mut u64,
) -> HsaStatus;

type FnCloseDmabuf = unsafe extern "C" fn(dmabuf_fd: i32) -> HsaStatus;

struct HsaDmabufLib {
    _lib: Library,
    export: FnExportDmabuf,
    _close: FnCloseDmabuf,
}

unsafe impl Send for HsaDmabufLib {}
unsafe impl Sync for HsaDmabufLib {}

static HSA: OnceLock<Result<HsaDmabufLib, String>> = OnceLock::new();

fn lib() -> Result<&'static HsaDmabufLib, String> {
    HSA.get_or_init(|| unsafe {
        let lib = Library::new("libhsa-runtime64.so.1")
            .or_else(|_| Library::new("libhsa-runtime64.so"))
            .map_err(|e| format!("dlopen libhsa-runtime64: {e}"))?;
        let export: Symbol<FnExportDmabuf> = lib
            .get(b"hsa_amd_portable_export_dmabuf\0")
            .map_err(|e| format!("symbol hsa_amd_portable_export_dmabuf: {e}"))?;
        let close: Symbol<FnCloseDmabuf> = lib
            .get(b"hsa_amd_portable_close_dmabuf\0")
            .map_err(|e| format!("symbol hsa_amd_portable_close_dmabuf: {e}"))?;
        let export = *export;
        let close = *close;
        Ok(HsaDmabufLib {
            _lib: lib,
            export,
            _close: close,
        })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Export an HSA-allocated buffer (returned by `hsa_amd_memory_pool_allocate`,
/// or — by extension — `hipMalloc`, since HIP wraps HSA) as a dmabuf fd.
/// Returns the fd plus the byte offset within it where the buffer starts
/// (HSA may pack multiple allocations into one underlying GTT region).
///
/// The returned `OwnedFd` closes the fd via `hsa_amd_portable_close_dmabuf`
/// is **NOT** what we do here — we close it via the standard `close()`
/// syscall when the OwnedFd drops, which is what the kernel side
/// expects. The `hsa_amd_portable_close_dmabuf` is a userspace-pool
/// hint that we don't currently need; standard close works for our
/// PRIME_FD_TO_HANDLE use case (the receiving driver bumps a refcount
/// on import, so the kernel-side dmabuf survives our close).
pub fn export(ptr: *const c_void, size: usize) -> Result<(OwnedFd, u64), String> {
    let l = lib()?;
    let mut fd: i32 = -1;
    let mut offset: u64 = 0;
    let status = unsafe { (l.export)(ptr, size, &mut fd, &mut offset) };
    if status != HSA_STATUS_SUCCESS {
        return Err(format!(
            "hsa_amd_portable_export_dmabuf({ptr:?}, {size}) failed: status={status}"
        ));
    }
    if fd < 0 {
        return Err(format!(
            "hsa_amd_portable_export_dmabuf returned status=0 but fd={fd}"
        ));
    }
    Ok((unsafe { OwnedFd::from_raw_fd(fd) }, offset))
}

/// Probe whether HSA dmabuf export is available without actually
/// allocating anything. Useful for capability detection at runtime.
pub fn available() -> bool {
    lib().is_ok()
}
