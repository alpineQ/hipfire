//! Minimal amdgpu DRM ioctl wrappers — JUST enough to allocate a GTT
//! BO from `/dev/dri/renderD128` and export it as a dmabuf fd, so we
//! can prove the iGPU↔NPU sharing path on Strix Halo (UMA).
//!
//! On UMA SoCs (Strix Halo / Phoenix / Krackan), an amdgpu GTT BO
//! lives in system RAM and is directly visible to the NPU once
//! imported via PRIME_FD_TO_HANDLE. No copy, no bounce buffer.
//!
//! For full GPU compute we use HIP/HSA via the `rdna-compute` crate;
//! these raw wrappers are only used to:
//!
//!   1. Test the dmabuf round-trip without entangling HIP allocation.
//!   2. Provide an escape hatch for the engine to allocate K cache
//!      memory through amdgpu directly (with a stable GEM handle and
//!      dmabuf fd) rather than going through HSA + portable_export.
//!
//! For HSA/HIP-native dmabuf export, the alternate path is
//! `hsa_amd_portable_export_dmabuf(ptr, size, &fd, &offset)` — used
//! when the engine has already HSA-allocated the buffer. See
//! `engine::npu::dmabuf_export_hsa` (planned).

use crate::ioctl::{
    drm_ioctl_amdgpu_gem_create, drm_ioctl_amdgpu_gem_mmap, drm_ioctl_gem_close,
    DrmAmdgpuGemCreate, DrmAmdgpuGemCreateIn, DrmAmdgpuGemMmap, DrmAmdgpuGemMmapIn, DrmGemClose,
};
use crate::prime::export_handle_to_fd;
use crate::{Result, XdnaError};
use std::os::fd::OwnedFd;
use std::os::unix::io::FromRawFd;

/// amdgpu memory domain bits (uapi `drm/amdgpu_drm.h`).
pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
pub const AMDGPU_GEM_DOMAIN_VRAM: u64 = 0x4;

/// amdgpu GEM_CREATE flag bits.
pub const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;
/// New since kernel 6.7. Forces GTT allocation eligible for dmabuf
/// sharing (no migration, no eviction).
pub const AMDGPU_GEM_CREATE_PREEMPTIBLE: u64 = 1 << 11;

/// Open the amdgpu render node at `/dev/dri/renderD<N>`. Caller owns
/// the fd. Closes on drop.
pub fn open_render_node(node: usize) -> Result<OwnedFd> {
    let path = format!("/dev/dri/renderD{}", 128 + node);
    let cpath = std::ffi::CString::new(path.clone()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if fd < 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!("open({path}, O_RDWR) failed (errno={errno})"),
        });
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Allocate a GTT-domain BO of the given size on the amdgpu render
/// node fd. Returns the GEM handle. CPU-access flag is set so the BO
/// can be mmap'd by the host.
pub fn gem_create_gtt(fd: i32, size: u64) -> Result<u32> {
    let mut req = DrmAmdgpuGemCreate {
        in_: DrmAmdgpuGemCreateIn {
            bo_size: size,
            alignment: 4096,
            domains: AMDGPU_GEM_DOMAIN_GTT,
            domain_flags: AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
        },
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_amdgpu_gem_create(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!("AMDGPU_GEM_CREATE(GTT, {size}) failed (errno={errno})"),
        });
    }
    Ok(unsafe { req.out.handle })
}

/// Get the mmap offset for an amdgpu GEM handle. Pass to mmap() with
/// the device fd to get a CPU-accessible mapping of the BO pages.
pub fn gem_mmap_offset(fd: i32, handle: u32) -> Result<u64> {
    let mut req = DrmAmdgpuGemMmap {
        in_: DrmAmdgpuGemMmapIn { handle, _pad: 0 },
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_amdgpu_gem_mmap(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!("AMDGPU_GEM_MMAP(handle={handle}) failed (errno={errno})"),
        });
    }
    Ok(unsafe { req.out.addr_ptr })
}

/// Close (destroy) an amdgpu GEM handle. Generic GEM_CLOSE works on
/// any DRM driver including amdgpu.
pub fn gem_close(fd: i32, handle: u32) -> Result<()> {
    let mut req = DrmGemClose { handle, pad: 0 };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_gem_close(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        return Err(XdnaError {
            code: errno,
            message: format!("GEM_CLOSE(handle={handle}) on amdgpu fd failed (errno={errno})"),
        });
    }
    Ok(())
}

/// One-shot helper: allocate, mmap, return (handle, mmap'd slice,
/// dmabuf fd). The slice borrows from a private mmap region; the
/// caller must ensure it's dropped before `gem_close` runs.
///
/// Returns (handle, dmabuf_fd, cpu_ptr, size).
pub fn alloc_and_export(
    fd: i32,
    size: u64,
) -> Result<(u32, OwnedFd, *mut u8)> {
    let handle = gem_create_gtt(fd, size)?;
    let off = match gem_mmap_offset(fd, handle) {
        Ok(o) => o,
        Err(e) => {
            let _ = gem_close(fd, handle);
            return Err(e);
        }
    };
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            off as libc::off_t,
        )
    };
    if p == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(0);
        let _ = gem_close(fd, handle);
        return Err(XdnaError {
            code: errno,
            message: format!("mmap(amdgpu BO handle={handle}, off={off:#x}, size={size}) failed"),
        });
    }
    let dmabuf_fd = match export_handle_to_fd(fd, handle) {
        Ok(fd) => fd,
        Err(e) => {
            unsafe { libc::munmap(p, size as usize); }
            let _ = gem_close(fd, handle);
            return Err(e);
        }
    };
    Ok((handle, unsafe { OwnedFd::from_raw_fd(dmabuf_fd) }, p as *mut u8))
}
