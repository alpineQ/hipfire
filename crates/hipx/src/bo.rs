//! Buffer Object (BO) lifecycle.
//!
//! Phase 1.0: SHMEM-type BO only. SHMEM is host-pinned pages visible
//! to both CPU and NPU — the right type for activation tensors and
//! anything we want to mmap into our process. DEV / DEV_HEAP / CMD
//! arrive in later phases.

use crate::ioctl::*;
use crate::{Result, XdnaError};
use std::ptr::NonNull;

/// A buffer object owned by an amdxdna device. Drop frees + unmaps.
///
/// Caller is responsible for keeping the device fd alive for the
/// BO's lifetime — that's a `Hipx` (or `Device`) outliving its BOs.
pub struct Bo {
    fd: i32,
    pub handle: u32,
    pub size: usize,
    pub xdna_addr: u64,
    map_offset: u64,
    mapping: Option<(NonNull<u8>, usize)>,
}

impl Bo {
    /// Allocate a SHMEM BO of the requested size. Page-aligns up.
    /// SHMEM = host-pinned pages, visible to NPU via IOMMU. Use for
    /// activation tensors, tokens, anything we want to mmap.
    pub fn alloc_shmem(fd: i32, size: usize) -> Result<Self> {
        Self::alloc_typed(fd, size, BO_SHMEM)
    }

    /// Allocate a DEV BO sub-allocated from the per-client DEV_HEAP.
    /// DEV BOs live in the heap region and are addressable by the NPU
    /// via the device VA returned in `xdna_addr`. Use for inputs the
    /// NPU reads via DMA without going through host-pinned pages.
    pub fn alloc_dev(fd: i32, size: usize) -> Result<Self> {
        Self::alloc_typed(fd, size, BO_DEV)
    }

    /// Allocate a CMD BO. CMD BOs are descriptor packets passed to
    /// EXEC_CMD; the firmware reads them to know what to launch.
    pub fn alloc_cmd(fd: i32, size: usize) -> Result<Self> {
        Self::alloc_typed(fd, size, BO_CMD)
    }

    /// Allocate a per-client DEV_HEAP BO. Required before CREATE_HWCTX.
    ///
    /// The driver enforces "the heap must have a registered user VA"
    /// before any sub-allocation. Per the kernel source
    /// (`amdxdna_gem_heap_alloc` checks `amdxdna_gem_uva(heap) ==
    /// AMDXDNA_INVALID_ADDR`), that user VA gets registered when we
    /// `mmap()` the device fd at the BO's `map_offset`. So this
    /// helper does both: CREATE_BO(DEV_HEAP) → mmap → pre-touch.
    /// Pre-touch ensures all pages are faulted before the firmware
    /// tries to DMA-map them via MAP_HOST_BUFFER.
    pub fn alloc_dev_heap(fd: i32, size: usize) -> Result<Self> {
        let mut bo = Self::alloc_typed(fd, size, BO_DEV_HEAP)?;
        let buf = bo.map()?;
        // Pre-fault every 4 KB page.
        let pagesz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let len = buf.len();
        let mut i = 0;
        while i < len {
            buf[i] = 0;
            i += pagesz;
        }
        Ok(bo)
    }

    fn alloc_typed(fd: i32, size: usize, ty: u32) -> Result<Self> {
        let pagesz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let aligned = size.div_ceil(pagesz) * pagesz;

        let mut req = DrmCreateBo {
            flags: 0,
            vaddr: 0,
            size: aligned as u64,
            ty,
            handle: 0,
        };
        let ret = unsafe {
            libc::ioctl(
                fd,
                drm_ioctl_amdxdna_create_bo(),
                &mut req as *mut _ as *mut libc::c_void,
            )
        };
        if ret != 0 {
            return Err(XdnaError {
                code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                message: format!("CREATE_BO(type={ty}, {aligned}) failed"),
            });
        }
        let handle = req.handle;

        let mut info = DrmGetBoInfo {
            handle,
            ..Default::default()
        };
        let ret = unsafe {
            libc::ioctl(
                fd,
                drm_ioctl_amdxdna_get_bo_info(),
                &mut info as *mut _ as *mut libc::c_void,
            )
        };
        if ret != 0 {
            // Best-effort cleanup
            let _ = close_handle(fd, handle);
            return Err(XdnaError {
                code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                message: format!("GET_BO_INFO(handle={handle}) failed"),
            });
        }

        Ok(Self {
            fd,
            handle,
            size: aligned,
            xdna_addr: info.xdna_addr,
            map_offset: info.map_offset,
            mapping: None,
        })
    }

    /// mmap the BO into the process address space. Idempotent.
    pub fn map(&mut self) -> Result<&mut [u8]> {
        if self.mapping.is_none() {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    self.size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd,
                    self.map_offset as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(XdnaError {
                    code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                    message: format!(
                        "mmap(handle={}, off={:#x}, size={}) failed",
                        self.handle, self.map_offset, self.size
                    ),
                });
            }
            self.mapping = Some((NonNull::new(p as *mut u8).unwrap(), self.size));
        }
        let (p, sz) = self.mapping.unwrap();
        Ok(unsafe { std::slice::from_raw_parts_mut(p.as_ptr(), sz) })
    }

    /// SYNC_BO with the given direction (TO_DEVICE / FROM_DEVICE).
    pub fn sync(&self, direction: u32) -> Result<()> {
        let mut req = DrmSyncBo {
            handle: self.handle,
            direction,
            offset: 0,
            size: self.size as u64,
        };
        let ret = unsafe {
            libc::ioctl(
                self.fd,
                drm_ioctl_amdxdna_sync_bo(),
                &mut req as *mut _ as *mut libc::c_void,
            )
        };
        if ret != 0 {
            return Err(XdnaError {
                code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                message: format!(
                    "SYNC_BO(handle={}, dir={direction}) failed",
                    self.handle
                ),
            });
        }
        Ok(())
    }
}

impl Drop for Bo {
    fn drop(&mut self) {
        if let Some((p, sz)) = self.mapping.take() {
            unsafe {
                libc::munmap(p.as_ptr() as *mut libc::c_void, sz);
            }
        }
        let _ = close_handle(self.fd, self.handle);
    }
}

fn close_handle(fd: i32, handle: u32) -> Result<()> {
    let mut req = DrmGemClose { handle, pad: 0 };
    let ret = unsafe {
        libc::ioctl(
            fd,
            drm_ioctl_gem_close(),
            &mut req as *mut _ as *mut libc::c_void,
        )
    };
    if ret != 0 {
        return Err(XdnaError {
            code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            message: format!("GEM_CLOSE(handle={handle}) failed"),
        });
    }
    Ok(())
}
