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
    /// Whether this BO is a DEV_HEAP (drives MAP_HUGETLB choice on
    /// mmap; the firmware MAP_HOST_BUFFER needs huge-page backing).
    pub(crate) is_dev_heap: bool,
}

impl Bo {
    /// Allocate a SHMEM BO of the requested size. Page-aligns up.
    /// SHMEM = host-pinned pages, visible to NPU via IOMMU. Use for
    /// activation tensors, tokens, anything we want to mmap.
    pub fn alloc_shmem(fd: i32, size: usize) -> Result<Self> {
        Self::alloc_typed(fd, size, BO_SHMEM)
    }

    /// Allocate a SHMEM BO with the EXACT requested size in the
    /// CREATE_BO ioctl (the kernel still page-aligns the actual
    /// allocation). This matters for some firmware validations that
    /// inspect the user-requested size — XRT for example asks for
    /// 8 bytes for the ctrlpkts BO and 1 byte for the trace BO,
    /// not the rounded-up 4096 we'd get from `alloc_shmem`.
    pub fn alloc_shmem_exact(fd: i32, size: usize) -> Result<Self> {
        Self::alloc_typed_unaligned(fd, size, BO_SHMEM)
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

    /// Like `alloc_typed` but passes the user-requested size directly
    /// to CREATE_BO without page-alignment. The kernel still aligns
    /// the actual allocation, but the request struct keeps the raw
    /// size — needed when firmware validates against the requested
    /// size rather than the kernel-aligned one.
    fn alloc_typed_unaligned(fd: i32, size: usize, ty: u32) -> Result<Self> {
        Self::alloc_typed_inner(fd, size, ty, /*page_align=*/false)
    }

    fn alloc_typed(fd: i32, size: usize, ty: u32) -> Result<Self> {
        Self::alloc_typed_inner(fd, size, ty, /*page_align=*/true)
    }

    fn alloc_typed_inner(fd: i32, size: usize, ty: u32, page_align: bool) -> Result<Self> {
        let pagesz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let aligned = if page_align {
            size.div_ceil(pagesz) * pagesz
        } else {
            size
        };
        // For the mmap, we still need a page-aligned size internally.
        let _ = pagesz;

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
            is_dev_heap: ty == BO_DEV_HEAP,
        })
    }

    /// mmap the BO into the process address space. Idempotent.
    ///
    /// DEV_HEAP requires a specific dance to satisfy the AIE-2P
    /// firmware's MAP_HOST_BUFFER protocol (verified via strace of
    /// xrt-smi):
    ///   1. mmap(anon, 2 × heap_size) to reserve VA space
    ///   2. find the next `heap_size`-aligned address inside it
    ///   3. mmap(MAP_SHARED | MAP_FIXED | MAP_LOCKED) at that addr
    ///      against the device fd at the BO's map_offset
    /// MAP_LOCKED page-locks the heap so PASID translation always
    /// hits resident pages; MAP_FIXED at an exact heap-size-aligned
    /// address is what makes the kernel build IOMMU mappings the
    /// firmware accepts.
    pub fn map(&mut self) -> Result<&mut [u8]> {
        if self.mapping.is_none() {
            let p = if self.is_dev_heap {
                self.mmap_dev_heap_aligned()?
            } else {
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
                // Lock the pages so PASID-based device access always
                // hits resident pages. Best-effort — ignore EINVAL on
                // file-backed mappings that don't support mlock.
                unsafe { libc::mlock(p, self.size); }
                p as *mut u8
            };
            self.mapping = Some((NonNull::new(p).unwrap(), self.size));
        }
        let (p, sz) = self.mapping.unwrap();
        Ok(unsafe { std::slice::from_raw_parts_mut(p.as_ptr(), sz) })
    }

    /// XRT-style aligned heap mmap: reserve 2×size as anonymous,
    /// find the next size-aligned address inside, then MAP_FIXED
    /// the device fd over it with MAP_LOCKED. The reservation
    /// remainder gets implicitly trimmed when MAP_FIXED overlaps it.
    fn mmap_dev_heap_aligned(&self) -> Result<*mut u8> {
        let alignment = self.size; // = dev_mem_size = 64 MB
        let reserve = self.size * 2;
        let anon = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserve,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if anon == libc::MAP_FAILED {
            return Err(XdnaError {
                code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                message: format!("DEV_HEAP anon-reserve mmap({reserve}) failed"),
            });
        }
        let base = anon as usize;
        let aligned = (base + alignment - 1) & !(alignment - 1);

        // Trim head/tail of reservation outside the aligned block.
        if aligned > base {
            unsafe { libc::munmap(base as *mut libc::c_void, aligned - base); }
        }
        let tail_start = aligned + self.size;
        let tail_end = base + reserve;
        if tail_end > tail_start {
            unsafe {
                libc::munmap(tail_start as *mut libc::c_void, tail_end - tail_start);
            }
        }

        // MAP_FIXED the device fd over the aligned region.
        let p = unsafe {
            libc::mmap(
                aligned as *mut libc::c_void,
                self.size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED | libc::MAP_LOCKED,
                self.fd,
                self.map_offset as libc::off_t,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(XdnaError {
                code: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                message: format!(
                    "DEV_HEAP MAP_FIXED|MAP_LOCKED mmap(addr={:#x}, off={:#x}, size={}) failed",
                    aligned, self.map_offset, self.size
                ),
            });
        }
        Ok(p as *mut u8)
    }

    /// Direct access to this BO's mapped pointer (must call `map()`
    /// first). Returns None for unmapped BOs, including DEV BOs that
    /// have no map_offset of their own — those are accessed through
    /// their parent DEV_HEAP's mapping (see `dev_slice_in_heap`).
    pub fn host_ptr(&self) -> Option<*mut u8> {
        self.mapping.as_ref().map(|(p, _)| p.as_ptr())
    }

    /// For a DEV BO sub-allocated from a DEV_HEAP, return a slice
    /// into the heap's mapped region pointing at this BO's data.
    /// The heap must have been mapped (Hipx::open does this).
    ///
    /// # Safety
    /// Caller asserts that `heap` is in fact this BO's parent DEV_HEAP.
    pub unsafe fn dev_slice_in_heap<'a>(&self, heap: &'a Bo) -> Result<&'a mut [u8]> {
        let heap_ptr = heap.host_ptr().ok_or_else(|| XdnaError {
            code: 0,
            message: "DEV_HEAP not mapped".to_string(),
        })?;
        if self.xdna_addr < heap.xdna_addr
            || self.xdna_addr + self.size as u64 > heap.xdna_addr + heap.size as u64
        {
            return Err(XdnaError {
                code: 0,
                message: format!(
                    "DEV BO xdna_addr={:#x}+{} not within heap xdna_addr={:#x}+{}",
                    self.xdna_addr, self.size, heap.xdna_addr, heap.size
                ),
            });
        }
        let offset = (self.xdna_addr - heap.xdna_addr) as usize;
        Ok(std::slice::from_raw_parts_mut(
            heap_ptr.add(offset),
            self.size,
        ))
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
