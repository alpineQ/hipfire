//! High-level `Hipx` runtime — opens the device, probes capabilities,
//! manages the per-client DEV_HEAP singleton, and gives ergonomic
//! constructors for BOs and hwctxs.
//!
//! Mirror of `redline::Device` for the NPU. Engine-side code should
//! generally talk to `Hipx`, not the lower-level `Device` / `Bo` /
//! `Hwctx` types directly.

use crate::bo::Bo;
use crate::device::{Device, NpuInfo};
use crate::hwctx::{Hwctx, HwctxBuilder};
use crate::Result;

/// One-per-process NPU runtime.
///
/// Holds the device fd and the singleton DEV_HEAP. The driver enforces
/// "one DEV_HEAP per client fd"; trying to allocate two returns EBUSY.
/// Multiple `Hwctx` can be live concurrently against the same `Hipx`.
pub struct Hipx {
    pub device: Device,
    pub info: NpuInfo,
    /// Per-client device heap. Required to exist before any
    /// CREATE_HWCTX call (driver-side enforcement).
    pub heap: Bo,
}

impl Hipx {
    /// Open `/dev/accel/accel0` (or override path) with a default
    /// 64 MB DEV_HEAP. 64 MB is plenty for early dispatch experiments;
    /// real workloads will want to size the heap to roughly
    /// `weight_size + activation_workspace + scratch`.
    pub fn open() -> Result<Self> {
        Self::open_with(None, 64 * 1024 * 1024)
    }

    pub fn open_with(path: Option<&str>, heap_size: usize) -> Result<Self> {
        let device = Device::open(path)?;
        let info = device.probe()?;
        let heap = Bo::alloc_dev_heap(device.fd, heap_size)?;
        Ok(Self { device, info, heap })
    }

    /// Allocate a SHMEM BO bound to this runtime's fd.
    pub fn alloc_shmem(&self, size: usize) -> Result<Bo> {
        Bo::alloc_shmem(self.device.fd, size)
    }

    /// Allocate a DEV BO sub-allocated from this runtime's heap.
    pub fn alloc_dev(&self, size: usize) -> Result<Bo> {
        Bo::alloc_dev(self.device.fd, size)
    }

    /// Get a host-visible mutable slice into a DEV BO via this
    /// runtime's heap mapping. DEV BOs don't have their own
    /// map_offset — they're sub-regions of the heap.
    ///
    /// # Safety
    /// Caller asserts `bo` is a DEV BO from this runtime's heap.
    pub unsafe fn dev_slice<'a>(&'a self, bo: &Bo) -> Result<&'a mut [u8]> {
        bo.dev_slice_in_heap(&self.heap)
    }

    /// Allocate a CMD BO for EXEC_CMD payloads.
    pub fn alloc_cmd(&self, size: usize) -> Result<Bo> {
        Bo::alloc_cmd(self.device.fd, size)
    }

    /// Create a hardware context. `core_rows` is sourced from the
    /// probed AIE metadata so callers don't have to re-query.
    pub fn create_hwctx(&self, b: &HwctxBuilder) -> Result<Hwctx> {
        Hwctx::create(self.device.fd, self.info.core_tiles.1 as u32, b)
    }
}
