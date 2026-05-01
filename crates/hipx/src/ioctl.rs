//! Raw `amdxdna` ioctl ABI.
//!
//! Hand-translated from `/usr/include/drm/amdxdna_accel.h`. Layouts must
//! match the kernel struct definitions exactly (field order, padding,
//! `__counted_by` flexible arrays passed via separate buffer).

#![allow(dead_code)]

// DRM uapi constants
const DRM_IOCTL_BASE: u8 = b'd';
const DRM_COMMAND_BASE: u8 = 0x40;

// Ioctl direction bits
const IOC_NONE: u32 = 0;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;

const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ((dir << IOC_DIRSHIFT)
        | ((ty as u32) << IOC_TYPESHIFT)
        | ((nr as u32) << IOC_NRSHIFT)
        | ((size as u32) << IOC_SIZESHIFT)) as libc::c_ulong
}

const fn drm_iowr<T>(nr: u8) -> libc::c_ulong {
    ioc(
        IOC_READ | IOC_WRITE,
        DRM_IOCTL_BASE,
        DRM_COMMAND_BASE + nr,
        std::mem::size_of::<T>(),
    )
}

const fn drm_iow<T>(nr: u8) -> libc::c_ulong {
    ioc(IOC_WRITE, DRM_IOCTL_BASE, nr, std::mem::size_of::<T>())
}

/// IOWR for *generic* DRM ioctls — those whose nr is below
/// DRM_COMMAND_BASE (e.g. PRIME, SYNCOBJ, GEM_CLOSE). The driver-
/// specific helpers (`drm_iowr`) add the base; this one does not.
const fn drm_iowr_generic<T>(nr: u8) -> libc::c_ulong {
    ioc(IOC_READ | IOC_WRITE, DRM_IOCTL_BASE, nr, std::mem::size_of::<T>())
}

// amdxdna ioctl nrs (see `enum amdxdna_drm_ioctl_id` in the uapi)
pub const NR_CREATE_HWCTX: u8 = 0;
pub const NR_DESTROY_HWCTX: u8 = 1;
pub const NR_CONFIG_HWCTX: u8 = 2;
pub const NR_CREATE_BO: u8 = 3;
pub const NR_GET_BO_INFO: u8 = 4;
pub const NR_SYNC_BO: u8 = 5;
pub const NR_EXEC_CMD: u8 = 6;
pub const NR_GET_INFO: u8 = 7;
pub const NR_SET_STATE: u8 = 8;
pub const NR_GET_ARRAY: u8 = 10;

// GET_INFO param values (enum amdxdna_drm_get_param)
pub const QUERY_AIE_STATUS: u32 = 0;
pub const QUERY_AIE_METADATA: u32 = 1;
pub const QUERY_AIE_VERSION: u32 = 2;
pub const QUERY_CLOCK_METADATA: u32 = 3;
pub const QUERY_SENSORS: u32 = 4;
pub const QUERY_HW_CONTEXTS: u32 = 5;
pub const QUERY_FIRMWARE_VERSION: u32 = 8;
pub const GET_POWER_MODE: u32 = 9;
pub const QUERY_TELEMETRY: u32 = 10;
pub const GET_FORCE_PREEMPT_STATE: u32 = 11;
pub const QUERY_RESOURCE_INFO: u32 = 12;
pub const GET_FRAME_BOUNDARY_PREEMPT_STATE: u32 = 13;

// ─── Structs (mirror amdxdna_accel.h exactly) ──────────────────────────

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmGetInfo {
    pub param: u32,
    pub buffer_size: u32,
    pub buffer: u64,
}

pub fn drm_ioctl_amdxdna_get_info() -> libc::c_ulong {
    drm_iowr::<DrmGetInfo>(NR_GET_INFO)
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryAieVersion {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryAieTileMetadata {
    pub row_count: u16,
    pub row_start: u16,
    pub dma_channel_count: u16,
    pub lock_count: u16,
    pub event_reg_count: u16,
    pub pad: [u16; 3],
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryAieMetadata {
    pub col_size: u32,
    pub cols: u16,
    pub rows: u16,
    pub version: QueryAieVersion,
    pub core: QueryAieTileMetadata,
    pub mem: QueryAieTileMetadata,
    pub shim: QueryAieTileMetadata,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryClock {
    pub name: [u8; 16],
    pub freq_mhz: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryClockMetadata {
    pub mp_npu_clock: QueryClock,
    pub h_clock: QueryClock,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QueryFirmwareVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub build: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct GetResourceInfo {
    pub npu_clk_max: u64,
    pub npu_tops_max: u64,
    pub npu_task_max: u64,
    pub npu_tops_curr: u64,
    pub npu_task_curr: u64,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct GetPowerMode {
    pub power_mode: u8,
    pub pad: [u8; 7],
}

// ─── BO management ─────────────────────────────────────────────────────

// amdxdna_bo_type
pub const BO_INVALID: u32 = 0;
pub const BO_SHMEM: u32 = 1;
pub const BO_DEV_HEAP: u32 = 2;
pub const BO_DEV: u32 = 3;
pub const BO_CMD: u32 = 4;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmCreateBo {
    pub flags: u64,
    pub vaddr: u64,
    pub size: u64,
    pub ty: u32,
    pub handle: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmGetBoInfo {
    pub ext: u64,
    pub ext_flags: u64,
    pub handle: u32,
    pub pad: u32,
    pub map_offset: u64,
    pub vaddr: u64,
    pub xdna_addr: u64,
}

pub const SYNC_TO_DEVICE: u32 = 0;
pub const SYNC_FROM_DEVICE: u32 = 1;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmSyncBo {
    pub handle: u32,
    pub direction: u32,
    pub offset: u64,
    pub size: u64,
}

// Generic DRM GEM_CLOSE — used to release any DRM BO handle.
const DRM_IOCTL_GEM_CLOSE_NR: u8 = 0x09;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmGemClose {
    pub handle: u32,
    pub pad: u32,
}

pub fn drm_ioctl_amdxdna_create_bo() -> libc::c_ulong {
    drm_iowr::<DrmCreateBo>(NR_CREATE_BO)
}

pub fn drm_ioctl_amdxdna_get_bo_info() -> libc::c_ulong {
    drm_iowr::<DrmGetBoInfo>(NR_GET_BO_INFO)
}

pub fn drm_ioctl_amdxdna_sync_bo() -> libc::c_ulong {
    drm_iowr::<DrmSyncBo>(NR_SYNC_BO)
}

pub fn drm_ioctl_gem_close() -> libc::c_ulong {
    drm_iow::<DrmGemClose>(DRM_IOCTL_GEM_CLOSE_NR)
}

// ─── HW context lifecycle ──────────────────────────────────────────────

// QoS priority levels (uapi)
pub const QOS_REALTIME_PRIORITY: u32 = 0x100;
pub const QOS_HIGH_PRIORITY: u32 = 0x180;
pub const QOS_NORMAL_PRIORITY: u32 = 0x200;
pub const QOS_LOW_PRIORITY: u32 = 0x280;

pub const INVALID_BO_HANDLE: u32 = 0;
pub const INVALID_CTX_HANDLE: u32 = 0;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct QosInfo {
    pub gops: u32,
    pub fps: u32,
    pub dma_bandwidth: u32,
    pub latency: u32,
    pub frame_exec_time: u32,
    pub priority: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmCreateHwctx {
    pub ext: u64,
    pub ext_flags: u64,
    pub qos_p: u64,
    pub umq_bo: u32,
    pub log_buf_bo: u32,
    pub max_opc: u32,
    pub num_tiles: u32,
    pub mem_size: u32,
    pub umq_doorbell: u32,
    pub handle: u32,
    pub syncobj_handle: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmDestroyHwctx {
    pub handle: u32,
    pub pad: u32,
}

pub fn drm_ioctl_amdxdna_create_hwctx() -> libc::c_ulong {
    drm_iowr::<DrmCreateHwctx>(NR_CREATE_HWCTX)
}

pub fn drm_ioctl_amdxdna_destroy_hwctx() -> libc::c_ulong {
    drm_iowr::<DrmDestroyHwctx>(NR_DESTROY_HWCTX)
}

// ─── HW context configuration (CU bind) ────────────────────────────────

/// Param values for amdxdna_drm_config_hwctx.
pub const HWCTX_CONFIG_CU: u32 = 0;
pub const HWCTX_ASSIGN_DBG_BUF: u32 = 1;
pub const HWCTX_REMOVE_DBG_BUF: u32 = 2;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmConfigHwctx {
    pub handle: u32,
    pub param_type: u32,
    pub param_val: u64,
    pub param_val_size: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct CuConfig {
    pub cu_bo: u32,
    pub cu_func: u8,
    pub pad: [u8; 3],
}

/// Header for HwctxParamConfigCu — followed by `num_cus` CuConfig
/// entries in the same buffer (max 4 KiB total per uapi note).
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct HwctxParamConfigCuHeader {
    pub num_cus: u16,
    pub pad: [u16; 3],
}

pub fn drm_ioctl_amdxdna_config_hwctx() -> libc::c_ulong {
    drm_iowr::<DrmConfigHwctx>(NR_CONFIG_HWCTX)
}

// ─── Command execution ─────────────────────────────────────────────────

pub const CMD_SUBMIT_EXEC_BUF: u32 = 0;
pub const CMD_SUBMIT_DEPENDENCY: u32 = 1;
pub const CMD_SUBMIT_SIGNAL: u32 = 2;

pub const INVALID_CMD_HANDLE: u64 = u64::MAX;
pub const INVALID_FENCE_HANDLE: u32 = 0;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmExecCmd {
    pub ext: u64,
    pub ext_flags: u64,
    pub hwctx: u32,
    pub ty: u32,
    pub cmd_handles: u64,
    pub args: u64,
    pub cmd_count: u32,
    pub arg_count: u32,
    pub seq: u64,
}

pub fn drm_ioctl_amdxdna_exec_cmd() -> libc::c_ulong {
    drm_iowr::<DrmExecCmd>(NR_EXEC_CMD)
}

// ─── State (power mode, AIE mem/reg writes, preempt) ───────────────────

pub const SET_POWER_MODE: u32 = 0;
pub const WRITE_AIE_MEM: u32 = 1;
pub const WRITE_AIE_REG: u32 = 2;
pub const SET_FORCE_PREEMPT: u32 = 3;
pub const SET_FRAME_BOUNDARY_PREEMPT: u32 = 4;

pub const POWER_MODE_DEFAULT: u8 = 0;
pub const POWER_MODE_LOW: u8 = 1;
pub const POWER_MODE_MEDIUM: u8 = 2;
pub const POWER_MODE_HIGH: u8 = 3;
pub const POWER_MODE_TURBO: u8 = 4;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmSetState {
    pub param: u32,
    pub buffer_size: u32,
    pub buffer: u64,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct SetPowerMode {
    pub power_mode: u8,
    pub pad: [u8; 7],
}

pub fn drm_ioctl_amdxdna_set_state() -> libc::c_ulong {
    drm_iowr::<DrmSetState>(NR_SET_STATE)
}

// ─── Bulk array queries (active hwctxs, async errors) ──────────────────

pub const HW_CONTEXT_ALL: u32 = 0;
pub const HW_LAST_ASYNC_ERR: u32 = 2;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmGetArray {
    pub param: u32,
    pub element_size: u32,
    pub num_element: u32,
    pub pad: u32,
    pub buffer: u64,
}

pub fn drm_ioctl_amdxdna_get_array() -> libc::c_ulong {
    drm_iowr::<DrmGetArray>(NR_GET_ARRAY)
}

// ─── DRM syncobj wait (generic, not amdxdna-specific) ──────────────────
//
// CREATE_HWCTX returns a syncobj_handle. To wait on command completion
// after EXEC_CMD we issue DRM_IOCTL_SYNCOBJ_WAIT against that handle.
// These ioctl numbers live in `<drm/drm.h>` below DRM_COMMAND_BASE.

// Per <drm/drm.h>: SYNCOBJ_WAIT=0xC3, TIMELINE_WAIT=0xCA. We had these
// wrong (0xCA / 0xCF) for the entire bring-up — SYNCOBJ_WAIT was hitting
// TIMELINE_WAIT (succeeding spuriously because TIMELINE_WAIT's struct
// has extra `points` after `handles`, which our SYNCOBJ_WAIT struct
// padded with zeros — a binary-syncobj is "signaled at point 0" by
// definition), and TIMELINE_WAIT was hitting SYNCOBJ_EVENTFD which
// rejected our struct shape with EINVAL. This is why Worker-class
// kernels appeared "flaky" — we were never actually fence-waiting,
// just sleeping 100ms via the fall-through.
const DRM_IOCTL_SYNCOBJ_WAIT_NR: u8 = 0xC3;
const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NR: u8 = 0xCA;

pub const SYNCOBJ_WAIT_FLAGS_WAIT_ALL: u32 = 1 << 0;
pub const SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;
pub const SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE: u32 = 1 << 2;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmSyncobjWait {
    pub handles: u64,
    /// Absolute ns timestamp (CLOCK_MONOTONIC). 0 = poll, MAX = block.
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
}

pub fn drm_ioctl_syncobj_wait() -> libc::c_ulong {
    drm_iowr_generic::<DrmSyncobjWait>(DRM_IOCTL_SYNCOBJ_WAIT_NR)
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmSyncobjTimelineWait {
    pub handles: u64,
    pub points: u64,
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
}

pub fn drm_ioctl_syncobj_timeline_wait() -> libc::c_ulong {
    drm_iowr_generic::<DrmSyncobjTimelineWait>(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NR)
}

// ─── DRM PRIME (dmabuf import/export) ──────────────────────────────────
//
// Generic across DRM drivers. Used to share BOs between amdgpu (iGPU)
// and amdxdna (NPU) without copying — the architectural moat for UMA
// dual-engine on Strix Halo.

const DRM_IOCTL_PRIME_HANDLE_TO_FD_NR: u8 = 0x2D;
const DRM_IOCTL_PRIME_FD_TO_HANDLE_NR: u8 = 0x2E;

pub const PRIME_FLAG_RDWR: u32 = libc::O_RDWR as u32;
pub const PRIME_FLAG_CLOEXEC: u32 = libc::O_CLOEXEC as u32;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct DrmPrimeHandle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
}

pub fn drm_ioctl_prime_handle_to_fd() -> libc::c_ulong {
    drm_iowr_generic::<DrmPrimeHandle>(DRM_IOCTL_PRIME_HANDLE_TO_FD_NR)
}

pub fn drm_ioctl_prime_fd_to_handle() -> libc::c_ulong {
    drm_iowr_generic::<DrmPrimeHandle>(DRM_IOCTL_PRIME_FD_TO_HANDLE_NR)
}
