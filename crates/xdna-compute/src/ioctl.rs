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
