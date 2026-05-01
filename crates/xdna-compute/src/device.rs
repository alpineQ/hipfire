//! Device open + GET_INFO probes.
//!
//! Phase 0: read-only metadata. No BO/hwctx/exec — those arrive in Phase 1
//! once the ABI is verified.

use crate::ioctl::*;
use crate::{Result, XdnaError};
use std::ffi::CString;
use std::mem::MaybeUninit;

const DEFAULT_PATH: &str = "/dev/accel/accel0";

/// An open amdxdna device.
pub struct Device {
    pub fd: i32,
}

/// Parsed device metadata. Whatever the driver returns from GET_INFO.
#[derive(Debug, Clone)]
pub struct NpuInfo {
    pub aie_version: (u32, u32),
    pub aie_cols: u16,
    pub aie_rows: u16,
    pub aie_col_size: u32,
    pub core_tiles: (u16, u16), // (start_row, row_count)
    pub mem_tiles: (u16, u16),
    pub shim_tiles: (u16, u16),
    pub mp_npu_clock_mhz: u32,
    pub h_clock_mhz: u32,
    pub firmware_version: (u32, u32, u32, u32), // major.minor.patch.build
    pub power_mode: u8,
    pub npu_tops_max: u64,
    pub npu_task_max: u64,
    pub npu_tops_curr: u64,
    pub npu_task_curr: u64,
}

impl Device {
    /// Open `/dev/accel/accel0` (or override).
    pub fn open(path: Option<&str>) -> Result<Self> {
        let p = path.unwrap_or(DEFAULT_PATH);
        let cpath = CString::new(p).unwrap();
        let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(XdnaError {
                code: errno,
                message: format!(
                    "open {p} failed (errno={errno}). Need 'render' group membership."
                ),
            });
        }
        Ok(Self { fd })
    }

    /// Run a GET_INFO query, returning the populated buffer.
    fn get_info<T: Default + Copy>(&self, param: u32) -> Result<T> {
        let mut buf = MaybeUninit::<T>::zeroed();
        let size = std::mem::size_of::<T>() as u32;
        let mut req = DrmGetInfo {
            param,
            buffer_size: size,
            buffer: buf.as_mut_ptr() as u64,
        };
        let ret = unsafe {
            libc::ioctl(
                self.fd,
                drm_ioctl_amdxdna_get_info(),
                &mut req as *mut _ as *mut libc::c_void,
            )
        };
        if ret != 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(XdnaError {
                code: errno,
                message: format!("GET_INFO param={param} ioctl failed (errno={errno})"),
            });
        }
        Ok(unsafe { buf.assume_init() })
    }

    pub fn query_aie_version(&self) -> Result<QueryAieVersion> {
        self.get_info::<QueryAieVersion>(QUERY_AIE_VERSION)
    }

    pub fn query_aie_metadata(&self) -> Result<QueryAieMetadata> {
        self.get_info::<QueryAieMetadata>(QUERY_AIE_METADATA)
    }

    pub fn query_clock_metadata(&self) -> Result<QueryClockMetadata> {
        self.get_info::<QueryClockMetadata>(QUERY_CLOCK_METADATA)
    }

    pub fn query_firmware_version(&self) -> Result<QueryFirmwareVersion> {
        self.get_info::<QueryFirmwareVersion>(QUERY_FIRMWARE_VERSION)
    }

    pub fn query_resource_info(&self) -> Result<GetResourceInfo> {
        self.get_info::<GetResourceInfo>(QUERY_RESOURCE_INFO)
    }

    pub fn get_power_mode(&self) -> Result<GetPowerMode> {
        self.get_info::<GetPowerMode>(GET_POWER_MODE)
    }

    /// One-shot helper that runs every Phase-0 probe.
    pub fn probe(&self) -> Result<NpuInfo> {
        let v = self.query_aie_version()?;
        let m = self.query_aie_metadata()?;
        let c = self.query_clock_metadata()?;
        let f = self.query_firmware_version()?;
        let r = self.query_resource_info()?;
        let pm = self.get_power_mode()?;
        Ok(NpuInfo {
            aie_version: (v.major, v.minor),
            aie_cols: m.cols,
            aie_rows: m.rows,
            aie_col_size: m.col_size,
            core_tiles: (m.core.row_start, m.core.row_count),
            mem_tiles: (m.mem.row_start, m.mem.row_count),
            shim_tiles: (m.shim.row_start, m.shim.row_count),
            mp_npu_clock_mhz: c.mp_npu_clock.freq_mhz,
            h_clock_mhz: c.h_clock.freq_mhz,
            firmware_version: (f.major, f.minor, f.patch, f.build),
            power_mode: pm.power_mode,
            npu_tops_max: r.npu_tops_max,
            npu_task_max: r.npu_task_max,
            npu_tops_curr: r.npu_tops_curr,
            npu_task_curr: r.npu_task_curr,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.fd >= 0 {
            unsafe { libc::close(self.fd) };
        }
    }
}
