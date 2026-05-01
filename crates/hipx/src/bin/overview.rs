//! `hipx-overview` — high-level smoke test.
//!
//! Opens the runtime, prints the SoC classification, allocates a
//! hwctx + a SHMEM BO, runs the BO sync round-trip, and tears
//! everything down. Single-shot diagnostic for "is hipx fully
//! plumbed end-to-end on this machine".

use std::process::ExitCode;

use hipx::dispatch::{
    classify, cols_available, core_rows, has_npu, npu_int8_tops, NpuFamily,
};
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::SYNC_TO_DEVICE;
use hipx::Hipx;

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Hipx::open failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("== hipx overview ==");
    let info = &hipx.info;
    let family = classify(info);
    println!("  SoC family:    {family:?}");
    println!("  AIE version:   {}.{}", info.aie_version.0, info.aie_version.1);
    println!("  Cols × rows:   {} × {}", info.aie_cols, info.aie_rows);
    println!("  Core tiles:    row_start={} row_count={}", info.core_tiles.0, info.core_tiles.1);
    println!("  Mem tiles:     row_start={} row_count={}", info.mem_tiles.0, info.mem_tiles.1);
    println!("  Shim tiles:    row_start={} row_count={}", info.shim_tiles.0, info.shim_tiles.1);
    println!("  NPU clock:     {} MHz", info.mp_npu_clock_mhz);
    println!("  H clock:       {} MHz", info.h_clock_mhz);
    println!("  Firmware:      {}.{}.{} build {}",
             info.firmware_version.0, info.firmware_version.1,
             info.firmware_version.2, info.firmware_version.3);
    println!("  TOPS max/curr: {}/{} INT8", info.npu_tops_max, info.npu_tops_curr);
    println!("  Tasks max:     {}", info.npu_task_max);
    println!("  Power mode:    {}", info.power_mode);
    println!();
    println!("  has_npu:       {}", has_npu(info));
    println!("  cols available: {}", cols_available(info));
    println!("  core_rows:     {}", core_rows(info));
    println!("  npu_int8_tops: {}", npu_int8_tops(info));

    if matches!(family, NpuFamily::Unknown) {
        println!("\n[overview] family Unknown — refusing dispatch test");
        return ExitCode::SUCCESS;
    }

    println!("\n  DEV_HEAP: handle={} size={} MB xdna_addr={:#x}",
             hipx.heap.handle, hipx.heap.size / (1024 * 1024), hipx.heap.xdna_addr);

    println!("\n== dispatch smoke ==");
    let mut b = HwctxBuilder::default();
    b.num_columns = 1;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create_hwctx failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("  hwctx: handle={} syncobj={} doorbell={:#x}",
             ctx.handle, ctx.syncobj_handle, ctx.umq_doorbell);

    let mut bo = match hipx.alloc_shmem(64 * 1024) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_shmem: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (b0, b4095) = {
        let buf = match bo.map() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("map: {e}");
                return ExitCode::FAILURE;
            }
        };
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        (buf[0], buf[4095])
    };
    if let Err(e) = bo.sync(SYNC_TO_DEVICE) {
        eprintln!("sync: {e}");
        return ExitCode::FAILURE;
    }
    println!("  shmem BO: handle={} size={} byte 0={} byte 4095={}",
             bo.handle, bo.size, b0, b4095);

    println!("\n[overview] PASS");
    ExitCode::SUCCESS
}
