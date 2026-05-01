//! Phase 1.1 — hwctx create + destroy round-trip.
//!
//! Driver invariant: exactly ONE DEV_HEAP per client fd. Subsequent
//! CREATE_BO(DEV_HEAP) returns EBUSY. So we allocate the heap once
//! and sweep only (num_columns, max_opc, log_buf) for hwctx.

use std::process::ExitCode;
use xdna_compute::bo::Bo;
use xdna_compute::device::Device;
use xdna_compute::hwctx::{Hwctx, HwctxBuilder};

fn main() -> ExitCode {
    let dev = match Device::open(None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open: {e}");
            return ExitCode::FAILURE;
        }
    };

    let meta = match dev.query_aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let core_rows = meta.core.row_count as u32;
    println!("[hwctx] device: {} cols x {} rows; core_rows={core_rows}",
             meta.cols, meta.rows);

    // Single per-client DEV_HEAP. Try a moderate size first.
    let heap = match Bo::alloc_dev_heap(dev.fd, 64 * 1024 * 1024) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("DEV_HEAP alloc: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[hwctx] DEV_HEAP: handle={} size={} MB xdna_addr={:#x}",
             heap.handle, heap.size / (1024 * 1024), heap.xdna_addr);

    // Persistent log_buf so we can test with-log and without-log.
    let log_bo = match Bo::alloc_shmem(dev.fd, 1024 * 1024) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("log_bo alloc warning (continuing without): {e}");
            None
        }
    };
    let log_handle = log_bo.as_ref().map(|b| b.handle).unwrap_or(0);
    println!("[hwctx] log_buf_bo: handle={log_handle}");

    // Sweep (cols, max_opc, log) — heap is fixed.
    let mut wins: Vec<String> = Vec::new();
    for cols in &[1u32, 2, 4, 8] {
        for max_opc in &[0u32, 1, 1024, 65536, 0x10_0000] {
            for use_log in &[false, true] {
                let mut b = HwctxBuilder::default();
                b.num_columns = *cols;
                b.max_opc = *max_opc;
                b.log_buf_bo = if *use_log { log_handle } else { 0 };
                let label = format!("cols={cols} max_opc={max_opc} log={use_log}");
                match Hwctx::create(dev.fd, core_rows, &b) {
                    Ok(c) => {
                        println!(
                            "[hwctx] {label} → OK handle={} syncobj={} doorbell={:#x}",
                            c.handle, c.syncobj_handle, c.umq_doorbell
                        );
                        wins.push(label);
                        drop(c);
                        // Tiny sleep to let the firmware settle between
                        // create/destroy cycles.
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    Err(e) => {
                        println!("[hwctx] {label} → FAIL {}", e);
                    }
                }
            }
        }
    }

    drop(log_bo);
    drop(heap);

    println!("\n[hwctx] {} winning combinations:", wins.len());
    for w in &wins {
        println!("    {w}");
    }
    if wins.is_empty() { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
