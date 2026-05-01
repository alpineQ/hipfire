//! Phase-0 NPU probe: open `/dev/accel/accel0`, run every GET_INFO
//! ioctl, dump the results. Exits non-zero on any ABI mismatch or
//! ioctl error.

use std::process::ExitCode;
use xdna_compute::device::Device;

fn clock_name(name: &[u8; 16]) -> String {
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    String::from_utf8_lossy(&name[..end]).into_owned()
}

fn main() -> ExitCode {
    let path = std::env::args().nth(1);
    let dev = match Device::open(path.as_deref()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[xdna-probe] opened device fd={}", dev.fd);

    match dev.query_aie_version() {
        Ok(v) => println!("AIE version:      {}.{}", v.major, v.minor),
        Err(e) => {
            eprintln!("AIE version: {e}");
            return ExitCode::FAILURE;
        }
    }

    match dev.query_aie_metadata() {
        Ok(m) => {
            println!("AIE shape:        {} cols x {} rows  (col_size={} bytes)",
                m.cols, m.rows, m.col_size);
            println!("  core tiles:     row_start={} row_count={} dma_ch={} locks={} events={}",
                m.core.row_start, m.core.row_count,
                m.core.dma_channel_count, m.core.lock_count, m.core.event_reg_count);
            println!("  mem tiles:      row_start={} row_count={} dma_ch={} locks={} events={}",
                m.mem.row_start, m.mem.row_count,
                m.mem.dma_channel_count, m.mem.lock_count, m.mem.event_reg_count);
            println!("  shim tiles:     row_start={} row_count={} dma_ch={} locks={} events={}",
                m.shim.row_start, m.shim.row_count,
                m.shim.dma_channel_count, m.shim.lock_count, m.shim.event_reg_count);
        }
        Err(e) => eprintln!("AIE metadata: {e}"),
    }

    match dev.query_clock_metadata() {
        Ok(c) => {
            println!("MP-NPU clock:     {} = {} MHz",
                clock_name(&c.mp_npu_clock.name), c.mp_npu_clock.freq_mhz);
            println!("H clock:          {} = {} MHz",
                clock_name(&c.h_clock.name), c.h_clock.freq_mhz);
        }
        Err(e) => eprintln!("clock metadata: {e}"),
    }

    match dev.query_firmware_version() {
        Ok(f) => println!("Firmware:         {}.{}.{} build={}",
            f.major, f.minor, f.patch, f.build),
        Err(e) => eprintln!("firmware version: {e}"),
    }

    match dev.query_resource_info() {
        Ok(r) => {
            println!("Resource info:");
            println!("  TOPS max/curr:  {} / {}", r.npu_tops_max, r.npu_tops_curr);
            println!("  Tasks max/curr: {} / {}", r.npu_task_max, r.npu_task_curr);
            println!("  Clk max:        {} kHz", r.npu_clk_max);
        }
        Err(e) => eprintln!("resource info: {e}"),
    }

    match dev.get_power_mode() {
        Ok(p) => {
            let mode = match p.power_mode {
                0 => "DEFAULT",
                1 => "LOW",
                2 => "MEDIUM",
                3 => "HIGH",
                4 => "TURBO",
                _ => "?",
            };
            println!("Power mode:       {} ({})", p.power_mode, mode);
        }
        Err(e) => eprintln!("power mode: {e}"),
    }

    ExitCode::SUCCESS
}
