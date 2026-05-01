//! Phase 1.0 — BO round-trip.
//!
//! Allocate a SHMEM BO, mmap it, write a pattern, SYNC TO_DEVICE then
//! FROM_DEVICE, verify pattern survives. Prints xdna_addr (the NPU's
//! view of the buffer) so we can confirm IOMMU/SVM is mapping pages
//! into the device VA space.

use std::process::ExitCode;
use xdna_compute::bo::Bo;
use xdna_compute::device::Device;
use xdna_compute::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};

const SIZE: usize = 64 * 1024;
const MAGIC: u32 = 0xCAFEF00D;

fn main() -> ExitCode {
    let dev = match Device::open(None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut bo = match Bo::alloc_shmem(dev.fd, SIZE) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("alloc_shmem: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[bo] handle={} size={} xdna_addr={:#x}", bo.handle, bo.size, bo.xdna_addr);

    let buf = match bo.map() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("map: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[bo] mmapped at {:p}", buf.as_ptr());

    // Fill with a deterministic pattern: u32 indices xor'd with MAGIC.
    let n = buf.len() / 4;
    for i in 0..n {
        let v = (i as u32).wrapping_mul(2654435761) ^ MAGIC;
        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let first_word = u32::from_le_bytes(buf[..4].try_into().unwrap());
    println!("[bo] wrote {n} words, first={:#x}", first_word);

    // SYNC_BO TO_DEVICE: cache-flush hint before device read. On x86
    // cache-coherent with IOMMU it's effectively a no-op success.
    match bo.sync(SYNC_TO_DEVICE) {
        Ok(()) => println!("[bo] SYNC_BO TO_DEVICE ok"),
        Err(e) => {
            eprintln!("SYNC TO_DEVICE: {e}");
            return ExitCode::FAILURE;
        }
    }

    // SYNC_BO FROM_DEVICE on SHMEM is rejected by the driver (EINVAL =
    // 22): SHMEM is host-pinned, the device writes through IOMMU into
    // the same pages we mmap'd, so there's no device-side copy to
    // fetch back. We log the expected EINVAL but keep going; for
    // DEV/DEV_HEAP BOs (later phases) this call will be required.
    match bo.sync(SYNC_FROM_DEVICE) {
        Ok(()) => println!("[bo] SYNC_BO FROM_DEVICE ok (unexpected — note for phase 2)"),
        Err(e) if e.code == 22 => {
            println!("[bo] SYNC_BO FROM_DEVICE rejected (EINVAL) — expected for SHMEM");
        }
        Err(e) => {
            eprintln!("SYNC FROM_DEVICE: {e}");
            return ExitCode::FAILURE;
        }
    }

    let buf = bo.map().expect("re-map");
    let mut bad = 0usize;
    for i in 0..n {
        let want = (i as u32).wrapping_mul(2654435761) ^ MAGIC;
        let got = u32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
        if got != want {
            if bad < 4 {
                eprintln!("  word {i}: got {got:#x} want {want:#x}");
            }
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("[bo] FAIL: {bad}/{n} words corrupted");
        return ExitCode::FAILURE;
    }
    println!("[bo] PASS: {n}/{n} words intact after sync round-trip");
    ExitCode::SUCCESS
}
