//! `hipfire_x_dmabuf_hsa` — prove HIP/HSA buffer ↔ NPU dmabuf path.
//!
//! Run on hipx with the `npu` feature:
//!   cargo run -p engine --features npu --example hipfire_x_dmabuf_hsa
//!
//! What this proves end-to-end:
//!
//!   1. HIP allocation via `hipMalloc` (the engine's normal path for
//!      K cache, weights, scratch).
//!   2. CPU-staged write into the HIP device buffer via `hipMemcpy`.
//!   3. `hsa_amd_portable_export_dmabuf` on the HIP device pointer
//!      returns a dmabuf fd. (HIP is implemented on top of HSA, so
//!      every HIP device pointer is a valid HSA pointer.)
//!   4. `hipx::Bo::from_imported_dmabuf` ingests that fd through
//!      amdxdna's PRIME_FD_TO_HANDLE.
//!   5. The amdxdna-side mmap returns a CPU view of the *same*
//!      physical pages as the HIP device pointer (UMA SoC).
//!   6. After SYNC_BO(TO_DEVICE) on the imported BO, the bytes the
//!      iGPU wrote via memcpy are visible from the NPU side.
//!   7. Reverse direction: bytes written via the imported BO's mmap
//!      are visible from the HIP device pointer.
//!
//! With this path validated, any engine-allocated K cache slice
//! becomes NPU-reachable by going through `hsa_dmabuf::export +
//! Bo::from_imported_dmabuf`. No copy. No bounce buffer. The asym3
//! codec kernel reads engine-resident K cache directly.

#[cfg(feature = "npu")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use hip_bridge::HipRuntime;
    use hipx::hsa_dmabuf;
    use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
    use std::os::fd::AsRawFd;

    const SIZE: usize = 1 << 20; // 1 MiB

    println!("[hsa-dmabuf] HSA dmabuf export available: {}", hsa_dmabuf::available());

    if !hsa_dmabuf::available() {
        return Err(
            "libhsa-runtime64 not loadable; check ROCm install".into(),
        );
    }

    println!("[hsa-dmabuf] init HIP runtime");
    let hip = HipRuntime::load().map_err(|e| format!("HipRuntime::load: {e}"))?;
    hip.set_device(0).map_err(|e| format!("set_device(0): {e}"))?;

    println!("[hsa-dmabuf] hipMalloc({SIZE})");
    let dev = hip.malloc(SIZE).map_err(|e| format!("malloc: {e}"))?;
    let dev_ptr = dev.as_ptr() as *const std::ffi::c_void;
    println!("  dev_ptr = {dev_ptr:?}");

    // Write a known pattern: byte i = (i * 31 + 7) & 0xff
    let mut host_in: Vec<u8> = (0..SIZE)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) & 0xff) as u8)
        .collect();
    hip.memcpy_htod(&dev, &host_in)
        .map_err(|e| format!("memcpy h2d: {e}"))?;
    hip.device_synchronize().map_err(|e| format!("dev sync: {e}"))?;

    println!("[hsa-dmabuf] hsa_amd_portable_export_dmabuf...");
    let (dmabuf, offset) = hsa_dmabuf::export(dev_ptr, SIZE)?;
    let dmabuf_fd = dmabuf.as_raw_fd();
    println!("  dmabuf_fd = {dmabuf_fd}, offset = {offset}");

    if offset != 0 {
        // We only care about the first SIZE bytes from offset 0; HSA
        // can pack multiple allocations into one underlying GTT region.
        // For our PoC the offset should be 0 since we just allocated.
        println!(
            "  WARN: HSA reports offset={offset}, but our allocation should start at 0. \
             Test will compare bytes at offset; if mismatch, HSA packs allocations differently \
             than we assumed."
        );
    }

    println!("[hsa-dmabuf] open NPU + import dmabuf via Bo::from_imported_dmabuf");
    let hipx_dev = hipx::Hipx::open().map_err(|e| format!("hipx open: {e}"))?;
    let mut npu_bo =
        hipx::Bo::from_imported_dmabuf(hipx_dev.device.fd, dmabuf_fd, SIZE + offset as usize)
            .map_err(|e| format!("Bo::from_imported_dmabuf: {e}"))?;
    let handle = npu_bo.handle;
    {
        let npu_view = npu_bo
            .map()
            .map_err(|e| format!("npu_bo.map: {e}"))?;
        println!(
            "  NPU handle={handle}, mmap'd {} bytes at {:p}",
            npu_view.len(),
            npu_view.as_ptr()
        );
    }

    // SYNC_BO(TO_DEVICE) ensures the imported BO's pages are enrolled
    // in the NPU's IOMMU domain. Required even though we're only
    // doing CPU reads here — keeps the test parallel to how the kernel
    // dispatch path would use it.
    npu_bo.sync(SYNC_TO_DEVICE)
        .map_err(|e| format!("sync to device: {e}"))?;

    // Verify forward direction: HIP wrote → NPU mmap reads same bytes.
    let mut errors = 0usize;
    {
        let npu_view = npu_bo.map().expect("re-map");
        let view = &npu_view[offset as usize..offset as usize + SIZE];
        for i in [0usize, 1, 1024, SIZE / 2, SIZE - 4096, SIZE - 1] {
            let want = ((i.wrapping_mul(31).wrapping_add(7)) & 0xff) as u8;
            if view[i] != want {
                errors += 1;
                eprintln!(
                    "  forward mismatch i={i}: want={want:#x} got={:#x}",
                    view[i]
                );
            }
        }
    }
    if errors == 0 {
        println!("  forward (HIP→NPU view): bytes match at 6 sample positions");
    } else {
        return Err(format!("forward direction: {errors} mismatches").into());
    }

    // Reverse direction: write through NPU mmap, read back through HIP.
    {
        let npu_view = npu_bo.map().expect("re-map");
        let view_mut = &mut npu_view[offset as usize..offset as usize + SIZE];
        for (i, b) in view_mut.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(17) ^ 0xa5) & 0xff) as u8;
        }
    }
    npu_bo.sync(SYNC_FROM_DEVICE)
        .map_err(|e| format!("sync from device: {e}"))?;
    hip.memcpy_dtoh(&mut host_in, &dev)
        .map_err(|e| format!("memcpy d2h: {e}"))?;
    hip.device_synchronize().map_err(|e| format!("dev sync: {e}"))?;
    let mut errors = 0usize;
    for i in [0usize, 1, 1024, SIZE / 2, SIZE - 4096, SIZE - 1] {
        let want = ((i.wrapping_mul(17) ^ 0xa5) & 0xff) as u8;
        if host_in[i] != want {
            errors += 1;
            eprintln!(
                "  reverse mismatch i={i}: want={want:#x} got={:#x}",
                host_in[i]
            );
        }
    }
    if errors == 0 {
        println!("  reverse (NPU view→HIP→host): bytes match at 6 sample positions");
    } else {
        return Err(format!("reverse direction: {errors} mismatches").into());
    }

    println!("\n=== HIP/HSA → dmabuf → NPU: PASS ===");
    println!(
        "Strix Halo iGPU (HIP/HSA) ↔ NPU (amdxdna) UMA sharing is open. \
         Engine-allocated buffers can be exposed to NPU kernels with no copy."
    );

    drop(npu_bo);
    drop(dmabuf);
    Ok(())
}

#[cfg(not(feature = "npu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("Build with --features npu to exercise the dmabuf path on Strix Halo.");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
