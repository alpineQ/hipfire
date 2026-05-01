# hipx — hipfire-x for AMD AIE-2P (Strix Halo NPU)

Rust-native NPU compute for AMD's XDNA / AIE-2P NPU. Direct ioctl
bypass of the AMD Ryzen AI userspace and XRT runtime — analogous to
how `redline` bypasses `libdrm_amdgpu` / HIP for the iGPU side.

Lives at the same architectural level as `rdna-compute` for the
RDNA iGPUs. Engine code routes per-op via `engine::npu::route()`;
NPU offload is opportunistic, never load-bearing — every call site
falls back to iGPU when the NPU runtime isn't initialized OR no
PDI exists for that op class.

## What works (verified on hipx — Ryzen AI MAX+ 395 / NPU5)

| Kernel              | Partition | Dataflow                   | Status |
|---------------------|-----------|----------------------------|--------|
| `passthrough_4k`    | 8 cols    | objectfifo direct          | ✅ PASS |
| `passthrough_dmas`  | 1 col     | MemTile-routed DMA         | ✅ PASS |
| `vec_scalar_mul`    | 8 cols    | Worker + core int compute  | ❌ — diagnostic open (see below) |

Engine-side smoke test:

```
$ cargo run --features deltanet,npu --example hipfire_x_init
[hipfire-x] NPU detected:
  family:        Aie2p
  cols:          8
  TOPS (INT8):   58
[hipfire-x] route() smoke test:
  KV codec (asym3)        -> Igpu  (no PDI)
  INT8 GEMM 9B prefill    -> Igpu  (no PDI)
  ...
[hipfire-x] engine-API NPU dispatch (passthrough_4k):
  PASS — 4096 bytes round-tripped through NPU
```

## Architecture

```
hipx/
├── ioctl.rs          — full 9-ioctl ABI mirror of <drm/amdxdna_accel.h>
│                       + DRM PRIME, syncobj wait/timeline, GEM_CLOSE
├── device.rs         — open /dev/accel/accel0 + GET_INFO probes
├── bo.rs             — Bo lifecycle: SHMEM | DEV_HEAP | DEV | CMD
│                       — DEV_HEAP uses XRT-style anon-reserve +
│                          MAP_FIXED|MAP_LOCKED at heap-aligned addr
├── hwctx.rs          — Hwctx lifecycle (CREATE_HWCTX / DESTROY_HWCTX)
├── ert.rs            — ERT_START_CU command-packet builder
│                       (ert_start_kernel_cmd, npu_data, regmap)
├── cmd.rs            — config_cus + submit_exec_cmd
├── fence.rs          — wait_many + timeline_wait (DRM syncobj)
├── prime.rs          — dmabuf export/import (PRIME ioctls)
├── runtime.rs        — high-level Hipx wrapper (Device + heap + ergonomics)
├── dispatch.rs       — predicates: NpuFamily, has_npu, npu_int8_tops,
│                       compute_target(op, info, pdi_avail) → ComputeTarget
├── kernels.rs        — embedded PDI bytes for compiled CUs
└── bin/
    ├── probe.rs               — DRM_AMDXDNA_GET_INFO dump
    ├── overview.rs            — high-level smoke test using runtime.rs
    ├── bo_roundtrip.rs        — Phase-1 BO regression
    ├── hwctx_roundtrip.rs     — 40-combo hwctx parameter sweep
    ├── passthrough.rs         — Phase-2 first end-to-end NPU dispatch
    ├── passthrough_dmas.rs    — MemTile DMA forwarding
    └── vec_scalar_mul.rs      — Worker-class (debugging in progress)
```

## Building from source

Setup script for new Strix Halo boxes: `scripts/setup-hipx-npu.sh`.

Required:
- AMD's OOT amdxdna driver (≥ v1.0.0) via DKMS — the in-tree v0.7.0
  in Linux 7.0 mainline doesn't support page-locked heap mmap that
  the firmware needs.
- `RLIMIT_MEMLOCK` = unlimited for the user (already set in
  `/etc/security/limits.d/90-kaden-memlock.conf` on hipx).

Optional (for compiling new kernels):
- MLIR-AIE / Peano toolchain — bootstrapped via uv-managed Python
  3.12 in `~/mlir-aie/ironenv` (Ubuntu 26.04 ships only 3.14).

## Open Worker-class blocker

`vec_scalar_mul` (the simplest IRON example with actual core compute
— Worker + ObjectFifo + scale function) submits cleanly to the
firmware and the job completes (`total completed jobs N` in dmesg),
but the worker tile never writes to the output BO. Verified via
in-kernel printk (`hipx-cmddump`) that our cmd packet is byte-
identical in shape to AMD's working version:

- Same header (`30010001`: state=NEW, count=16, opcode=ERT_START_CU=0,
  type=ERT_CU=3)
- Same cu_mask (`0x1`)
- Same all-five-BO arg layout (bo0..bo4 each non-zero host VAs;
  bo3/bo4 even when not logically used by the kernel — XRT
  allocates placeholder BOs and we now do too)
- Same kernel-id (0x901), same ops/cycle (2048)

Only delta is the `instr_ptr` xdna_addr (ours 0x04020000 vs AMD's
0x04028000) and other heap-relative offsets, due to the OOT
driver's internal allocations consuming different heap slots
between our flow and XRT's.

The Worker tile's compiled code may expect a specific layout
relationship between PDI BO offset and instr BO offset that XRT's
allocation sequence happens to satisfy. Padding our DEV allocations
shifts the offsets but keeps the same gap between them.

This blocks INT8 GEMM and KV-codec dequant (both Worker-class). Path
forward: deeper inspection of how XRT's `xrt::module` lays out BOs
relative to each other, or kernel-level instrumentation to log the
firmware-side message bodies (not just the cmd_bo bytes).

The DMA-only kernel classes (passthrough_4k, passthrough_dmas) are
unaffected — they don't use core tiles, just shim+mem DMAs.
