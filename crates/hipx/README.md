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
| `vec_scalar_mul`    | 8 cols    | Worker + core int compute  | ✅ PASS (flaky, fix pending) |

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

## Worker-class status: works (flaky)

`vec_scalar_mul` (Worker + ObjectFifo + core int multiply) now
dispatches end-to-end and produces correct output (`output[i] =
input[i] * 7 (i16)` for 4096 elements). The decisive find: a kernel
printk patch we'd added during diagnostics was calling
`amdxdna_gem_vmap()` on the cmd BO, which created a kernel-space
mapping that interfered with the firmware's IOMMU mapping for
Worker-class kernels. Reverting the patch unblocked the dispatch.

The dispatch is currently flaky — first run after a fresh module
load reliably PASSes; multi-iteration in-process tests show some
iterations PASS and some partial-fail (e.g. 480/4096 mismatches —
88% correct, suggesting writes start but don't complete). The
remaining work is sync semantics — proper `SYNCOBJ_TIMELINE_WAIT`
usage (currently we always EINVAL and fall through to a 100ms sleep)
or a state reset between submissions.

The flakiness gates the press-coverage perf number but not the
architectural moat — KV-codec / INT8 GEMM kernels can be authored
and dispatched today; they'll just need the sync fix to ship as
production-quality offload.
