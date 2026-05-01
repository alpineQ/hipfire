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

Bring-up tier:

| Kernel              | Partition | Dataflow                   | Status |
|---------------------|-----------|----------------------------|--------|
| `passthrough_4k`    | 8 cols    | objectfifo direct          | ✅ PASS |
| `passthrough_dmas`  | 1 col     | MemTile-routed DMA         | ✅ PASS |
| `vec_scalar_mul`    | 8 cols    | Worker + core int compute  | ✅ PASS (50/50 multi-iter) |

Performance tier:

| Kernel              | Shape          | Standalone        | Engine API zero-copy |
|---------------------|----------------|-------------------|----------------------|
| `matvec`            | 288×288 i16    | 0.31 GOp/s         | 0.30 GOp/s            |
| `matmul-512`        | 512^3 i16      | **1.04 TOp/s**     | 0.43 TOp/s            |
| `matmul-i8`         | 512^3 i8       | **1.97 TOp/s**     | 1.10 TOp/s            |
| `matmul-i8-1024`    | 1024^3 i8      | **4.46 TOp/s**     | 2.23 TOp/s            |
| `matmul-i8-2048`    | 2048^3 i8      | 4.36 TOp/s         | (no engine API)       |
| `matmul-bf16`       | 512^3 bf16     | 0.83 TOp/s         | (no engine API)       |
| `matmul-bf16-1024`  | 1024^3 bf16    | **0.97 TOp/s**     | **1.03 TOp/s**        |

The 4.46 TOp/s INT8 figure is sustained — 50 iterations, 482 µs mean
dispatch, 2.1 GMACs per call. Strix Halo NPU peak is ~50 TOPS INT8,
so we sit at ~9% peak on a generic non-tuned MLIR-AIE kernel. The
remaining 91% is hand-tuned MAC inner loops + larger tile sizes —
follow-up work, not bring-up.

The bf16 kernels deliver bit-perfect output vs host f32 reference
(small powers-of-2 inputs over deep accumulation fit exactly in
bf16). They're the natural offload target for FP16 hot-path tensors
(attention scoring, hidden-state projection); INT8 stays the speed
champion for quantized-weight workloads.

**Concurrent dispatch demo** (`hipfire_x_concurrent` example):

```
NPU 1024^3 i8 alone:                    1015 µs/op  (2.13 TOp/s)
iGPU 32 MiB memset alone:                 51 µs/op
NPU + 32 MiB iGPU concurrent:            985 µs/op  ← negative cost
```

Adding a 32 MiB iGPU memset to the NPU matmul makes the wall-clock
*shorter* than NPU alone — the iGPU dispatch fits entirely inside
the NPU compute window. This is what "free overlap" means in the
size-matched regime that real LLM workloads operate in.

**Multi-layer pipeline demo** (`hipfire_x_pipeline` example, 28
layers ≈ 27B Gemma depth, per-layer iGPU work matched to NPU
dispatch time):

```
A. iGPU-only forward:           33184 us total (1185 us/layer baseline)
B. iGPU + NPU serial:           67881 us total (2424 us/layer)
C. iGPU + NPU pipelined:        42308 us total (1511 us/layer)
saved by pipelining:            25573 us (37% wall-clock vs serial)
NPU added cost vs baseline:       326 us/layer (vs 1239 us if serial)
NPU effective throughput:        1.42 TOp/s INT8 sustained
```

Read: adding 1.42 TOp/s of NPU compute to a realistic per-layer
LLM workflow costs only 326 µs per layer of additional wall-clock
— 26% of what serial dispatch would cost. **74% of the NPU work
is hidden behind iGPU work** via the previous-wait/current-iGPU
overlap pattern + no-copy NPU wait + zero-copy A/B (caller pre-
fills once via `_a_buf()`/`_b_buf()` and calls `_sync_inputs()`,
then `_submit_zero_copy()` skips the A/B SYNC_BO ioctls per
dispatch).

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

## Worker-class status: deterministic

`vec_scalar_mul` (Worker + ObjectFifo + core int multiply) dispatches
end-to-end and produces correct output (`output[i] = input[i] * 7 (i16)`
for 4096 elements) across 50/50 in-process iterations.

Three bugs combined to make Worker-class kernels look "flaky" during
bring-up:

1. **Wrong DRM ioctl numbers**: `SYNCOBJ_WAIT_NR` was 0xCA (should be
   0xC3); `TIMELINE_WAIT_NR` was 0xCF (should be 0xCA). Per `<drm/drm.h>`
   uapi. SYNCOBJ_WAIT was hitting TIMELINE_WAIT (extra `points` field
   zero-padded looked "signaled at point 0"); TIMELINE_WAIT was hitting
   SYNCOBJ_EVENTFD which EINVAL'd, leaving the binaries to fall through
   to a 100 ms safety sleep. The "first run PASS, later runs FAIL"
   pattern was firmware completing in <100 ms cold and missing on the
   warmer iterations.
2. **Cmd-packet state field**: stays at COMPLETED after each submit;
   firmware skips re-execution of a still-COMPLETED packet. Added
   `ert::reset_state(&mut buf[..4])` to patch the state nibble back to
   NEW between submissions. XRT does this implicitly when it rebuilds
   the cmd packet each time; we re-use the BO for perf.
3. **SYNC_BO required for SHMEM-backed BOs**: the prior code had a
   comment claiming "PASID + cache-coherent x86" obviated `SYNC_BO`
   calls — wrong. AMD's working test does them, `passthrough_dmas`
   (which works) does them; only `vec_scalar_mul` had skipped them.
   The worker tile's writes don't propagate to the host CPU view
   without `DRM_IOCTL_AMDXDNA_SYNC_BO {SYNC_FROM_DEVICE}`.

With those three fixed, Worker-class is production-quality. KV-codec /
INT8 GEMM kernels can be authored against this surface today.
