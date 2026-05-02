# Phase 0 — XDNA / AIE-2P baseline (Strix Halo NPU)

Date: 2026-05-01
Worktree: `strix-halo` (gfx1151 + aie2p scope)
Host: `hipx` (Minisforum MS-S1 MAX, Ryzen AI MAX+ 395, kernel 7.0.0-14-generic)

## TL;DR

The amdxdna kernel driver is loaded, firmware is up, `/dev/accel/accel0` is
open, and the kernel uapi header (`/usr/include/drm/amdxdna_accel.h`, 9 DRM
ioctls) is shipped by the distro. **Zero userspace is installed** — no XRT,
no AMD Ryzen AI runtime, no ROCm-side NPU library. That's the bypass story:
hipfire opens the DRM accel node directly and drives the 9 ioctls itself,
exactly the way `redline` drives `/dev/dri/renderD128` for amdgpu.

The hard part is **not** the runtime. It's the **kernel artifact**: AIE-2P
doesn't have a public ISA we compile against. CUs (compute units) are
produced by MLIR-AIE / IRON / Peano and packaged as PDI binaries. This is a
compiler problem, not a driver problem.

## Live state on hipx (probed 2026-05-01)

```
=== amdxdna ===
amdxdna               172032  0
amd_pmf               131072  1 amdxdna
gpu_sched              69632  2 amdxdna,amdgpu
filename: /lib/modules/7.0.0-14-generic/kernel/drivers/accel/amdxdna/amdxdna.ko.zst
description: amdxdna driver
author: XRT Team <runtimeca39d@amd.com>
firmware: amdnpu/17f0_11/npu_7.sbin   <- AIE-2P / NPU5 = Strix Halo
alias:    pci:v00001022d000017F0...
license: GPL

=== /dev/accel ===
crw-rw---- 1 root render 261, 0 May  1 05:30 /dev/accel/accel0

=== boot ===
amdxdna 0000:be:00.1: [drm] Load firmware amdnpu/17f0_11/npu_7.sbin
amdxdna 0000:be:00.1: enabling device (0000 -> 0002)
[drm] Initialized amdxdna_accel_driver 0.7.0 for 0000:be:00.1 on minor 0

=== userspace ===
xrt:    none installed
xdna:   none installed
ryzen-ai: none installed
ldconfig | grep -iE "xrt|xdna|aie": empty
/opt/rocm-7.2.0/lib | grep -iE "xrt|xdna|aie|npu": empty

=== headers ===
/usr/include/drm/amdxdna_accel.h        (709 lines, GPL-2.0 WITH Linux-syscall-note)
/usr/src/linux-headers-7.0.0-14/include/uapi/drm/amdxdna_accel.h
```

PCI ID `1022:17f0` confirms this is the AMD NPU silicon present in Strix
Halo. Firmware filename `npu_7.sbin` is AMD's internal codename for AIE-2P
(NPU generation 5, ~50 TOPS INT8). Group is `render`, user `kaden` is
already a member.

## ABI surface (all 9 DRM ioctls)

From `/usr/include/drm/amdxdna_accel.h`:

| nr | ioctl                       | purpose                                              |
|----|-----------------------------|------------------------------------------------------|
| 0  | DRM_AMDXDNA_CREATE_HWCTX    | allocate AIE column partition; returns hwctx handle + syncobj |
| 1  | DRM_AMDXDNA_DESTROY_HWCTX   | tear down hwctx                                      |
| 2  | DRM_AMDXDNA_CONFIG_HWCTX    | bind CU configurations to a hwctx (CU = compiled kernel) |
| 3  | DRM_AMDXDNA_CREATE_BO       | allocate BO of type SHMEM/DEV_HEAP/DEV/CMD          |
| 4  | DRM_AMDXDNA_GET_BO_INFO     | query map_offset (mmap), VA, xdna device addr        |
| 5  | DRM_AMDXDNA_SYNC_BO         | cache sync TO_DEVICE / FROM_DEVICE                   |
| 6  | DRM_AMDXDNA_EXEC_CMD        | submit command BO(s) to a hwctx; returns sequence    |
| 7  | DRM_AMDXDNA_GET_INFO        | metadata queries (AIE size, clock, firmware, etc.)   |
| 8  | DRM_AMDXDNA_SET_STATE       | power mode, AIE mem/reg writes, preempt              |
| 10 | DRM_AMDXDNA_GET_ARRAY       | enumerate hwctxs / async errors                      |

Synchronization is via standard DRM syncobj fences (handle returned from
CREATE_HWCTX). No driver-specific fence protocol — this is the same
syncobj pool used by amdgpu / xe / nouveau, so once we have BO + syncobj
fds we can `dmabuf` them between iGPU and NPU.

`DRM_AMDXDNA_GET_INFO` enumerates the device shape — that's the natural
Phase-0 probe target (no command submission required, no PDI required,
just open + ioctl).

## Execution model

1. **Hardware context (hwctx)** = a slice of the AIE tile array (some
   number of columns), bound to a process. Allocated with QoS hints
   (priority, fps, latency, gops). Holds a user-mode queue (UMQ) doorbell
   + log buffer. Syncobj attached for fence completion.
2. **Compute Unit (CU)** = a pre-compiled kernel that runs on a tile
   subset. Multiple CUs per hwctx. The CU is supplied as a BO containing
   AMD's PDI (Platform Device Image) format — produced by the AIE
   compiler (Peano/MLIR-AIE backend), not by us, not by LLVM AMDGPU.
3. **Command** = a CMD-type BO containing arguments that point at input
   and output BOs. Submit with EXEC_CMD; wait on syncobj.
4. **BO types**:
   - `SHMEM` — pinned host pages, NPU sees them too (UMA on Strix Halo).
   - `DEV_HEAP` — device-managed heap region.
   - `DEV` — device-resident allocation inside DEV_HEAP.
   - `CMD` — command buffer (descriptor packet).

This is a **descriptor-driven** model, not a PM4-stream model. Closer to
how we'd dispatch on AIE than how we dispatch on RDNA. No SGPR/VGPR
layout to decode, no kernarg ABI to reconstruct.

## What the bypass actually buys us

Compared to "use AMD's Ryzen AI software stack":

| Concern                  | Ryzen AI stack         | hipfire bypass            |
|--------------------------|------------------------|---------------------------|
| Linux availability       | Preview, Windows-first | already shipping (kernel) |
| ROCm dependency          | None today, future TBD | none                      |
| Python in hot path       | Yes (ONNX/IRON)        | no (project rule)         |
| Build-time dependencies  | XRT + Vitis AI toolchain | bindgen + libc           |
| Cross-engine fusion w/ gfx1151 | unknown          | dmabuf import — trivial   |
| Versioning lock-in       | XRT API stability ABI  | DRM uapi (kernel-stable)  |

The DRM uapi is stable across kernel versions in the ABI sense (new
ioctls get higher nrs, existing struct layouts don't change). That's a
better bet than a vendor userspace whose Linux story is "preview".

## What's hard: the kernel-compiler problem

Unlike RDNA where we compile `.hip` → `.hsaco` ELF with embedded AMDGPU
ISA (LLVM backend, totally inside our toolchain), AIE-2P kernels need:

- **MLIR-AIE** (LLVM-based, OSS at github.com/Xilinx/mlir-aie) — emits
  AIE core code + DMA configurations.
- **Peano** — LLVM AIE backend (the actual codegen for AIE tiles).
- **PDI packager** — produces the binary blob loaded by the firmware as
  a CU.

Three viable paths for sourcing CU artifacts (ranked):

1. **Author kernels in IRON (Python frontend), compile offline, ship
   PDI bytes embedded in hipfire.** Same model as our `kernels/*.hip` →
   `.hsaco` bake step. Build-time only; runtime is pure Rust + ioctl.
   Lowest effort, highest portability across hipx/k9lin/CI.
2. **Vendor a pre-built MLIR-AIE/Peano toolchain in a hipfire docker
   image, compile at hipfire build time.** Higher cost; ties build to a
   specific MLIR-AIE commit. Avoid until path 1 hits a kernel we can't
   author in IRON.
3. **Embed Peano as a Rust dep and compile at runtime.** Mirrors
   `rdna-compute/src/compiler.rs` but the AIE compiler is heavier than
   amd-comgr. Probably never worth it.

Path 1 is the natural fit. The PDI is just bytes; bake them into a const
slice and pass to `CREATE_BO(type=AMDXDNA_BO_DEV_HEAP)`.

## Integration shape

Mirror the existing `redline` (KMD bypass) layer:

```
crates/
├── redline          ← amdgpu KMD bypass (existing)
│   └── /dev/dri/renderD128 + libdrm_amdgpu + PM4
├── xdna-compute     ← amdxdna KMD bypass (NEW)
│   └── /dev/accel/accel0 + 9 DRM ioctls + syncobj
├── rdna-compute     ← high-level RDNA dispatch (existing)
│   └── GpuTensor, Gpu, predicates (has_wmma_f16, ...)
└── (future)         ← high-level NPU dispatch
    └── NpuTensor, Npu, predicates (has_npu, npu_int8_tops, ...)
```

Whether we split bridge vs dispatch into two crates (mirror
`redline` + `rdna-compute`) or fold them into one (`xdna-compute`)
depends on size. Single crate first; split if it grows past ~3k LoC.

## Predicate analogues (for engine dispatch)

`crates/rdna-compute/src/dispatch.rs` defines arch-tagged predicates the
engine uses to pick a kernel. The NPU predicates would look like:

```rust
fn has_npu(soc: &str) -> bool   // "strix-halo" | "phoenix" | ... → true
fn npu_int8_tops(soc: &str) -> u32  // 50 for AIE-2P / Strix Halo
fn npu_max_columns(soc: &str) -> u32  // queried at runtime via GET_INFO
fn npu_uma_with_igpu(soc: &str) -> bool  // true for APUs (no PCIe copy)
```

Last predicate is the load-bearing one for Strix Halo. UMA-shared
memory + dmabuf-importable BOs means a tensor produced by gfx1151 can be
consumed by the NPU with no copy. That's the architectural win of the
SoC and the reason this work isn't just "another backend".

## First viable workload

INT8 GEMV/GEMM is the AIE sweet spot (the silicon literally exists for
INT8 MAC). Concrete first kernel: a tiled `int8 matmul` with i32
accumulator, taking BF16 inputs (quantize on shim, accumulate in tile),
sized for Q4_K-style group-quant prefill matmuls hipfire already runs
on the iGPU. Expected outcome (theoretical):

- AIE-2P @ ~50 TOPS INT8 vs gfx1151 @ ~28 TFLOPS FP16
- For prefill (compute-bound, INT8-friendly), NPU ≥ iGPU on TOPS alone
- For decode (memory-bound, FP16 KV cache), iGPU still wins
- Strategy: route prefill to NPU, decode to iGPU, KV via UMA dmabuf

## Phase-0 exit criterion — ✅ PASSED 2026-05-01

`crates/xdna-compute/src/bin/probe.rs` opens `/dev/accel/accel0` and runs
all six GET_INFO queries. First-try result on hipx:

```
AIE version:      1.1
AIE shape:        8 cols x 6 rows  (col_size=504 bytes)
  core tiles:     row_start=2 row_count=4 dma_ch=2 locks=16 events=4
  mem tiles:      row_start=1 row_count=1 dma_ch=6 locks=64 events=6
  shim tiles:     row_start=0 row_count=1 dma_ch=2 locks=16 events=4
MP-NPU clock:     MP-NPU Clock = 1267 MHz
H clock:          H Clock = 1800 MHz
Firmware:         1.1.2 build=65
Resource info:
  TOPS max/curr:  58 / 58
  Tasks max/curr: 16 / 0
  Clk max:        1800 kHz
Power mode:       0 (DEFAULT)
```

Sanity checks:
- 8 cols × 4 core tiles/col = **32 core tiles**, matches AIE-2P NPU5 spec.
- 58 TOPS at 1.267 GHz ⇒ ~45.8 ops/cycle/core — consistent with one
  256-bit INT8 MAC per cycle per core (32 tiles × 256/8 / 2 ≈ 50–60 TOPS).
- 16 task slots is the hwctx ceiling per process — plenty for a single
  inference engine (we expect to run 1–4 hwctxs concurrently).
- Power mode DEFAULT = firmware-driven DPM (good — manual TURBO is
  available for short bursts via SET_STATE if we want it).

The bypass path is structurally validated: 9-ioctl DRM ABI works without
any userspace dependency beyond `libc::ioctl`. Ready to move to Phase 1
(BO allocation + syncobj round-trip with a no-op CU).

## Open questions / follow-ups

- **MLIR-AIE state for AIE-2P specifically.** The compiler works for
  Phoenix (NPU1) and Strix Point (NPU4); AIE-2P / Strix Halo is the
  newest chip. Need to verify upstream MLIR-AIE has Peano backend
  emitting valid PDIs for `npu_7`.
- **dmabuf cross-import to amdgpu.** Standard DRM PRIME export should
  work, but Strix Halo iGPU's render node is a sibling DRM minor, not
  the same device. Confirm BO sharing works without extra mapping
  shims (likely yes — DRM_PRIME spec mandates this).
- **PASID + IOMMU.** Driver expects SVM with PASID for unified VA. We
  set `iommu=pt` in cmdline; need to verify the NPU sees CPU pages
  natively or whether userptr BOs are required.
- **Power gating.** AIE-2P idles at near-zero power; first dispatch will
  trigger DPM ramp. Need to characterize warm-up latency for short
  bursts (matters for spec-decode draft passes < 10 ms).

## Files to create

- `crates/xdna-compute/Cargo.toml`
- `crates/xdna-compute/src/lib.rs` — `XdnaError`, `Result`, module decls
- `crates/xdna-compute/src/ioctl.rs` — DRM_IOWR helpers + 9 ioctl numbers
- `crates/xdna-compute/src/device.rs` — `Device::open()` + GET_INFO probes
- `crates/xdna-compute/src/bin/probe.rs` — CLI that opens accel0 and
  dumps all GET_INFO fields. The Phase-0 exit binary.

Existing workspace `Cargo.toml` gets one new line in `members`.
