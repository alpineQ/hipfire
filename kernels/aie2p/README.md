# AIE-2P kernels for hipx

This directory hosts kernel sources for the AMD AIE-2P NPU (Strix
Halo's `npu_7` silicon). Mirrors the `kernels/` HIP-source directory
that hipfire's `rdna-compute` builds on the iGPU side, but the source
language and toolchain differ.

## Toolchain (planned)

Per `findings/phase1-strategy.md` the chosen path is:

1. **Author kernels in IRON / MLIR-AIE.** Open-source compiler at
   github.com/Xilinx/mlir-aie (v1.3+), Vitis-free since AIE-2P is
   targeted by the Peano LLVM backend that's bundled.
2. **Compile offline** to PDI (Platform Device Image) bytes.
3. **Embed PDI bytes** in `crates/hipx/src/kernels.rs` via
   `pub const FOO_PDI: &[u8] = include_bytes!(...)` — same model as
   `crates/rdna-compute/src/kernels.rs` does for `.hsaco`.
4. **At runtime**, hipx loads the PDI bytes into a CMD BO and binds
   it to a hwctx as a CU via `DRM_AMDXDNA_CONFIG_HWCTX`.

## What lands here

- `passthrough.mlir` — Phase 1.3 syncobj round-trip target. A no-op
  CU that just signals completion. Smallest possible "valid CU" for
  proving the EXEC_CMD path.
- `gemv_int8.mlir` — Phase 1.5 first useful kernel. INT8 GEMV
  matching the shapes hipfire's MMQ kernels run.

## Build

(Not yet wired — Phase 1.3 will land a `compile-aie-kernels.sh`
analog of `compile-kernels.sh`.)

Manual for now:
```bash
cd ~/mlir-aie  # cloned separately
source utils/env_setup.sh
aiecc.py path/to/passthrough.mlir
# produces .pdi + ELF + xclbin in cwd
```

## Why AIE kernels aren't HIP

HIP / OpenCL / Vulkan compute kernels target a SIMT model where one
shader runs across many threads on a SIMD core. AIE tiles run a
**dataflow** model: each tile is an independent VLIW core (Peano
backend), tiles communicate via streaming buffers and shared memory,
and the kernel "compiles" into a graph of tile programs + DMA
descriptors. There's no SIMT to map HIP to. The MLIR-AIE dialects
(`aie`, `aie_device`, `aievec`) are the right abstraction, not HIP.
