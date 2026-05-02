# 32c kernel bring-up progress (2026-05-02)

State of #45 after first authoring iteration. Branch tip: 249dda1.

## What works

  matmul_i8_512_32c builds clean, runs, and passes correctness.
  - Build: kernels/aie2p/matmul_i8_512_32c/build.sh
  - Bin:   crates/hipx/src/bin/matmul_i8_512_32c.rs (hipx-matmul-i8-512-32c)
  - PDI:   committed (150 KB)
  - Insts: committed (4.2 KB)
  - Topology: full AIE-2P NPU2 array (8 cols x 4 rows compute = 32 tiles)
  - Generator: mlir-aie whole_array_placed.py, npu2, m=64 k=64 n=32

## Measured throughput (hipx, gfx1151, AIE 1.1)

  Cold dispatch: 3729 us (lazy init)
  30-iter steady: mean 155 us, max 262 us
  Effective: 1.73 TOp/s (raw)
  Compute-only (subtract ~67 us dispatch floor): ~3.0 TOp/s

## Comparison anchors

  4c i8 1024^3: 4.55 TOp/s @ 472 us per dispatch (8x more compute)
  4c i8 2048^3: 4.60 TOp/s @ 3744 us per dispatch (64x more compute)
  iGPU INT8 prefill: 18-20 TOp/s
  NPU 32c THIS BUILD at 512^3: 1.73 TOp/s (well below 4c on a smaller shape)

## Why 32c at 512^3 underperforms 4c at 1024^3

Per-core utilization: at 512^3 with m=64, n=32 the math is
  tiles_per_core = (M/m) * (N/n) / n_aie_cores
                 = (512/64) * (512/32) / 32
                 = 8 * 16 / 32 = 4 tiles per core
With K/k = 8 inner iterations per tile, each core does 32 inner iters
total before barrier. Bandwidth bound, not compute bound.

The 32c topology also fundamentally differs from 4c in A-broadcast:
  n_shim_mem_A = n_aie_rows = 4   (when n_aie_cols > n_aie_rows)
So only 4 shim tiles feed A across 8 columns, halving per-column A
bandwidth vs the 4c topology which has 1 shim tile per column.

## Blocker for scaling up

M=K=N=1024 with m=64 k=64 n=32 hits an mlir-aie DMA descriptor size
limit (max 1023 per dim). Specifically a `<size = 1024, stride = 1024>`
descriptor is rejected with:
  'aie.dma_bd' op Size 1 exceeds the [0:1023] range.

Tried workarounds in this iter:
  - 768^3 with m=64 k=64 n=32: tile-group-divisibility error
  - 1024^3 with m=64 k=128 n=32: same DMA size 1024 error (descriptor
    pattern unchanged in K dim)

Likely paths forward:
  1. Try non-power-of-2 with valid divisibility (e.g. M=K=N=896 with
     m=32 k=64 n=28) - need to compute valid (M,K,N,m,k,n) candidates
  2. Pass --b-col-maj 1 to the generator - may produce different DMA
     patterns
  3. Shape the bench around chunked dispatches: call 512^3 kernel
     multiple times to cover a larger logical GEMM (adds N dispatch
     overhead per chunk)
  4. Author K-chunked MLIR by hand (most work, most control)

## Next iteration plan

Priority: find a working larger shape to confirm 32c can scale.

Try in order:
  1. Generate 1024^3 with --b-col-maj 1 - if it builds and runs,
     measure
  2. Generate non-power-of-2 (M=896 or 1280 etc) - search the
     valid-divisibility space
  3. If neither works in one iter, fall back to 512^3 chunks across
     a 1024^3 logical GEMM and measure end-to-end. The 32c-512^3
     kernel still beats nothing, but it would set a floor for the
     #46 prefill viability decision.

## Implications for #46 (the decision)

If 32c can hit ~25-30 TOp/s at production shape:
  Concurrent split lift = 25/19.5 = 1.28x or 30/19.5 = 1.54x
  At 1.4x threshold: marginal (1.54x is over, 1.28x under)

At currently-measured 32c throughput of 1.73 TOp/s:
  Concurrent split lift = 1.73/19.5 = 1.09x = only 9% lift
  Below threshold. Decision: NO if we can't scale 32c.

The decision hinges entirely on whether we can author or generate a
32c kernel that delivers >5 TOp/s at production shape (>= 4608 dims).
That's the gate of #46.
