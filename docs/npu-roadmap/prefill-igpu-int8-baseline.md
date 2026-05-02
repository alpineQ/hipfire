# iGPU INT8 GEMM baseline at production prefill shapes (gfx1151)

Captured 2026-05-02 via crates/engine/examples/bench_int8_gemm_prefill on
hipx (Strix Halo, gfx1151, ROCm 7.2). 100 iters per (shape, batch),
10-iter warmup, hipEvent timing.

Path: gemm_hfq4g256_mmq_set (Q8_1 activations x HFQ4 weights, i8 WMMA
accumulate). MMQ kernel uses 128x128 tiles; below batch=128 it underfills
tiles and is a poor proxy for throughput, but those rows are kept for the
launch-overhead floor signal.

## Raw measurements

  Shape                          batch  ms/op    TOp/s
  -----                          -----  -----    -----
  27b-attn-proj 4608x4608         32   0.275      4.95
  27b-attn-proj 4608x4608         64   0.297      9.17
  27b-attn-proj 4608x4608        128   0.267     20.38
  27b-attn-proj 4608x4608        256   0.520     20.89
  27b-attn-proj 4608x4608        512   1.027     21.18
  27b-ffn-down 4608x18432         32   1.219      4.46
  27b-ffn-down 4608x18432         64   1.307      8.32
  27b-ffn-down 4608x18432        128   1.232     17.64
  27b-ffn-down 4608x18432        256   2.471     17.60
  27b-ffn-down 4608x18432        512   5.006     17.37
  27b-ffn-gate+up 36864x4608      32   2.477      4.39
  27b-ffn-gate+up 36864x4608      64   2.676      8.12
  27b-ffn-gate+up 36864x4608     128   2.408     18.06
  27b-ffn-gate+up 36864x4608     256   4.492     19.36
  27b-ffn-gate+up 36864x4608     512   8.711     19.97

## Steady-state throughput

At production batches (>= 128):
  attn-proj:  20-21 TOp/s
  ffn-down:   17-18 TOp/s
  ffn-gate+up: 18-20 TOp/s

Aggregate iGPU INT8 GEMM ground floor: ~18-20 TOp/s at production prefill.

## Comparison anchors

  NPU INT8 GEMM 4c (matmul_i8_2048):  4.55 TOp/s
  NPU 32c (linear scaling projection): ~37 TOp/s [theoretical only]
  NPU 32c (realistic 6-7x scaling):   ~28-32 TOp/s [conservative]

  iGPU peak INT8 (gfx1151, theoretical): ~32 TOp/s
  iGPU measured at prefill (this doc):  ~18-20 TOp/s = ~60% of peak

## Implications for prefill viability (#46 decision)

Concurrent split lift = NPU_TOps / iGPU_TOps (additive, since UMA bandwidth
is shared but compute is independent). At various NPU scaling assumptions:

  NPU 4c stays at 4.55 TOp/s -> +23% potential lift (4.55/19.5)
  NPU 32c hits 28 TOp/s      -> +144% potential lift
  NPU 32c hits 37 TOp/s      -> +190% potential lift

Threshold gate is >=1.4x lift -> proceed. At 4c (current measured kernel),
we are well below threshold and #45 (32c kernel build) is mandatory before
any go/no-go decision. At realistic 32c throughput, lift is 1.4-1.9x even
with conservative scaling assumptions.

The decision narrows to: can we land a working 32c i8 GEMM kernel on
AIE-2P, and does its measured throughput exceed ~25 TOp/s (~5.5x of 4c)?

## Caveats

1. UMA bandwidth contention. Both iGPU and NPU read weights from the same
   physical memory. Weights for one matmul are 4608*36864*0.5 bytes (HFQ4)
   plus scale/zero metadata - around 80 MB. Concurrent reads may cause
   bandwidth contention not modeled in pure-compute TOp/s.

2. Per-dispatch overhead at small batches. NPU dispatch is ~67us; for
   pp32 the iGPU GEMM is ~275us (attn) to ~2.5ms (FFN), so NPU could
   absorb 1-2 ms of work per dispatch without overhead dominating.

3. The MMQ path on iGPU does not represent bf16 prefill throughput. If
   the production prefill stack uses bf16 (not Q8_1), throughput would
   be roughly half this number, making concurrent split more attractive.

## Pointers

  Bench source:  crates/engine/examples/bench_int8_gemm_prefill.rs
  Raw output:    bench/prefill-igpu-int8-20260502.txt
  NPU 4c data:   bench/npu-stage-1.5-scoping-2026-05-02.txt
                 bench/npu-multicore-scaling-2026-05-02.txt
  Speed-gate:    tests/speed-baselines/gfx1151.txt (end-to-end tok/s)
