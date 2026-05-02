# Prefill viability session summary (2026-05-02)

Branch: npu-roadmap/2026-05-02
Tip: 926cb0e

## Goal

After stage 1.5 (NPU dequant + iGPU score) closed last session as
80x slower than iGPU baseline, this session pivoted to test whether
NPU 32-core INT8 GEMM concurrent with iGPU prefill could lift
end-to-end prefill tok/s by >=1.4x via additive compute on UMA.

## What got done

Tasks #41 through #45 in two phases.

Phase 1 (parallel infrastructure scan): #41/#42/#43.
  #41 XDNA cmd-list ABI: ALREADY EXISTS. hipx submit_exec_cmd already
      wraps it. Lever 1 (command-list batching) is a real win that
      neither IRON nor hipx use today.
  #42 IRON dispatch path: also single-cmd-per-ioctl. Gap to hipx is
      ~5-10us (instruction BO pooling), not the 50us suspected.
  #43 iGPU INT8 GEMM at production prefill shapes: 18-20 TOp/s on
      gfx1151 via gemm_hfq4g256_mmq_set (Q8_1 act x HFQ4 weights x
      i8 WMMA accum). Three shapes (4608^2, FFN-down, FFN-gate+up),
      five batches each.

Phase 2 (kernel build): #45.
  Built three 32-core variants in increasing size. mlir-aie's
  whole_array_placed.py (one-time MLIR gen, then static; no Python
  in build path). All correct (Knuth-hash pseudorandom inputs across
  full M*N output, per codex-driven verifier hardening):

    matmul_i8_512_32c   1.73 TOp/s (dispatch-amortization-bound)
    matmul_i8_1024_32c  4.74 TOp/s (best; b-col-maj dodges DMA limit)
    matmul_i8_2048_32c  ~3.0 TOp/s cold (bandwidth-saturated)

## Key measurements

  iGPU INT8 prefill:                       18-20 TOp/s
  4c i8 1024^3 (single-row baseline):       4.55 TOp/s
  32c i8 1024^3 (full array, best shape):   4.74 TOp/s   (only +4%)
  Concurrent split lift at best 32c:        1.24x        (gate: 1.4x)

## Decision

Prefill viability via concurrent NPU 32c GEMM closes NO-LIFT.

Root cause is verified, not projected: NPU2's whole_array generator
sets n_shim_mem_A = n_aie_rows = 4 when n_aie_cols = 8. Only 4 shim
tiles feed A across 8 compute columns. Per-column A bandwidth is
half the 4c topology. Adding more cores doesn't add throughput.

Default HIPFIRE_NPU_DEQUANT stays 0. No engine integration of
prefill NPU offload.

Documented as ESCALATED-6 in MANUAL_REVIEW.md.

## What stays in the tree as reusable infrastructure

  3 32c kernels (512^3, 1024^3, 2048^3) and their dispatch bins
    correct, Knuth-hash-verified
    can be useful for any future NPU work that fits the constraint
  The XDNA abi findings + IRON dispatch analysis
    answer the "should we copy IRON" question definitively (no)
    Lever 1 batching ABI scope clear
  iGPU INT8 GEMM bench
    bench_int8_gemm_prefill is a clean reference for any future
    iGPU compute-throughput work

## Remaining options

  Task #28: HXQ / XQ4 split iGPU+NPU concurrent quant (deferred,
            different concurrency angle, may have different bandwidth
            profile than INT8 GEMM)
  Task #47: Lever 1 command-list batching impl (4-8h, defensive infra
            for any future NPU work; no near-term consumer)
  Pivot:    The npu-roadmap branch may be at a natural stopping point.
            Triattn decode: closed (ESCALATED-3/4/5).
            Prefill INT8 GEMM: closed (ESCALATED-6).
            HXQ/XQ4 (#28) is the only remaining NPU-concurrency angle
            and was already deferred earlier. Without a clear positive
            wedge, the right move may be merge-to-master + close branch.

## Pointers (for future you)

  MANUAL_REVIEW.md ESCALATED-6                 (closure narrative)
  findings/prefill-xdna-abi-scan.md             (Lever 1 ABI verified)
  findings/prefill-iron-dispatch-analysis.md    (XRT vs hipx gap small)
  findings/prefill-igpu-int8-baseline.md        (iGPU 18-20 TOp/s)
  findings/prefill-32c-bringup-progress.md      (mid-investigation state)
  bench/prefill-igpu-int8-20260502.txt          (raw iGPU numbers)
  kernels/aie2p/matmul_i8_512_32c/              (kernel + build)
  kernels/aie2p/matmul_i8_1024_32c/             (kernel + build)
  kernels/aie2p/matmul_i8_2048_32c/             (kernel + build)
  crates/hipx/src/bin/matmul_i8_*_32c.rs        (dispatch + verify)
