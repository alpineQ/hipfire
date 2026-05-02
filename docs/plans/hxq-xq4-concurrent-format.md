# HXQ / XQ4 — concurrent iGPU+NPU split quant (deferred design note)

**Status**: parked. Decision to prototype the concurrent-dispatch path first
with existing quants (INT8/BF16 for NPU side, MQ4/HFQ4g256 for iGPU side
unchanged). Revisit this format design after the dispatch path is validated.

## What this is

A weight quantization format where the bits of each weight are *split between
two physical engines* on the same Strix Halo SoC, so the iGPU and NPU each
consume a complementary half of the weight tensor and produce partial output
tensors that sum on UMA.

Unlike all current hipfire quants — which target one engine — HXQ would be
the first format that **requires** dual-engine UMA to execute.

The press story: "hipfire ships a quant that the AMD Ryzen AI MAX can run
3.5× faster than any single-engine implementation, because the bits are
literally inseparable from the dual-dispatch architecture."

## Sketch (not committed shape)

For each weight `w` (typically fp16 or bf16 in source), produce:

```
w_hi  := top-k bits, packed to a hipfire MQ-like 4-bit format
w_lo  := residual = w - dequant(w_hi), packed to INT8 with per-tile scale
```

iGPU GEMM consumes `w_hi` via the existing `gemm_hfq4g256_*` family
(unchanged kernel). NPU GEMM consumes `w_lo` via a new
`matmul_i8_residual` kernel that takes `(activation, w_lo, partial_in)`
and produces `partial_in + activation @ w_lo`. The two partials sum on
UMA via a tiny `add_inplace` kernel on whichever engine finishes second.

Bit budget (rough):
- iGPU half: 4 bits/weight (HFQ4g256-shaped). Carries the dominant
  signal; alone produces ~95 % of correct output.
- NPU half: 8 bits/weight as INT8 residual with per-tile fp16 scale.
  Carries the precision-recovery delta. Effective only when the
  iGPU half's quantization error is non-trivial (i.e., not after the
  FWHT-rotation absorbs it).

Total memory: 12 bits/weight effective (vs MQ4 at 4 bits) — **3× the
weight size**. That's the hard cost. UMA's 96 GB pool fits a 27B at
12 bits/weight but ~no headroom for KV/activations beyond ~8 K context.

## Why we DON'T build it yet

1. **The dispatch infrastructure isn't proven on a real workload yet.**
   `hipfire_x_overlap_rigor` shows 43 % wall-clock saved for a synthetic
   pair (1024^3 NPU + 2048^3 rocBLAS GEMM). Production iGPU per-op time
   (per-layer DFlash-batched at 27B-3.5 = 260 µs/layer post-rebase) is
   *smaller* than the NPU's compute window. Until we ship the asym3
   codec or an INT8 weight-offload variant and *measure* the actual
   tok/s lift on a real model, we don't know what `(t_iGPU, t_NPU)`
   look like in production.
2. **We don't yet know the right bit split.** 4/8 is a guess. The right
   split depends on how badly the NPU's residual contributes vs how
   much memory it costs. That's an empirical sweep, gated on the
   measurement infrastructure of (1).
3. **Two new kernels per layer, both production-quality.** iGPU side
   stays unchanged but NPU residual GEMM + cross-engine sum kernel are
   real engineering. ~3-6 months at current pace.
4. **Numerical validation is a separate full project.** A novel quant
   has to pass the coherence-gate-dflash 3-tier attractor checks,
   which historically have required multi-week iteration on existing
   formats (MQ3 / asym3) before they shipped clean.
5. **Press-worthy without it.** The dual-dispatch demo (43 % saved on
   a synthetic pair, clean inter-trial CV) is already novel as a story.
   The single-engine market doesn't have an equivalent. HXQ would be
   the *next* press story.

## What needs to happen first (gating checklist)

1. ✅ Pure-ioctl NPU dispatch via Rust (hipx).
2. ✅ Engine API (NpuRuntime + route() + opportunistic dispatch).
3. ✅ Working iGPU + NPU concurrent dispatch demo (`hipfire_x_overlap_rigor`,
   43 % saved, σ < 0.4 %).
4. ⏳ Production NPU dispatch on a real engine op:
   - asym3 codec K-prefetch on NPU (spec'd in `asym3-codec-budget.md`),
     OR an INT8 layer offload for spec-decode draft.
5. ⏳ Measured tok/s lift vs current iGPU-only baseline on 27B model.
6. ⏳ Sweep iGPU/NPU compute-time ratio across model sizes/contexts.
7. THEN: spec HXQ format with empirically-grounded bit split,
   author it, validate with coherence-gate.

## When to revisit

Trigger conditions:
- (4) and (5) above ship measurable iGPU+NPU concurrent throughput on
  a real production op.
- The measured per-op `(t_iGPU, t_NPU)` distribution gives a clear
  bit-split heuristic.
- The 3× weight memory cost is acceptable given current model sizes
  (i.e., we want larger models on Strix Halo, not the same models
  faster). Strix Halo's 96 GB UMA could host a 70B at HXQ's 12 bpw,
  which is a market the iGPU alone can't reach at MQ4.

## Naming

- **HXQ**: Hipfire-X Quant. Most literal.
- **XQ4**: 4-bit Cross-engine Quant. Reads more like a precision tier
  among MQ3/MQ4/MQ6. Pushes the "this is a 4-bit format that uses
  both engines" angle to first-glance readers.

Pick later. Doesn't matter pre-prototype.

## Pointers

- Concurrent dispatch demo (sets the empirical floor we'll improve on):
  `crates/engine/examples/hipfire_x_overlap_rigor.rs`
- NPU INT8 GEMM kernel that would carry the residual half:
  `crates/hipx/src/bin/matmul_i8_1024.rs` (4.46 TOp/s standalone /
  2.23 TOp/s engine API zero-copy)
- iGPU MQ4 GEMM that the high half would slot into:
  `kernels/src/gemm_*_hfq4g256_*.hip`
- Quant authoring playbook (groove for landing a new format cleanly):
  `.skills/hipfire-kernel-tuning/playbook.md`,
  `docs/methodology/perf-benchmarking.md`,
  the DFlash coherence gate (`scripts/coherence-gate-dflash.sh`).
