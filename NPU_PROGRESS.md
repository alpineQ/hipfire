# NPU Roadmap Progress (2026-05-02)

## Summary

| Stage | State | Branch |
|-------|-------|--------|
| 1.1 Correctness verifier | ESCALATED (bf16 mul semantics) | npu-roadmap/2026-05-02 |
| 1.2 Embed PDI | blocked on 1.1 | feat/npu-stage-1.2-embed-pdi |
| 1.3 Engine wiring | blocked on 1.1 | feat/npu-stage-1.3-engine-wiring |
| 1.5 Bench | blocked on 1.3 | feat/npu-stage-1.5-bench |
| 2.6 Fused score | blocked on 1.5 | feat/npu-stage-2.6-fused-score |
| Cross-cutting: kernel build automation | active | npu-roadmap/2026-05-02 |

## Stage 1.1: in progress

Goal: bit-for-bit match between NPU `asym3_dequant_256` and a CPU reference for the same input. 100 random seeds clean before stage 1.2 starts.

Plan:
1. Locate the engine's authoritative asym3 dequant. Mirror codebook, bit packing, output layout.
2. Author CPU reference in Rust under `crates/hipx/examples/verify_asym3_dequant.rs`.
3. Author dispatcher in same binary, reuse the dmabuf scaffold.
4. First run will likely mismatch. Capture pattern.
5. Iterate fix > rebuild > rerun until 100 seeds clean.

## Action log (append-only, newest at bottom)

- 2026-05-02 init: branch `npu-roadmap/2026-05-02` created. NPU_PROGRESS.md committed. Beginning stage 1.1 step 1 (locate engine reference).
- 2026-05-02 stage 1.1 setup: located engine reference at `kernels/src/triattn_score_asym3.hip:54-64` and `kernels/src/turbo_common.h::TURBO_C3_256`. Identified codebook approximation bug in kernel before first run.
- 2026-05-02 stage 1.1 fix 1: kernel codebook converted to exact engine values. First verifier run: 256/256 mismatches all 5 seeds. Diagnosed: random per-seed values, kernel computing but wrong function.
- 2026-05-02 stage 1.1 fix 2: aie::select argument order corrected (was reversed; `select(v1, v2, m) = m == 0 ? v1 : v2`). Mismatches dropped to 121-256/256 with 1 ULP scatter.
- 2026-05-02 stage 1.1 diagnostic: ASYM3_DEBUG dump shows NPU output 1 ULP off from f32-mul-then-RNE bf16. Neither RNE nor RTZ matches.
- 2026-05-02 stage 1.1 fix 3: added calibration phase. Byte-identical codebook confirmed between kernel and CPU reference. Isolates remaining mismatches to `aie::mul` rounding chain.
- 2026-05-02 stage 1.1 ESCALATED: filed `MANUAL_REVIEW.md` ESCALATED-1. Pivoting to cross-cutting kernel build automation per contract three-strike rule.
