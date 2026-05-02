# NPU Roadmap Progress (2026-05-02)

## Summary

| Stage | State | Branch |
|-------|-------|--------|
| 1.1 Correctness verifier | in progress | feat/npu-stage-1.1-asym3-verifier (not yet) |
| 1.2 Embed PDI | pending | feat/npu-stage-1.2-embed-pdi |
| 1.3 Engine wiring | pending | feat/npu-stage-1.3-engine-wiring |
| 1.5 Bench | pending | feat/npu-stage-1.5-bench |
| 2.6 Fused score | pending | feat/npu-stage-2.6-fused-score |

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
