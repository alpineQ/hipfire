# NPU Roadmap Progress (2026-05-02)

## Summary

| Stage | State | Branch |
|-------|-------|--------|
| 1.1 Correctness verifier | DONE (3-gate acceptance, AIE-2P-shape doc'd) | npu-roadmap/2026-05-02 |
| 1.2 Embed PDI | DONE (PDI+insts in kernels.rs; 100/100 verifier still PASS) | npu-roadmap/2026-05-02 |
| 1.3 Engine wiring | next up | feat/npu-stage-1.3-engine-wiring |
| 1.5 Bench | blocked on 1.3 | feat/npu-stage-1.5-bench |
| 2.6 Fused score | blocked on 1.5 | feat/npu-stage-2.6-fused-score |
| LUT-based bit-exact verifier (deferred) | future work | -- |

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
- 2026-05-02 stage 1.1 resumed (principal direction: AIE-2P-shape semantics, debug-friendly): authored `asym3_dequant_256_f32` diagnostic kernel (returns accfloat as f32 instead of bf16). Sweep across 14336 (cnorm bf16, cb_idx) pairs proved `aie::mul -> to_vector<float>` is bit-faithful (ratio = 1.0 across all pairs). Discrepancy isolated to `to_vector<bfloat16>` down-conversion.
- 2026-05-02 stage 1.1 manual case analysis: three bf16-EXACT cnorm cases tested. NPU bf16 outputs are -2 ULP, +1 ULP, and 0 ULP from RAZ predictions respectively. No single-mode rounding fits. The bf16 down-conversion has hardware-specific non-uniform behavior. Logged in `MANUAL_REVIEW.md` ESCALATED-1.
- 2026-05-02 stage 1.1 resolution path (proposed): change kernel to take bf16 cnorm directly + build empirical LUT of (cnorm_bf16, cb_idx) -> bf16_output via sweep. Bit-exact verifier by construction. Estimate: 1.5h work.
- 2026-05-02 stage 1.1 DONE (principal accepted ULP-bounded gate proposal): three gates landed in verifier (determinism, max ULP <= 2, |mean signed| <= 0.5). 100/100 seeds PASS on hipx. AIE-2P-shape characterization filed at docs/plans/aie2p-bf16-mul-shape.md. Strict bit-for-bit preserved as ASYM3_STRICT=1 toggle. LUT-based verifier deferred to future work.
- 2026-05-02 stage 1.2 starting: embed main.pdi + insts.bin via include_bytes! in crates/hipx/src/kernels.rs. Pattern matches existing matmul_i8 entries.
- 2026-05-02 stage 1.2 DONE: ASYM3_DEQUANT_256_{PDI,INSTS} added to crates/hipx/src/kernels.rs. Verifier defaults to embedded; ASYM3_PDI / ASYM3_INSTS env vars override with file paths for kernel-rebuild iteration. Committed binary artifacts (PDI 2784 B, insts 420 B). 100/100 seeds still PASS on hipx with embedded PDI. Next: stage 1.3 engine wiring under HIPFIRE_NPU_DEQUANT flag.
