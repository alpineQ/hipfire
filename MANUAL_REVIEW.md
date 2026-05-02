# Manual Review Queue (npu-roadmap/2026-05-02)

## ESCALATED-5: Stage 1.5 measured A/B (final, 2026-05-02)

The earlier escalations (ESCALATED-3 + ESCALATED-4) were
projection-based. This is the actual measurement on hipx (Strix
Halo, gfx1151 iGPU + AIE-2P NPU2):

  bench/stage-1.5-ab-20260502-204713.txt
  Workload: seq_len=128, 8 kv_heads x 128 pos = 1024 chunks/dispatch
  100 trials, release build, warm cache

  Path A (iGPU baseline, triattn_score_asym3 inline):
    median:  17.03 us
    p95:     17.64 us

  Path B (NPU asym3_dequant_layer + iGPU triattn_score_bf16):
    median:  1371.95 us  (80x slower than Path A)
    p95:     1632.45 us
    breakdown:
      NPU asym3_dequant_layer:   1241 us  (90.5% of total)
      bf16 K upload host->iGPU:    97 us  ( 7.1% of total)
      iGPU triattn_score_bf16:     25 us  ( 1.8% of total)

The iGPU baseline at 17 us is dramatically faster than projection
math suggested. The iGPU's fused dequant + Givens + RoPE + score
is a single-pass GEMV-shape kernel that's well-tuned for gfx1151.

Even hypothetical 16-core SIMD-optimized NPU dequant
(1241 us / 16 = 78 us) + upload (97 us) + bf16 score (25 us) =
200 us total Path B vs 17 us Path A. Still 12x slower.

**Decision**: Stage 1.5 closes definitively as no-lift, with
measured numbers. Default `HIPFIRE_NPU_DEQUANT` stays 0. The
NPU dequant offload is fundamentally not competitive even with
the most optimistic optimization stack assumptions.

Why the projection was off:
  - Plan-doc per-iter estimate was 322 cycles. Reality (single
    core, scalar, packed input layout) is ~1240 us / 1024 iters
    = 1.2 us per iter = ~1900 cycles. 6x higher than estimate.
  - iGPU baseline assumed 67 ms/token from speed-baselines.
    Per-token triattn cost (per-layer × 46 layers) is much
    smaller than the full-token decode, ~17 us × 46 = 0.78 ms.
    iGPU is far better at this than the baseline number suggested.

Committed bench harness at
`crates/engine/examples/hipfire_x_stage_1_5_ab.rs`; rerun any time
to re-validate.



**Stage**: 2.6 (post correctness shipping)

**Status**: Stage 2.6 score kernel is mathematically correct and
end-to-end verified vs both a matching reference (NR sqrt + trig-
eliminated reformulation) and an independent iGPU-shape reference
(libm sqrt + atan2 + cos directly). Reformulation via cos(A-B) is
proven valid: max rel error 3.36e-6 across 6400 element comparisons.

But the perf gap is too wide to close with reasonable effort:

  Measured (single-core scalar, N_ITERS=128):
    per-iter           = 210 us / (head, pos)
    per-dispatch       = 26.84 ms

  Projection at hypothetical 16-core SIMD (16x speedup):
    per-iter           = ~13 us
    per-dispatch (full layer N=32768) = ~27 ms
    per-token (46 layers) = 1.24 s
    vs iGPU 67 ms       = 18x slower

Even stacking all four optimizations (SIMD + multi-core + per-layer
broadcast + multi-shape) projects to single-digit-x slower than
iGPU baseline at full production shape, not faster.

The 14% lift projection in `docs/plans/asym3-fused-score-plan.md`
assumed ~322 cycles per (head, pos). Measured is closer to 330,000
cycles per iter (1000x higher than estimate). Most of that is NOT
transcendentals - which we already eliminated via reformulation -
but the 128-band Givens scalar loop, ObjectFifo per-iter overhead,
and memory access latency. The ~322 cycle estimate counted only
arithmetic ops without memory or per-iter dispatch overhead.

**Decision** (per contract escalation rule):

  1. Default `HIPFIRE_NPU_DEQUANT` stays 0.
  2. Stage 2.6 closes as ESCALATED.
  3. The npu-roadmap reaches a conclusion: AIE-2P NPU2 is
     fundamentally not the right tool for this triattn workload
     given the iGPU's attention path. The chain dequant -> score
     stays on iGPU.
  4. The shipped kernels (asym3_dequant_layer, asym3_score_one,
     asym3_score_layer, all multi-core variants) remain in the
     repo as reusable scaffolding for future workloads where the
     compute density better matches AIE-2P's strengths (e.g.
     INT8 GEMM, FFT, unstructured matmul).

**Telemetry**:
  - bench/npu-multicore-scaling-2026-05-02.txt
  - bench/npu-stage-1.5-scoping-2026-05-02.txt
  - bench/stage-2.6-perf-status-2026-05-02.txt
  - 24 commits across 21 /loop iterations on npu-roadmap/2026-05-02

**Tasks**: #38 (Stage 2.6) closes as ESCALATED. #20 (Phase 4 LLM
integration via NPU) remains in_progress as the umbrella task; can
be marked completed once this MANUAL_REVIEW entry is acknowledged.

**Future work** (out of npu-roadmap scope):
  - Strix Halo NPU INT8 GEMM for matmul offload (different workload)
  - HXQ/XQ4 split iGPU+NPU concurrent quant (#28)
  - Investigate AIE-2P for prefill rather than decode (different
    arithmetic intensity profile)



## ESCALATED-3: Stage 1.5 NPU dequant offload does not produce a lift

**Stage**: 1.5 (post multi-core scaffolding)

**Status**: Multi-core scaffolding (1c/2c/4c/8c kernels) shipped
and verified. Measured per-iter compute on Strix Halo AIE-2P is
1.74 us at N_ITERS=1024 single-core. Linear extrapolation to
production shape (N_ITERS=32768) projects:

| N_cores | Per-dispatch | Per-token (46 layers) | vs iGPU 67 ms |
|---------|--------------|------------------------|---------------|
|    8    |  7.84 ms     |  360 ms                | 5.4x slower   |
|   16    |  4.30 ms     |  198 ms                | 3.0x slower   |
|   32    |  2.51 ms     |  115 ms                | 1.7x slower   |

Even fully-utilized 32-core configuration projects 1.7x slower
than iGPU baseline. The NPU dequant offload alone does not beat
the iGPU's existing fused dequant + score path.

**Decision** (per contract escalation rule "If lift is < 5% or
quality drifts: do not flip default. Escalate to MANUAL_REVIEW.md
with telemetry; stage 2.6 (fused score) may be the right next
move instead."):

  1. Default `HIPFIRE_NPU_DEQUANT` stays 0.
  2. Stage 1.5 closes as ESCALATED with this telemetry.
  3. Stage 2.6 (fused asym3 score on NPU) is the actual perf
     lever. Eliminates the 16 MiB bf16 K writeback per layer +
     moves Givens + RoPE + atan2/cos/sin/sqrt off iGPU.
  4. Multi-core scaffolding from 1.4a is reusable for 2.6.

**Telemetry**: `bench/npu-multicore-scaling-2026-05-02.txt`

**Tasks**: #37 (Stage 1.5) closed as resolved-via-escalation.
#40 (multi-core) closed; scaffolding shipped, perf path moves to
#38 (Stage 2.6).



## RESOLVED-2: Tighten verifier bounds from (4, 1.0) to (3, 0.5)

Resolved 2026-05-02. The 1000-seed layer verifier shows max ULP 2 /
mean +0.0088, the shadow harness shows max ULP 2 / mean -0.02
(per-layer max bias +0.24). Bounds tightened to (3, 0.5) in:
  - crates/hipx/src/bin/verify_asym3_dequant.rs::main
  - crates/hipx/src/bin/verify_asym3_dequant_layer.rs::main
  - crates/engine/examples/hipfire_x_asym3_shadow.rs env defaults
  - docs/plans/aie2p-bf16-mul-shape.md envelope description

Codex caught that an initial tighten to (2, 0.5) had zero
headroom over the empirical max and would flake on a single
rogue measurement. Final landing bound = 3 gives 1 ULP of
headroom while still catching structural bugs (which produce
>>2 ULP errors).

Original record below for context (the recommendation field's
"1 ULP headroom" math was off by 1: bound 2 == empirical max 2,
zero headroom; bound 3 gives the actual 1 ULP):

**Stage**: 1.1 (post codex review fixes, commit 5adfd07)

**Status**: After fixing the codebook RTZ-vs-RNE bias the codex
review caught (P2 #3), the empirical envelope dropped from
(max=3, mean=+0.70) to (max=2, mean=-0.06). The 1 ULP and 0.5 ULP
of headroom in the current bounds (4, 1.0) is now far more
permissive than necessary. Real bug classes still produce >>2 ULP
errors, but the gap to "actually catch new sources of drift" is
wider than ideal.

**Recommendation**: tighten to (max_ulp=2, mean_bias=0.5). This is
1 ULP of headroom over the empirical max and ~10x of headroom
over the empirical mean, which still passes today but flags any
future regression that pushes max_ulp to 3 or mean past 0.5
(both of which would suggest a new bias source worth
investigating).

**Why deferred**: scoping pivot - stage 1.4 (production layer
kernel) is the next priority, and tightening bounds is a
zero-risk infrastructure improvement that can land alongside or
after 1.4 once we know the layer-batched kernel has the same
ULP envelope.

**Action**: 1-line change in
`crates/hipx/src/bin/verify_asym3_dequant.rs::main` (the
`max_ulp_bound` and `mean_bias_bound` initializers), plus
matching update in
`crates/engine/examples/hipfire_x_asym3_shadow.rs` env defaults,
plus matching update in
`docs/plans/aie2p-bf16-mul-shape.md` envelope description.



## ESCALATED-1: AIE-2P bf16 mul rounding semantics

**Stage**: 1.1 (correctness verifier)

**Status**: Two real bugs found and fixed via the verifier (codebook precision, `aie::select` arg order). Mismatches dropped from 256/256 to ~120-220 of 256. The remaining mismatches are 1 ULP scattered. The kernel is functionally correct in the asym3 sense; the open question is whether the CPU reference can be made bit-for-bit equivalent to AIE-2P bf16 multiplication, or whether stage 1.1 should accept a 1-ULP tolerance.

**Evidence** (with `ASYM3_DEBUG=1`):

Three sample failures from the run on commit `f87211f`:

```
seed 1 dim 0 (tid 0 i 0) idx=5 cb=0x3d3e(0.0463867) cnorm=0xbff5(-1.9140625)
  f32_prod=-0.0887870789  via_rne=0xbdb6  via_trunc=0xbdb5
  cpu=0xbdb6(-0.0888672)  npu=0xbdb7(-0.0893555)

seed 2 dim 3 (tid 0 i 3) idx=6 cb=0x3dab(0.0834961) cnorm=0xbf70(-0.9375000)
  f32_prod=-0.0782775879  via_rne=0xbda0  via_trunc=0xbda0
  cpu=0xbda0(-0.0781250)  npu=0xbda1(-0.0786133)

seed 3 dim 0 (tid 0 i 0) idx=7 cb=0x3e0a(0.1347656) cnorm=0x3ec8(0.3906250)
  f32_prod=0.0526428223  via_rne=0x3d58  via_trunc=0x3d57
  cpu=0x3d58(0.0527344)  npu=0x3d56(0.0522461)
```

The CPU reference computes `bf16(cnorm) -> f32 -> mul f32 -> RNE bf16`. Both RNE and truncate-low-16 yield bf16 values that NPU does not match. The NPU result back-solves to suggest the kernel's effective codebook entry is 1 ULP below my expected value (e.g., kernel codebook[7] behaves as `0x3E09` not `0x3E0A`).

**Hypotheses** (none confirmed):

1. AIE-API `bfloat16(float)` constructor uses non-RNE rounding for the codebook constants. The dropped low 16 bits can produce a bf16 value 1 ULP off from a correctly RNE'd result.
2. The `::mul_elem_32(bf16, bf16)` intrinsic uses bf16-mantissa internal precision (not f32 internal), and its rounding to the `accfloat` accumulator differs from "promote, mul in fp32, round to bf16."
3. The `to_vector<bfloat16>()` accumulator-to-bf16 conversion uses round-toward-zero or some other non-RNE mode.

**Approaches tried**:

- A: Use exact engine codebook constants (`-0.134860f` etc) — fixed a real bug; mismatches went from 256/256 wrong to scattered.
- B: Fix `aie::select` argument order (was reversed; `select(v1, v2, m) = m == 0 ? v1 : v2`, we wanted mask-set to take `code_e`) — major fix; mismatches halved.
- C (in progress, escalating): Model AIE-2P bf16 mul exactly. Several rounding models tried in the CPU reference (RNE, RTZ); none match the NPU output.

**Open questions for the principal**:

1. Is 1-ULP-scatter acceptable for stage 1.1? The kernel is correct in the asym3 quant sense; the rounding LSB is a property of AIE-2P bf16 hardware. Many published AIE kernels accept similar tolerance.
2. If bit-for-bit is mandatory, should the next attempt be:
   a. Run a calibration phase: cnorm=1.0 with packed encoding all-idx=k for k=0..7; observe the NPU's actual stored codebook bf16 representations. Use those as the CPU reference's codebook.
   b. Build a "kernel trace" reference: dump intermediate accfloat values via debug instrumentation in the kernel and reconstruct semantics from those.
   c. Search the AIE-API source / AMD docs for explicit `mul_elem_*` rounding spec.

**Branch**: `npu-roadmap/2026-05-02`, tip `f87211f` carries the diagnostic instrumentation. Toggle with `ASYM3_DEBUG=1` env. Run with `./target/debug/verify_asym3_dequant <N_SEEDS>`.

**Recommendation**: option (a) calibration. It gives bit-for-bit determinism without depending on AIE-API undocumented internals. The downside is the verifier becomes "kernel's codebook == measured codebook" rather than "kernel's codebook == engine's codebook," so we'd want a separate sanity check that the measured codebook is within 1 ULP of the engine values.

**Time spent**: ~45 minutes from the first failed run to escalation. Two genuine bugs caught.

---

### Update after calibration (commit `d6f726b`)

Calibrated each of the 8 codebook entries by sending all-idx=k inputs with cnorm=1.0 (which is exactly representable in bf16). The kernel's stored bf16 codebook is **byte-for-byte identical** to my CPU reference codebook for every entry:

```
[calibrate] codebook[0] kernel=0xbe0a(-0.1347656) ref=0xbe0a(-0.1348600) ==
[calibrate] codebook[1] kernel=0xbdab(-0.0834961) ref=0xbdab(-0.0833200) ==
[calibrate] codebook[2] kernel=0xbd3e(-0.0463867) ref=0xbd3e(-0.0464690) ==
[calibrate] codebook[3] kernel=0xbc79(-0.0151978) ref=0xbc79(-0.0151760) ==
[calibrate] codebook[4] kernel=0x3c79(0.0151978) ref=0x3c79(0.0151760) ==
[calibrate] codebook[5] kernel=0x3d3e(0.0463867) ref=0x3d3e(0.0464690) ==
[calibrate] codebook[6] kernel=0x3dab(0.0834961) ref=0x3dab(0.0833200) ==
[calibrate] codebook[7] kernel=0x3e0a(0.1347656) ref=0x3e0a(0.1348600) ==
```

After running with the calibrated codebook in the CPU reference, mismatches are unchanged (159/256, 121/256, etc., same first-diff dims). **This isolates the bug to the `aie::mul(bf16, bf16) -> accfloat -> bf16` rounding chain, NOT codebook conversion.**

Implication for the open questions above:

- Hypothesis 1 (`bfloat16(float)` ctor non-RNE) is FALSE: the codebook bytes match exactly under standard RNE.
- Hypothesis 2 or 3 (mul or accumulator-to-bf16 rounding) is the active explanation.
- Option (a) calibration of the codebook does not unblock bit-for-bit. Calibration of the mul itself is intractable (combinatorial input space).

**Refined recommendation**: relax stage 1.1 acceptance to "100 random seeds match the CPU reference up to 1 bf16 ULP per element." Document this as the achievable correctness floor for AIE-2P bf16 mul. The kernel IS deterministic and CORRECT in the asym3 sense; the mul's last-bit behavior is hardware. If the principal disagrees, the next attempt would be to dump intermediate accfloat values via a custom kernel that returns the accumulator pre-conversion, comparing those against my f32 product. That would tell us whether the divergence is in the mul or in the to_vector conversion.

---

### Update after f32 diagnostic kernel + sweep + manual case analysis (commit df36d8a)

Built `kernels/aie2p/asym3_dequant_256_f32` that returns the
`aie::mul` accumulator as fp32 instead of rounding to bf16. Swept 14336
(cnorm bf16 x cb_idx) pairs through it. Result: ratio between CPU
fp32-faithful product and NPU f32 acc = **1.0000000000 across all
14336 pairs**. The mul is bit-faithful when output is fp32.

This isolates the discrepancy to the bf16 down-conversion path:
`accum.to_vector<bfloat16>()`. Tested three cases manually with bf16-
exact cnorm (eliminating cnorm conversion as a variable):

```
Case A: cnorm 0x3EC8 (0.39), cb 0x3E0A. f32 product = 0x3D57A000
        RAZ predicts 0x3D58; NPU produces 0x3D56  (2 ULPs LESS magnitude)

Case B: cnorm 0xBFF5 (-1.91), cb 0x3D3E. f32 product = 0xBDB5D600
        RAZ predicts 0xBDB6; NPU produces 0xBDB7  (1 ULP MORE magnitude)

Case C: cnorm 0xBF70 (-0.94), cb 0x3DAB. f32 product = 0xBDA05000
        RAZ predicts 0xBDA1; NPU produces 0xBDA1  (matches RAZ)
```

**Mixed directions**. No single-mode rounding (RNE / RTZ / RAZ /
RAZ+1mag / etc.) fits all three. The bf16 down-conversion's
behavior depends on bits beyond what a simple rounding rule captures.
Hypotheses:

1. AIE-2P's `accfloat -> bfloat16` is a hardware instruction with
   non-IEEE rounding (saturating, or magnitude-aware bias). Not
   documented publicly.
2. `accum::to_vector<bfloat16>` for the bf16 path uses a different
   precision than `to_vector<float>` reports. The "f32 acc" we
   observe via `to_vector<float>` may itself be a rounded view of a
   wider internal accumulator, while `to_vector<bfloat16>` uses the
   full accumulator. (Plausible but the sweep showed `to_vector<float>`
   is bit-faithful w.r.t. the CPU fp32 product, suggesting accfloat
   IS f32-precision-equivalent on bf16 inputs. Unclear.)

Reverse-engineering this rigorously is open-ended.

**Tractable bit-exact path**: ditch closed-form modeling. Build a
LUT-based reference. Sweep all 65536 cnorm bf16 values x 8 codebook
indices through the bf16 kernel directly. ~524k dispatches at
~250 us each = ~131 s for the full sweep. Resulting LUT is 1 MB
(524k * 2 bytes). For verification, look up `(cnorm_bf16, cb_idx)`
in the LUT. Bit-exact by construction.

For the f32 -> bf16 cnorm conversion question, we need either:
(a) characterize the kernel's `(bfloat16)(*float_ptr)` rounding
    via a separate calibration sweep across f32 inputs of varying
    low-bit patterns, OR
(b) change the kernel's interface to take bf16 cnorm directly,
    move the f32 -> bf16 conversion to host-side IEEE RNE.
    Production engine integration would do this anyway since
    cnorm is computed once per layer per token, so paying the
    host-side conversion cost is fine.

Option (b) is structurally cleaner: kernel only handles bf16
operations, all f32 -> bf16 conversion happens host-side under
known IEEE rules. CPU reference becomes deterministic for any
input.

**Recommendation post-refinement**: option (b). Modify the kernel
to take bf16 cnorm. Build the LUT for bf16 inputs. Use the LUT in
the verifier. Bit-exact stage 1.1 by construction.

Estimated work: kernel mod + rebuild (~30 min), LUT generation
sweep (~3 min runtime + plumbing 30 min), verifier rewrite to
LUT-based (~30 min). Total ~1.5h for clean stage 1.1 win.
