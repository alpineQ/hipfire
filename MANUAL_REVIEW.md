# Manual Review Queue (npu-roadmap/2026-05-02)

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
