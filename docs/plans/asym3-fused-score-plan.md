# Stage 2.6: asym3 fused score on NPU

## Why now

Stage 1.5 escalation (commit bfec23d) showed that even maximum
core count (32) for dequant-only NPU offload projects 1.7x slower
than iGPU baseline at full layer shape. The actual perf lever per
the contract is fused score: keep K on-NPU through scoring,
eliminating the 16 MiB bf16 K writeback per layer + offloading
Givens / RoPE / atan2 / cos / sin / sqrt math from iGPU.

## Architecture

Per iGPU `triattn_score_asym3.hip`, each (head, position) computes:

  1. Dequant: 256 packed 3-bit -> 256 f32 (cnorm * codebook[idx])
  2. Inverse Givens for 4 band pairs: (a, b) <- rotation(-theta)(a', b')
  3. Per-band: k_mag = sqrt(re*re + im*im), k_phase = atan2(im, re)
  4. RoPE: omega = exp(-2f/n_rot * log(rope_theta)),
           angle = omega*p_q + c_phase - k_phase
  5. s_trig += c_mag * k_mag * cos(angle)
  6. s_norm += (1 - min(c_mag/c_abs, 1)) * c_abs * k_mag
  7. Reduce 4 bands -> 1 score per (head, pos)
  8. Write 1 f32 score

Output collapses from 16 MiB (bf16 K) to 128 KB (1 f32 per
(head, pos) at full shape). 128x less BW than dequant-only.

## AIE-2P transcendental support

Peano libm.a provides sinf, cosf, atan2f, sqrtf, expf, logf for
aie2p target. These are SCALAR implementations - AIE-API does
not vectorize transcendentals. Per-call latency is unknown without
measurement, but assume 50-200 cycles/op based on typical RISC
libm (aie2p core is roughly RISC-class for scalar).

Per (head, position):
  - 4 sqrt calls (k_mag per band)
  - 4 atan2 calls (k_phase per band)
  - 4 exp + 4 log calls (RoPE omega per band; *can be precomputed*)
  - 4 cos calls (angle per band)
  - 1 sqrt for c_mag (*can be precomputed in centers*)
  - 1 atan2 for c_phase (*can be precomputed in centers*)

Trig calls per (head, position) AFTER precomputation:
  - 4 sqrt + 4 atan2 + 4 cos = 12 transcendentals

Per-layer (32768 iters): 32768 * 12 = 393216 transcendentals.
At ~100 cycles each, single-core: 39.3M cycles = 24.5 ms.
At 32 cores: 24.5 / 32 = 0.77 ms compute.

This is genuinely promising:
  - 32-core fused score: 0.77 ms compute + 0.5 ms host overhead
    = 1.27 ms per dispatch
  - Per-token (46 layers): 1.27 * 46 = 58 ms
  - vs iGPU baseline: 67 ms (1.16x faster, 14% lift)

That assumes ~100 cycle/transcendental, which is optimistic for
aie2p scalar libm. If actual is 200 cycles, projection becomes
~115 ms/token (1.7x slower than iGPU). The actual number could
land anywhere in this range.

## Key reformulation: eliminate per-iter atan2 + cos

The iGPU kernel computes:
  k_phase = atan2(k_im, k_re)
  angle = omega*p_q + c_phase - k_phase
  s_trig += c_mag * k_mag * cos(angle)

Apply cos(A-B) = cos(A)cos(B) + sin(A)sin(B):
  cos(angle) = cos(omega*p_q + c_phase) * cos(k_phase)
             + sin(omega*p_q + c_phase) * sin(k_phase)

And cos(k_phase) = k_re / k_mag, sin(k_phase) = k_im / k_mag.
So:
  k_mag * cos(angle) = cos(omega*p_q + c_phase) * k_re
                     + sin(omega*p_q + c_phase) * k_im

The k_mag and k_phase BOTH disappear from the s_trig term:
  s_trig += c_mag * (cos_a[f] * k_re + sin_a[f] * k_im)

where cos_a[f] = cos(omega[f]*p_q + c_phase[f]) and similarly
sin_a[f] depend ONLY on (f, p_q), not on (head, pos). Per
dispatch (one decode step) p_q is constant, so cos_a and sin_a
are computed ONCE on the host (128 trig pairs total), then
shipped to the NPU as a 2 * 128 = 256 f32 input vector.

The per-(head, pos) compute on NPU becomes:
  - Dequant 256 indices to v[256]                     ~128 cycles SIMD
  - Givens: 768 scalar ops -> 48 SIMD-16 ops          ~48 cycles
  - k_mag = sqrt(k_re*k_re + k_im*k_im) per band      ~64 cycles SIMD
  - s_trig accumulate: 4 mul + 1 add per band x 128   ~32 cycles SIMD
  - s_norm: 5 ops per band x 128                      ~40 cycles SIMD
  - Reduce 128 partial sums                           ~10 cycles
  Total: ~322 cycles per (head, pos), zero trig

At 32 cores, 32768 iters per layer, 1.6 GHz:
  per-layer compute = 32768 * 322 / 32 / 1.6e9 = 205 us
  per-dispatch = 205 us + 0.5 ms host overhead = 0.7 ms
  per-token (46 layers) = 32 ms
  vs iGPU 67 ms = 2.1x faster (52% lift)

Even at 2x pessimistic (real memory latency, ObjectFifo overhead):
  per-token = 64 ms = on par with iGPU

Either way, this is the path to real lift. The reformulation is
mathematically equivalent (modulo float roundoff) and the parity
test will catch any divergence from the iGPU reference.

## Mitigation: precompute angle LUT

If trig is too slow, observe that omega(f) is per-band-only
(constant per model+layer). The per-position angle = omega(f) * p
can be precomputed for all p in [0, max_seq] x [0, n_bands]:
  16 KB LUT (4096 positions x 128 bands x 1 byte radians-quantized)

This eliminates 4 of the 12 transcendentals (the cos's argument
is then just a lookup + mul + add). Reduces per-iter to 8
transcendentals:
  - 4 sqrt + 4 atan2 = 8 transcendentals/iter

Plus the cos still has to evaluate the actual angle. Could also
precompute cos(angle) as a 2D LUT but it depends on k_phase which
is data-dependent, so no.

Combined: 32-core kernel with 8 transcendentals/iter, 100 cycles
each = 0.51 ms compute + overhead = 1.0 ms/dispatch = 46 ms/token
(1.4x faster than iGPU, 32% lift).

## Implementation phases

Phase A: Author single-(head, pos) C++ kernel using Peano libm.
  - No multi-core, no SIMD; just verify the math compiles and
    produces the right answer per the iGPU reference.
  - Bench standalone: cycles per (head, pos) for raw scalar libm.
  - This is 4-8 hours of authoring + debug.

Phase B: Multi-core fan-out using the existing 8c MLIR template.
  - Reuse columnar split. Each compute tile runs Phase A C++.
  - Bench at full layer shape.
  - 1-2 hours after Phase A.

Phase C: Optimize once measurements land.
  - If trig is fast enough (Phase B beats iGPU): ship.
  - If marginal: add precomputed LUT (cos angle table).
  - If still slow: investigate vectorized trig via Taylor approx
    or CORDIC.

## Verification

CPU-side reference is the existing `triattn_score_asym3` kernel
result downloaded as bytes; per-element scalar comparison per the
existing `triattn_gpu_parity_bf16` pattern. Tolerance: same as the
bf16 parity test (Pearson > 0.999, max rel < 1e-2).

## Risks

  1. Scalar transcendental latency is unknown until measured.
     Could blow up the projected lift.
  2. AIE-2P L1 SRAM is 32 KB per tile. Per-(head, pos)
     intermediate state (k_mag, k_phase, c_mag, c_phase, angle,
     omega per band x 4 bands x 8 floats x 4 cores per col = 1
     KB per col) is small but not free.
  3. atan2/sqrt may have weird denormal handling on aie2p that
     introduces subtle differences vs iGPU.

## Pointers

  iGPU score kernel:    kernels/src/triattn_score_asym3.hip
  Multi-core template:  kernels/aie2p/asym3_dequant_layer_8c/
  Stage 1.5 telemetry:  bench/npu-stage-1.5-scoping-2026-05-02.txt
  Multi-core scaling:   bench/npu-multicore-scaling-2026-05-02.txt
  Spec:                 docs/plans/asym3-codec-budget.md
