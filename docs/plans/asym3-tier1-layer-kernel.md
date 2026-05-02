# Stage 1.4: asym3_dequant_layer Tier 1 production kernel

This doc scopes the per-layer-batched asym3 dequant kernel that is
required before the stage 1.5 iGPU vs NPU A/B bench can produce
a real tok/s number.

## Motivation

`asym3_dequant_256` (shipped as the 1.1 verifier kernel) is single-
(head, position): one dispatch covers 256 dim values for one head at
one cache position. Steady-state dispatch latency on hipx is ~67 us
per call (measured by the 1.3b shadow harness at 1024 dispatches).

For 27B Gemma decode at seq_len=4096, the iGPU consumes K cache for
46 layers x 8 kv_heads x 4096 positions per token. If we naively
called `asym3_dequant_256` for each (layer, head, position), one
token's K-cache reads would cost roughly:

    46 * 8 * 4096 * 67 us = 101 seconds per token

versus the iGPU baseline of 67 ms per token (14.9 tok/s). Off by
1500x. Per-(head, position) dispatch is fundamentally wrong
granularity.

The codec budget doc (`docs/plans/asym3-codec-budget.md`) already
called this: the right granularity is **per-layer-batched** (one
dispatch per layer covering all heads x all positions). That's
Tier 1 production scope.

## Target shape

One layer of 27B Gemma K cache at 4K context:

| dim                | value                                  |
|--------------------|----------------------------------------|
| n_kv_heads         | 8                                      |
| head_dim           | 256                                    |
| seq_len            | up to 4096                             |
| input bytes        | 8 * 4096 * (4 + 96) = 3.2 MiB          |
| output bytes (bf16)| 8 * 4096 * 256 * 2 = 16 MiB            |
| element count      | 8 * 4096 * 256 = 8.4 M                 |

Single dispatch = one layer = 8.4 M output elements. Compute is
trivially parallel across (head, position, dim_chunk) so all 32 AIE
cores can chew on it concurrently. Per the codec budget, expected
compute time is ~70 us at 1.03 TOp/s BF16 sustained (compute) and
~630 us at 30 GB/s NPU memory-subsystem bandwidth (BW-bound on the
16 MiB writeback). The 630 us figure is the practical per-layer
budget; the prefetch pipeline assumption is "iGPU layer time >
630 us" so the NPU dequant fits inside one iGPU layer's window.

For 27B raw decode (1457 us / layer, comfortable 2.3x margin) the
prefetch closes. For DFlash 27B-3.5 LRU code (260 us / layer
post-rebase) it does NOT close with the dequant-only kernel; that's
why Tier 2 (fused score) is the answer for the headline DFlash
workload.

## Kernel structure (proposed)

Source: `kernels/aie2p/asym3_dequant_layer/`

```cpp
// Per-layer batched: caller sets n_pos at dispatch time via a
// scalar arg (or builds shape-specialized variants if scalar arg
// is awkward in the AIE-2P intrinsics).
extern "C" void asym3_dequant_layer(
    uint8_t *packed,    // n_kv_heads * n_pos * 100 bytes
    float   *cnorms,    // n_kv_heads * n_pos f32 (broadcast per dim)
    bfloat16 *out,      // n_kv_heads * n_pos * 256 bf16
    int32_t n_kv_heads,
    int32_t n_pos
);
```

Engine API mirror:

```rust
// in NpuRuntime
pub fn asym3_dequant_layer(
    &mut self,
    packed: &[u8],     // n_kv_heads * n_pos * 100 bytes
    cnorms: &[f32],    // n_kv_heads * n_pos
    out: &mut [u8],    // n_kv_heads * n_pos * 256 * 2 bytes
    n_kv_heads: usize,
    n_pos: usize,
) -> Result<(), hipx::XdnaError>;
```

Lazy-init pattern matches asym3_dequant_256: persistent hwctx + bound
CU + reused max-size BOs (sized for n_kv_heads=8, n_pos=4096, ~3.5
MiB packed + 16 MiB output). Shrinkable on smaller dispatches via
SYNC_BO partial size.

## Tile placement

Per-tile work plan (approximate; exact partitioning is a kernel
authoring decision):

- 32 cores (8 cols x 4 rows) divide n_kv_heads x n_pos x dim_chunks.
- Dim chunks of 16 lanes (existing SIMD width) gives 16 chunks per
  256-dim head.
- Per (head, position): 16 chunks of compute, ~1024 cycles total.
- Streaming over heads x positions across cores reduces per-core
  iteration count by 32x.

## Verification

- Same gates as 1.1 verifier (max 4 bf16 ULP, mean signed bias <= 1
  ULP per element, deterministic).
- Compare output against `asym3_dequant_256` ran 1 (head, position)
  at a time over the same input. Bit-for-bit equivalent expected
  (same kernel logic, just larger batching).
- Re-run the 1.3b shadow harness shape (16 simulated layers x 8
  heads x 8 positions) using the new kernel. Expected: same ULP
  envelope, dramatically lower wall clock (~46 layers' worth in
  ~30 ms instead of seconds).

## What stage 1.5 does after this lands

1. Wire `NpuRuntime::asym3_dequant_layer` call site into the engine
   K-cache decode hot path. Specifically: cask.rs::eviction_step
   (per-eviction) AND the per-token attention scoring path that
   consumes asym3 K. The latter requires a matching iGPU
   `triattn_score_bf16` kernel that takes pre-dequanted bf16 K
   instead of asym3-encoded K (the iGPU side currently fuses
   dequant with scoring, so a separated bf16 input path doesn't
   exist yet).

2. K-prefetch wiring: kick off layer N+1 NPU dispatch while iGPU
   does layer N. Async fence per dispatch.

3. Bench: 27B MQ4 raw decode (per `tests/speed-baselines/gfx1151.txt`
   the iGPU baseline is 14.9 tok/s). 3 trials, sigma reported,
   prompt md5, governor pinned, quality diff against 5 deterministic
   prompts. Lift = NPU enabled - baseline; stat-significant lift
   plus quality-clean = flip default `HIPFIRE_NPU_DEQUANT=1`.

4. If lift is < 5% or quality drifts: do not flip default. Escalate
   to MANUAL_REVIEW.md with telemetry; stage 2.6 (fused score) may
   be the right next move instead.

## Effort estimate

- Kernel C++ + aie.mlir authoring with batched DMA flow: ~4-8 h.
- Build via the same hand-rolled aiecc pipeline as
  asym3_dequant_256: trivial after the source lands.
- Verifier + shadow harness extension: ~1 h.
- Engine API + integration into iGPU score path (requires new iGPU
  `triattn_score_bf16` kernel too): ~6-12 h.
- Bench + quality diff: ~2 h.

Total: 12-24 hours of focused work to close stage 1.5 properly.
This is the next major chunk on the chain.

## Why not just author it now in this iteration

Per the npu-roadmap contract three-strike rule, this is a
substantial unanticipated scope expansion (stage 1.5 in the
contract was sized for "wire integration + bench", not "author a
new production kernel + new iGPU kernel + integrate"). Documenting
it as stage 1.4 with a clear scope is the correct unblock action.
