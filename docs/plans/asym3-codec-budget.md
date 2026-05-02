# asym3 KV codec on NPU — budget analysis before MLIR-AIE authoring

**Status**: pre-authoring spec. Decides at what granularity an asym3
dequant kernel becomes profitable to dispatch through `engine::npu::
NpuRuntime` versus the existing iGPU path
(`gpu.kv_fold_asym3` / `gpu.triattn_score_asym3`).

**Audience**: hipfire-x maintainer choosing whether to invest the
~3 days of MLIR-AIE authoring + verification work to ship an
NPU-side codec.

## TL;DR

**Per-layer dequant + K-prefetch pipeline is profitable for
non-spec decode** (1457 µs iGPU layer hides the 630 µs NPU BW
window with a 2.3× margin). Ship `asym3_dequant_layer_to_bf16`
as Tier 1.

**DFlash-batched decode needs the fused score kernel.** Post-rebase
iGPU is 27 % faster (260 µs/layer instead of 330 µs), so the
prefetch window no longer closes for the headline DFlash 27B-3.5
workload. The fix is to drop the bf16 K writeback (84 % of the
NPU I/O) by fusing dequant+score on NPU. Tier 2:
`asym3_score_all_layers`.

**Hard prereq for either tier**: dmabuf import iGPU↔NPU
(Task #9). Without zero-copy K cache access from NPU, the
copy cost (~100 µs/layer) eats most of the DFlash budget.

**Per-call dispatch (`kv_fold_asym3`, naive per-layer-per-token
dequant) is not profitable** — dispatch overhead is 5–40×
larger than the compute. Don't author these.

## Format details (engine source of truth)

Source: `kernels/src/kv_fold_asym3.hip`,
`crates/engine/src/cask.rs:86-90`.

K cache layout per (position, head):
```
[4-byte cnorm: f32] [(head_dim × 3) / 8 bytes: packed 3-bit indices]
```
For head_dim = 256 → 4 + 96 = **100 bytes per head**.

Dequant rule:
```
k[d] = cnorm × TURBO_C3_256[idx_d]
```
where `TURBO_C3_256` is an 8-entry constant codebook
(`-0.135, -0.083, -0.046, -0.015, +0.015, +0.046, +0.083, +0.135`)
and `idx_d` is the d-th 3-bit field unpacked from the byte stream.

The dequant is *post-Givens-rotated*: the engine stores K in a
basis where Givens rotations have already been applied at insert
time, so dequant reproduces the rotated K. Score / fold ops then
either apply inv-Givens or operate in rotated space (the fold path
exploits orthogonality and stays in rotated space — see commentary
in `kv_fold_asym3.hip`).

## Reference workload (27B Gemma, decode)

| field            | value      |
|------------------|-----------:|
| n_layers         |         46 |
| n_kv_heads       |          8 |
| head_dim         |        256 |
| seq_len (decode) |     4 096+ |
| dtype dequanted  | bf16 / fp16 |

## Per-call compute estimates

### Op A: kv_dequant_asym3 (standalone dequant K → bf16 for scoring)

Inputs per layer per token:
- K cache slice for this layer: `seq_len × n_kv × bytes_per_head`
  = 4096 × 8 × 100 = **3.2 MiB** read
- Output bf16: `seq_len × n_kv × head_dim × 2`
  = 4096 × 8 × 256 × 2 = **16 MiB** write

Element count: `seq_len × n_kv × head_dim` = 8 388 608.
Compute: 1 multiply per element = **8.4 MOp** of effective work.

| metric                          | value         |
|---------------------------------|---------------|
| element count                   | 8.4 M         |
| FMAs (1 per dequanted element)  | 8.4 MOp       |
| compute @ 1.03 TOp/s BF16       | **8 µs**      |
| compute @ 2.23 TOp/s INT8       | **4 µs**      |
| dispatch overhead (no-copy)     | ~80 µs        |
| dispatch overhead (full)        | ~580 µs       |

Verdict: dispatch is **10–70× larger than compute**. Per-layer
per-token offload is a clear loss; the iGPU path has already
fused dequant into the score kernel, paying ~zero overhead for it.

### Op B: per-token whole-forward dequant (all 46 layers batched)

Same arithmetic, ×46:
- element count: 386 M
- compute @ 1.03 TOp/s BF16: **375 µs**
- compute @ 2.23 TOp/s INT8: **172 µs**
- dispatch overhead (no-copy): ~80 µs (one call)

Verdict: dispatch is **~25 % of compute** at BF16, **~50 % at
INT8** — viable. NPU compute fraction crosses 50 % at this
granularity.

But: this requires the engine to expose K cache slices for ALL
layers concurrently, which doesn't match the current per-layer
forward pass. The engine processes layers sequentially because
each layer's residual feeds the next layer's input.

A workable compromise: **K-prefetch async pipeline.** While
layer N's iGPU work runs (~1.2 ms — see `hipfire_x_overlap_rigor`
mode B), kick off NPU dequant of layer N+1's K cache. By the
time the iGPU finishes layer N, layer N+1's dequanted K is
ready, hiding the NPU dispatch entirely behind iGPU work.

### Op C: kv_fold_asym3 (eviction-time merge)

Source: `kernels/src/kv_fold_asym3.hip`, called once per
eviction event (every ~M=4 tokens at typical CASK m-fold).

Per fold call:
- output positions: `budget` (e.g. 256)
- per output: `m × n_kv × head_dim` MACs + a renormalize/requant tail
  = 4 × 8 × 256 = 8 192 MACs per output
- total: 256 × 8192 = **2.1 MOps**

Compute @ 1.03 TOp/s BF16: **2 µs**.
Dispatch overhead (no-copy): ~80 µs.

Verdict: dispatch is **40× larger than compute**. Per-call
fold offload is a worse loss than per-layer dequant. Not worth
authoring.

## Decision matrix

| op                          | granularity        | offload? | rationale                            |
|-----------------------------|--------------------|----------|--------------------------------------|
| dequant_asym3 per layer/token| 1 layer            | ❌ no    | 8 µs compute < 80 µs overhead        |
| dequant_asym3 all-layers     | 46 layers batched  | ✅ yes   | 375 µs compute @ BF16, 25 % overhead |
| kv_fold_asym3                | per eviction       | ❌ no    | 2 µs compute, ~40× overhead          |
| **K-prefetch pipeline**      | layer N+1 hidden   | ✅ yes   | overhead hides behind iGPU layer N   |

## Design refinement (2026-05-02)

After surveying the engine call sites: a standalone `dequant_asym3
→ bf16` kernel on NPU has **no current downstream consumer**.

- `gpu.triattn_score_asym3` does fused dequant+score inline; it
  takes asym3 K directly and would need a parallel `_bf16` variant
  (and the matching iGPU kernel doesn't exist).
- `gpu.kv_fold_asym3` operates in asym3-space (Givens orthogonality
  exploit) — does not need bf16 K either.
- Per-token decode `gpu.attention_asym3_kv` likewise fuses dequant.

So the offload target is not `dequant_asym3` standalone — it's a
**full asym3 score kernel** on NPU that subsumes both dequant and
the score reduction. This re-shapes the spec:

- Kernel: `asym3_score_all_layers` — input is the fa_layer_ids set
  of K caches + a pre-rotated centers matrix; output is `[n_layers
  × budget × n_kv_heads × head_dim]` scores in bf16.
- Compute per call (27B Gemma, 4K context, 46 fa-layers):
  46 × (4096 × 8 × 256) = 386M MACs → 375 µs at 1.03 TOp/s bf16.
- Dispatch overhead: 80 µs no-copy (one call covers all 46 layers).
- Engine call site: `crates/engine/src/cask.rs::eviction_step`
  line 134 (`for (fa_i, &layer_idx) in self.fa_layer_ids…`).
  Hook converts the per-layer loop into a single batched NPU
  submit + per-layer iGPU continuation.
- Replaces / shadows: `gpu.triattn_score_asym3` per-layer call
  (line 146 in cask.rs); the NPU runs ahead of the iGPU.

## Recommended kernel to author

**Two-tier authoring plan (revised 2026-05-02):**

1. **Tier 1 — `asym3_dequant_layer_to_bf16`**: ships first as the
   simpler kernel. Validated for *non-spec decode* and *prefill*
   (1457 µs iGPU layer comfortably hides the 630 µs NPU BW window).
   Lets us prove out the dispatch + dmabuf import + engine wiring
   on a tractable kernel shape. Decoupling MLIR-AIE risk from
   engine-integration risk.

2. **Tier 2 — `asym3_score_all_layers` fused dequant + score**:
   eliminates the bf16 K writeback (84 % of NPU I/O) which closes
   the prefetch window for DFlash workloads (260 µs iGPU layer).
   Compute density goes from 1 mul/element (BW-bound) to
   `head_dim` MACs/element (compute-bound at 1.03 TOp/s BF16).
   Higher MLIR-AIE risk; gated on Tier 1 shipping.

Tier 1 kernel signature:

```rust
pub fn asym3_dequant_layer(
    &mut self,
    layer: usize,
    seq_len: usize,
    k_cache_layer: &[u8],   // [seq_len × n_kv × bytes_per_head] = ~3.2 MiB
    out_bf16: &mut [u16],   // [seq_len × n_kv × head_dim] = ~8.4 M elements
) -> Result<u64, hipx::XdnaError>;  // returns NPU seq for later wait
```

The kernel itself is a streaming dequant: read 3 bytes → 8 indices →
8 codebook lookups → 8 fma with cnorm → 8 bf16 stores. ~9 instructions
per output element. AIE-2P single-core sustained at ~30 GOp/s on this
shape; with 4 cores that's 120 GOp/s, and 8.4 MOp / 120 GOp/s = 70 µs
of compute — well inside the iGPU's per-layer budget.

The kernel does NOT need:
- Givens rotation (output stays in rotated space, consistent with
  the engine's downstream score path — same simplification the
  fold kernel uses)
- Renormalization / requantization (only fold needs that)
- Adjacent-position interactions (each (pos, head) decodes
  independently — embarassingly parallel)

That's the right shape for MLIR-AIE: a 1-D streaming kernel with
LUT lookup, no inter-tile dependencies, easily mapped to the
4-core whole_array pattern.

## What "shipping it" looks like

1. Author `asym3_dequant_layer.mlir` (m=k=variable=0, vector
   dequant). Reuse the codebook as a constant memtile load.
2. Build for several seq_len shapes: 1024, 2048, 4096, 8192. The
   dequant is parametric on seq_len so build cost is small.
3. Add `engine::npu::NpuRuntime::asym3_dequant_layer_*` plumbing
   parallel to matmul_i8_*; expose `_a_buf` for K cache, `_c_view`
   for bf16 output.
4. Wire into `engine/src/cask.rs::eviction_step` to run K-prefetch
   for layer N+1 concurrent with iGPU eviction for layer N.
5. Bench: replicate the `hipfire_x_overlap_rigor` shape but with
   real engine ops; report tok/s on 27B Gemma decode with
   asym3_dequant on/off.

## What NOT to author yet

- `kv_fold_asym3` on NPU — too small per call, dispatch dominates.
- `triattn_score_asym3` on NPU — fused dequant+score is in the
  iGPU sweet spot already (~10 TFLOPs sustained on gfx1151 fp16
  WMMA), and the NPU's strength is INT8/BF16 GEMM not the rotation
  arithmetic that scoring needs.
- Per-layer dequant — only worthwhile under the prefetch pipeline.

## Risk register

- **Codebook fits in memtile**: 8 × f32 = 32 bytes, trivially.
- **Streaming bandwidth**: 3.2 MiB read + 16 MiB write per layer
  = 19 MiB I/O ÷ ~30 GB/s host bandwidth = 630 µs *if not overlapped*.
  Compute is much faster (70 µs), so this is BW-bound, not compute-
  bound. The K-prefetch pipeline assumption (NPU work hides behind
  iGPU layer N) only holds if iGPU layer time > 630 µs.
  → **VALIDATED 2026-05-02 on hipx (gfx1151) via the existing
    speed-baseline file.** From `tests/speed-baselines/gfx1151.txt`
    (commit 42566de, captured on hipx hardware):

    ```
    27b mq4 gen:               14.9 tok/s = 67 ms/token
                              ÷ 46 layers = 1457 µs / layer
    27b 3.5 dflash lru code:   65.83 tok/s with τ=8.85
                              effective batch-of-9 verify:
                              1000 / 65.83 / 46 = 330 µs / layer
                              (× ~9 verify slots = wider iGPU GEMMs)
    ```

    Pure single-token decode: **1457 µs/layer = 2.3× the 630 µs
    BW threshold.** Pipeline closes comfortably.

    DFlash batched verify: 330 µs/layer is *below* the threshold,
    but the apparent shortness is because each layer is now doing
    a batch-9 GEMM (target.verify on draft tokens) which is GEMM-
    fraction-dominated rather than GEMV-bandwidth-dominated. The
    iGPU layer is still touching K cache, just packed denser per
    wall-clock unit. Net: the prefetch pipeline still has at
    least 1457 µs of "raw decode" time to hide the 630 µs NPU
    BW window in, so the close holds for `prefill / non-spec /
    long-context` paths. For DFlash-batched decode it's a tighter
    fit and benches need to confirm the close at 4K+ context
    (where K cache reads grow linearly and the iGPU layer time
    stretches accordingly).

  - **Refresh 2026-05-02 (post-rebase + ROCm 7.2.2):** master's
    MMQ-auto + per-weight screen + gemv-fusion + k2x32 work
    landed and lifted gfx1151 prefill +11–68 % across all sizes.
    Decode is *unchanged* (BW-wall on weight reads — fundamental
    LPDDR5X-8000 ≈ 256 GB/s ceiling). DFlash 27B-3.5 LRU code
    workload is *faster*: 65.83 → 83.76 tok/s, τ 8.85 → 10.64.
    New per-layer numbers from `tests/speed-baselines/gfx1151.txt`
    @ commit `8759d93`:

    ```
    27b mq4 gen:               14.9 tok/s = 67 ms/token
                              ÷ 46 layers = 1457 µs / layer  (UNCHANGED)
    27b 3.5 dflash lru code:   83.76 tok/s with τ=10.64
                              1000 / 83.76 / 46 = 260 µs / layer
                              (was 330 µs; iGPU is 27 % faster
                               so the NPU prefetch window tightens)
    ```

    Pure decode close still holds (2.3× margin). **DFlash-batched
    close fails harder post-rebase**: 260 µs iGPU layer < 630 µs
    NPU BW window. The K-prefetch pipeline as originally specced
    cannot hide NPU dequant under DFlash workloads. Three options:

    1. **Drop the bf16 K writeback** — fuse dequant+score on
       NPU so only the smaller score output crosses the memory
       subsystem. The 16 MiB bf16 K write is the dominant BW
       term (3.2 MiB read + 16 MiB write = 19 MiB; the 16 MiB
       writeback is 84 % of the I/O). Eliminating it brings the
       NPU window from 630 µs to ~110 µs at 30 GB/s, well inside
       the 260 µs DFlash budget. **Requires authoring the full
       fused score kernel, not just dequant — i.e., the
       `asym3_score_all_layers` path in the design refinement
       below.**
    2. **Keep the per-layer dequant kernel + skip prefetch
       pipeline on DFlash decode** — gate offload to non-spec
       paths (prefill, raw single-token decode, long-context
       refresh). Loses the headline workload but ships the
       simpler kernel.
    3. **Restructure to batch multiple-layer dequant per call**
       — amortize per-call BW across ≥3 layers so prefetch
       headroom returns. Adds plumbing complexity but stays in
       the dequant-only kernel shape.

    **Recommendation: option (1).** The fused score kernel is
    where the NPU's INT8/BF16 GEMM strength actually shows up —
    a standalone dequant is bandwidth-bound on the writeback,
    not compute-bound. The compute density of fused score is
    roughly 8× the dequant kernel (head_dim MACs vs 1 mul per
    output element), which directly shifts the kernel from
    BW-bound to compute-bound and unlocks the 1.03 TOp/s BF16
    headline. See revised plan under "Recommended kernel to
    author" below.

- **bf16 vs fp16**: engine uses fp16 in many spots, NPU produces
  bf16. Mantissa precision differs. Need to verify the score
  kernel's accuracy doesn't regress on bf16-dequanted K — likely
  fine (asym3 quant error is the dominant precision loss anyway,
  bf16 vs fp16 is third-order) but a coherence-gate run is a hard
  prereq before landing.

- **K cache memory residency**: K cache is allocated through
  ROCm/HSA on the iGPU side. NPU consumption requires either
  (a) dmabuf import (Task #9, pending) — the K cache memory
  mapped into NPU's address space, true zero-copy; (b) DMA copy
  per layer at ~100 µs cost, eating most of the 260 µs DFlash
  budget; or (c) re-allocating K cache through PASID-shared
  SHMEM, which forces all asym3 ops (including the iGPU score
  path) onto the same allocator. Path (a) is the only viable
  route for production codec offload — **dmabuf import is the
  hard prereq, not the kernel itself.**

## Pointers

- Source asym3 dequant inside score: `kernels/src/triattn_score_asym3.hip`
- Source asym3 fold kernel: `kernels/src/kv_fold_asym3.hip`
- Engine call site: `crates/engine/src/cask.rs:146`
  (`gpu.triattn_score_asym3`) and `:275` (`gpu.kv_fold_asym3`)
- Existing NPU API patterns to copy: `engine::npu::NpuRuntime::
  matmul_i8_1024_4c_*` (init/buf/sync/submit/wait_no_copy/c_view)
- Overlap budget reference: `hipfire_x_overlap_rigor` example,
  per-trial median saved 448–457 µs (43 % wall-clock) at this
  shape pair.
