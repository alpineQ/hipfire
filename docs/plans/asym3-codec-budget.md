# asym3 KV codec on NPU — budget analysis before MLIR-AIE authoring

**Status**: pre-authoring spec. Decides at what granularity an asym3
dequant kernel becomes profitable to dispatch through `engine::npu::
NpuRuntime` versus the existing iGPU path
(`gpu.kv_fold_asym3` / `gpu.triattn_score_asym3`).

**Audience**: hipfire-x maintainer choosing whether to invest the
~3 days of MLIR-AIE authoring + verification work to ship an
NPU-side codec.

## TL;DR

**Per-layer dispatch is not profitable.** With 1024^3 INT8 at
2.23 TOp/s and BF16 at 1.03 TOp/s through the engine API, a
single layer's K dequant + scoring at 27B Gemma shapes is
~16 µs of compute — 5–35× smaller than the dispatch overhead.

**Full-forward-pass dispatch is profitable.** Batching all 46
layers into one NPU call drops the dispatch fraction below 1 %.
At that granularity, INT8 quant + dequant fuses cleanly into the
existing 1024^3 GEMM kernel and runs at ~700 µs total — directly
overlappable with the iGPU pipeline measured in
`hipfire_x_overlap_rigor` (43 % wall-clock saved).

**Author the kernel as a forward-pass-batched dequant**, not a
per-layer one. Per-layer routing keeps the iGPU path.

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

## Recommended kernel to author

**`asym3_dequant_layer_to_bf16`** — operates on a single layer's
K cache slice but dispatched ASYNC under the K-prefetch pipeline.
Signature:

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

- **bf16 vs fp16**: engine uses fp16 in many spots, NPU produces
  bf16. Mantissa precision differs. Need to verify the score
  kernel's accuracy doesn't regress on bf16-dequanted K — likely
  fine (asym3 quant error is the dominant precision loss anyway,
  bf16 vs fp16 is third-order) but a coherence-gate run is a hard
  prereq before landing.
- **bf16 vs fp16**: engine uses fp16 in many spots, NPU produces
  bf16. Mantissa precision differs. Need to verify the score
  kernel's accuracy doesn't regress on bf16-dequanted K — likely
  fine (asym3 quant error is the dominant precision loss anyway,
  bf16 vs fp16 is third-order) but a coherence-gate run is a hard
  prereq before landing.

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
