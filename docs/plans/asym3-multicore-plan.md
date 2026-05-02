# asym3_dequant_layer multi-core MLIR plan

## Why

Single-core layer kernel is BW-bound at output (524 KB at N=1024 in
2.46 ms = 213 MB/s, far below NPU's 30 GB/s peak). At full layer
shape (N=32768, 16 MiB output), single-core compute alone is ~6 ms
per dispatch, projecting 276 ms/token for 27B Gemma decode. iGPU
baseline is 67 ms/token. Single-core does NOT beat iGPU.

Multi-core projection at full layer shape:
  | Cores | Compute | Per-dispatch | Per-token (46 layers) | vs iGPU |
  |-------|---------|--------------|----------------------|---------|
  |   1   | 6 ms    | 6.5 ms       | 299 ms               | 4.5x slower |
  |   4   | 1.5 ms  | 2.0 ms       |  92 ms               | 1.4x slower |
  |   8   | 0.7 ms  | 1.2 ms       |  55 ms               | 1.2x faster |
  |  32   | 0.18 ms | 0.7 ms       |  32 ms               | 2.1x faster |
  | BW    | 0.53 ms | 1.0 ms       |  46 ms               | 1.5x faster |

The 30 GB/s memory subsystem caps the win at ~1.5x: the 16 MiB
writeback per layer is the floor. 32 cores is enough to saturate
BW; more cores are wasted. 8 cores is the smallest configuration
that beats iGPU baseline.

Stage 1.5 needs at least 8 cores for a meaningful "lift" measurement.
Stage 2.6 fused score is independent of core count (it eliminates
the writeback entirely by keeping K on-NPU through scoring).

## Architecture

Strix Halo NPU2 geometry: 8 columns, 5 rows.
  - row 0: shim_noc (host DMA bridge)
  - row 1: mem_tile (L2 SRAM, 256 KB per tile)
  - rows 2-5: compute tiles (32 KB local SRAM each, AIE-2P core)
  - 8 cols x 4 compute rows = 32 compute tiles total

For dequant, iterations are fully independent (no cross-core data
flow). Simplest topology: per-column strip handles N_ITERS/N_COLS
chunks, with each column having its own input/output ObjectFifo
chain. Within a column, distribute across rows for further fan-out
(packed/cnorm broadcast from mem to all 4 rows; outputs aggregate
back to mem).

## Implementation phases

Each phase ships independently and is verified via the existing
verify_asym3_dequant_layer harness (which is N_ITERS-agnostic via
the embedded constants).

### Phase A: 4-core single-column

  4 compute tiles, all in column 0, rows 2-5.
  N_ITERS = 1024, each core handles 256 chunks.
  Scope: ~150 lines of MLIR (still single-column DMA path, just
  broadcast packed/cnorm + aggregate output).
  Estimated effort: 3-4 hours including debug.
  Expected: ~5x compute speedup vs single-core; per-dispatch latency
  drops from 2.46 ms to ~0.6 ms.

### Phase B: 8-core, 2 columns

  Split N_ITERS across 2 columns, 4 rows each = 8 compute tiles.
  Two parallel column-strips, each running the Phase A topology
  on half the data.
  Estimated effort: 1-2 hours after Phase A (mostly duplication).
  Expected: ~10x compute speedup; per-dispatch ~0.4 ms.

### Phase C: 32-core, full array

  All 8 columns, all 4 compute rows = 32 cores.
  Estimated effort: 2-3 hours after Phase B (duplication + DMA
  channel routing). Hits BW wall, so further parallelism past
  this point would be wasted.
  Expected: per-dispatch ~0.7 ms (BW-bound write-back).

## Verifier compatibility

The existing verify_asym3_dequant_layer.rs doesn't care how many
cores the kernel uses; it just sets up the BO inputs / dispatches /
reads back outputs. The same 100-seed and 1000-seed harnesses
exercise multi-core variants via env-overridable PDI/insts paths.

Determinism gate becomes more important post-multi-core: any race
condition or per-core state leakage shows up as run-to-run divergence.
The two-dispatch determinism check in the verifier is the existing
guard.

## Engine integration post-multi-core

Once Phase C is in (or even Phase B at meaningful perf), wire into
crates/engine/src/triattn.rs::Mode::Asym3 path:

```rust
Mode::Asym3 => {
    if std::env::var("HIPFIRE_NPU_DEQUANT").map(|v| v == "1").unwrap_or(false) {
        // NPU path: dequant to bf16, then bf16 score.
        npu_runtime.asym3_dequant_layer(packed, cnorms, &mut bf16_k_buf)?;
        gpu.upload_bytes(&bf16_k_buf, &bf16_k_dev_buf)?;
        gpu.triattn_score_bf16(
            &bf16_k_dev_buf, &centers, &cos_theta, &sin_theta, &scores,
            n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
        )
    } else {
        // iGPU baseline path (existing).
        gpu.triattn_score_asym3(
            k_cache, &centers, &cos_theta, &sin_theta, &scores,
            n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
        )
    }
}
```

Two follow-ups required:
  1. Variable n_pos: kernel hardcodes N_ITERS=1024, but real decode
     has variable seq_len. Either (a) shape-specialized variants
     for power-of-2 seq_len buckets and pad to next bucket, or
     (b) runtime-N parameterization in MLIR (substantial rewrite).
     Option (a) embeds 6-7 PDI binaries (~25 KB total). Recommended.
  2. Prefetch wiring: kick layer N+1 NPU dispatch on its own stream
     while iGPU runs layer N. AsyncFence. Engine-side scheduler.

## Sequencing

  Phase A (4-core) -> Verify -> Bench at single n_pos = 128
  Phase B (8-core) -> Verify -> Bench at N_ITERS=1024
  Phase C (32-core) -> Verify -> Stage 1.5 A/B at full shape
  Variable-N variants -> Engine integration -> Stage 1.5 lift gate

Phase A + B is the minimum that proves multi-core works and beats
the BW-bound floor at moderate shapes. Phase C is required for the
production stage 1.5 bench.

## Risk register

  - Multi-core ObjectFifo routing has subtle ordering rules that
    aren't documented in the IRON examples; first attempt may need
    multiple iterations to get correct.
  - mem_tile stage in the cascade might add latency at our small
    per-iter chunk size (96+4 bytes input, 512 bytes output). Need
    to measure whether L2 staging is necessary or shim->core direct
    is faster.
  - Determinism across multi-core: ObjectFifo acquire ordering may
    not be deterministic if core scheduling is non-deterministic.
    The two-dispatch determinism check covers this.
  - Build time may grow significantly (32-core compile is ~5x of
    single-core in the matmul reference).

## Pointers

  Single-core source: kernels/aie2p/asym3_dequant_layer/
  Multi-core reference: /home/kaden/mlir-aie/programming_examples/
    basic/matrix_multiplication/whole_array/build/
    aie_1024x1024x1024_32x32x32_4c.mlir (641 lines)
  Verifier: crates/hipx/src/bin/verify_asym3_dequant_layer.rs
  Spec: docs/plans/asym3-tier1-layer-kernel.md
  iGPU baseline: tests/speed-baselines/gfx1151.txt
