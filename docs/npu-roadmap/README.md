# NPU Roadmap (Strix Halo AIE-2P NPU2) — archive

**Status: paused. Both wedges closed as not-viable.**

This directory archives the npu-roadmap exploration on AMD Strix Halo
(gfx1151 iGPU + AIE-2P NPU2). Two attempted wedges, both closed with
measured negative results in `MANUAL_REVIEW.md`:

  ESCALATED-3/4/5: triattn decode (NPU dequant + iGPU score) -> 80x
                    slower than iGPU baseline. Closed 2026-05-02.
  ESCALATED-6:     prefill via concurrent NPU 32c i8 GEMM -> 1.24x
                    max lift, below 1.4x gate; bandwidth-bound on
                    NPU2's 4-shim-for-8-cols A topology. Closed
                    2026-05-02.

The kernels, dispatch wrappers, benches, and verifiers built during
the investigation stay in the tree as reusable infrastructure (see
"What's still in the tree" below). The default
`HIPFIRE_NPU_DEQUANT=0` stays; nothing in the engine depends on
NPU offload.

## Why this archive exists

A future contributor (or future-you) may want to revisit one of:
  - HXQ / XQ4 split iGPU+NPU concurrent quant (#28, deferred; different
    bandwidth profile than INT8 GEMM, may be viable)
  - Lever 1 command-list batching (#47, defensive infra; ABI exists,
    no near-term consumer)
  - Persistent NPU kernel pattern (#48, would change the bandwidth
    economics fundamentally; ~40-80h of MLIR authoring)
  - Different NPU workload (fused GEMM + activation + quant, higher
    compute-to-bandwidth ratio, may overlap better on UMA)

The findings here document what was tried, what worked, what didn't,
and crucially WHY the dataflow constraints make the tested approaches
not viable. Don't re-run the same experiments.

## Index of findings

  COMPACTION_HANDOFF.md         Snapshot of mid-session state at the
                                  pivot from triattn (closed) to prefill
                                  exploration. Useful as a summary of
                                  the phase boundary.

  prefill-session-summary-2026-05-02.md
                                ONE-PAGE summary of the prefill
                                investigation (start here if reading
                                cold).

  prefill-xdna-abi-scan.md      AMDXDNA kernel driver UAPI scan. Verdict:
                                Lever 1 multi-cmd-per-EXEC_CMD ABI
                                ALREADY EXISTS via cmd_count + cmd_handles
                                (DrmExecCmd, /home/kaden/xdna-driver/
                                include/uapi/drm/amdxdna_accel.h:264-267).
                                hipx submit_exec_cmd already supports it.

  prefill-iron-dispatch-analysis.md
                                Trace of mlir-aie IRON's Python +
                                pyxrt dispatch path on hipx. Verdict:
                                IRON also single-cmd-per-ioctl. Gap
                                between hipx and IRON is ~5-10us
                                (instruction BO pooling), not 50us.
                                Lever 1 batching is a real win nobody
                                takes today.

  prefill-igpu-int8-baseline.md iGPU INT8 GEMM throughput at production
                                Qwen3-27B prefill shapes on gfx1151:
                                18-20 TOp/s steady, batch >= 128.
                                gemm_hfq4g256_mmq_set (Q8_1 act x HFQ4 w
                                x i8 WMMA accum). Numbers feed the lift
                                gate.

  prefill-32c-bringup-progress.md
                                Mid-investigation state (after 512^3
                                worked, before 1024^3 / 2048^3). Notes
                                the DMA descriptor 1023-per-dim limit
                                that blocks default tile sizes at 1024^3
                                and the workaround via b-col-maj.

  phase0-xdna-baseline.md
  phase1-blockers.md
  phase1-strategy.md            Earliest exploration: kernel module install,
                                ABI surface, "above BW class" thesis.
                                Now dated.

  mmq-channel-test-{9b,27b}-gfx1151.md
                                Earlier Strix Halo iGPU MMQ channel
                                tests. Predate this branch's NPU work.

  SESSION_SUMMARY.md
  loop-progress.md              Earlier session notes from before the
                                triattn decode close.

## What's still in the tree (reusable)

```
crates/hipx/                                     hand-rolled XDNA Rust wrapper
crates/hipx/src/bin/matmul_i8_512_32c.rs          full-array i8 GEMM dispatch
crates/hipx/src/bin/matmul_i8_1024_32c.rs            + bench, Knuth-hash verifier
crates/hipx/src/bin/matmul_i8_2048_32c.rs            (3 shapes, all correct)
crates/hipx/src/bin/matmul_i8_{512,1024,2048}_4c.rs  4-core variants
crates/hipx/src/bin/matmul_bf16_*.rs                 bf16 variants
crates/hipx/src/bin/asym3_dequant_*.rs               asym3 K-cache dequant
crates/engine/examples/bench_int8_gemm_prefill.rs   iGPU INT8 GEMM bench
crates/engine/examples/hipfire_x_stage_1_5_ab.rs    triattn A/B harness
crates/engine/src/npu.rs                            NpuRuntime engine API
                                                     (gated on HIPFIRE_NPU_DEQUANT)
kernels/aie2p/matmul_i8_*_4c/                       4-core matmul artifacts
kernels/aie2p/matmul_i8_*_32c/                      32-core matmul artifacts
kernels/aie2p/asym3_dequant_*/                      dequant kernels
kernels/aie2p/asym3_score_*/                        fused-score kernels
kernels/aie2p/matmul_bf16_*_4c/                     bf16 matmul artifacts
docs/plans/asym3-tier1-layer-kernel.md
docs/plans/asym3-fused-score-plan.md
docs/plans/asym3-multicore-plan.md
docs/plans/aie2p-bf16-mul-shape.md                  AIE-2P-shape doc
                                                     (RTZ + RAZ rounding)
bench/prefill-igpu-int8-20260502.txt
bench/stage-1.5-ab-20260502-204713.txt
bench/stage-2.6-perf-status-2026-05-02.txt
bench/npu-multicore-scaling-2026-05-02.txt
bench/npu-stage-1.5-scoping-2026-05-02.txt
```

## Final measurements (anchors for any future revisit)

  iGPU INT8 GEMM (gfx1151) production prefill:    18-20 TOp/s
  4c i8 1024^3 (single-row baseline):              4.55 TOp/s
  32c i8 1024^3 (full array, best shape):          4.74 TOp/s   (+4%)
  Concurrent split lift at best 32c:               1.24x        (gate: 1.4x)
  NPU dispatch overhead (steady-state):            ~67 us
  Triattn iGPU baseline (single dispatch):         17 us
  Triattn NPU+iGPU path B:                         1372 us      (80x slower)

## Key architectural facts (do not relitigate)

  1. AIE-2P NPU2 array: 4 rows x 8 cols compute, 8 mem tiles, 8 shim tiles
  2. mlir-aie whole_array generator sets n_shim_mem_A = min(n_aie_rows,
     n_aie_cols) = 4 when n_aie_cols = 8. Per-column A bandwidth is
     therefore HALF the 4c topology. This is the prefill bandwidth wall.
  3. AIE-2P i8 microkernel mac dim is 8x8x8 (1024 ops/cycle/core)
  4. AIE-2P bf16 mul has hardware-shape rounding: RTZ on cnorm, RNE on
     codebook, RAZ on output. See docs/plans/aie2p-bf16-mul-shape.md.
  5. mlir-aie DMA descriptor size limit: max 1023 per dim. Row-major
     B at 1024^3 hits this; --b-col-maj 1 uses a different DMA pattern
     that fits.
  6. Per-dispatch overhead floor (hipx and IRON both): ~50-67us. Comes
     from ioctl + KMD scheduling. Lever 1 batching could amortize this
     across N kernels.
  7. KMQ (kernel-managed queue) is the only mode AIE-2P exposes; UMQ
     (user-managed) doorbell field is present but inert.
  8. ROCm 7.2.2 + xdna-driver DKMS v0.7.0 + mlir-aie + Peano installed
     on hipx; hand-rolled clang++/aiecc build path (no Python in build).

## When to come back

A revisit would be justified by ONE of:
  - A dataflow change in mlir-aie that lifts the n_shim_mem_A = 4 limit
    (would require generator update; check upstream changelog)
  - A workload that's NOT GEMM-shaped: HIGH compute-to-bandwidth ratio,
    fits in compute tile L1, doesn't need broadcast across rows
  - A new dispatch lever (Lever 2 persistent kernel) implemented and
    measured to change the per-dispatch economics
  - A different NPU generation (Strix Halo successor) with different
    array topology

Without one of those, the triattn and prefill conclusions stand.
