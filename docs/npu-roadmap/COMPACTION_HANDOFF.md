# Compaction handoff: prefill viability exploration on Strix Halo NPU

Supersedes the prior dequant/score handoff. The npu-roadmap chain
(stages 1.4-2.6) closed last session with measured A/B confirming
NPU is not viable for triattn decode. Pivot: prefill via concurrent
INT8 GEMM. This file is the handoff for the post-compaction iter.

## Context recap

Stages 1.4-2.6 escalation (commit 0c1365b on npu-roadmap/2026-05-02):

  Path A (iGPU baseline triattn_score_asym3):  17 us median
  Path B (NPU dequant + iGPU bf16 score):    1372 us (80x slower)

Default `HIPFIRE_NPU_DEQUANT=0` stays. ESCALATED-3/-4/-5 in
MANUAL_REVIEW.md document the closure.

## Current direction: prefill viability

Prefill is the OPPOSITE shape from triattn:
  - compute-bound (dominated by FFN matmul)
  - embarrassingly parallel
  - NPU-iGPU concurrency is additive (UMA shared memory)

### Foundational measurements taken this session

NPU sustained throughput (release build, hipx):
  matmul_i8  1024^3 (4c):  4.55 TOp/s   (8% of 58 TOPS peak)
  matmul_i8  2048^3 (4c):  4.60 TOp/s   (8%)
  matmul_bf16 1024^3 (4c): 1.15 TOp/s   (BF16 is dead, 4x penalty)

Read: 4-core kernels achieve ~8% of peak. INT8 holds across
shapes (compute-bound). Going to 32-core (linear scaling
projection): ~37 TOp/s.

iGPU baseline numbers (from speed-baselines/gfx1151.txt):
  27b_mq4_pp32_prefill_tok_s = 163.2
  27b_mq4_pp128_prefill_tok_s = 167.3

iGPU INT8 GEMM TOp/s: NOT YET MEASURED (task #43).

### User insight: dispatch path may be the wrong optimization target

User parallel: hipfire's "redline" custom KMD dispatch path was
slower than HIP runtime despite removing layers. Likely same
applies here: our hipx raw-ioctl dispatch may be slower than
AMD's official XRT runtime path. AND more importantly, the real
lever is eliminating per-dispatch overhead entirely:

  Lever 1: command-list submit (multi-cmd-per-ioctl, hipGraph analog)
  Lever 2: persistent NPU kernel (descriptor ring polling, sub-us
           effective dispatch)

For triattn (already escalated): per-dispatch overhead is only
16% of total cost. Even if XRT made dispatch free, triattn drops
1372 -> 1170 us, still 70x slower than iGPU.

For prefill (proposed): if 32c lands ~37 TOp/s and dispatch
overhead is small (200 us out of e.g. 640 us), concurrent-split
with iGPU could give 1.5-2.2x prefill lift.

### What the user said: "we could just copy whatever IRON does"

Yes, tractable: read XRT C++ source and port relevant fast paths
to hipx Rust. Targets: BO pool reuse, ioctl batching, persistent
host-pinned mappings. ~4-12h after task #42 identifies the actual
optimizations.

## Task DAG (current TaskList #41-#48)

  #41 Scan xdna ABI for command-list / chained dispatch
       -> findings/prefill-xdna-abi-scan.md
       -> grep amdxdna_accel.h for list/queue/chain/batch terms
       -> ~30 min
  #42 Read mlir-aie IRON dispatch path
       -> findings/prefill-iron-dispatch-analysis.md
       -> identify XRT optimizations IRON uses
       -> ~1-2h
  #43 Bench iGPU INT8 GEMM at production prefill shapes
       -> bench/igpu-int8-gemm-prefill-shapes-<ts>.txt
       -> 4608x4608 + 4608x36864 (FFN up shape)
       -> ~30-60 min
  #44 IRON Python vs hipx Rust 1:1 dispatch comparison [BLOCKED #42]
       -> bench/iron-vs-hipx-dispatch-<ts>.txt
       -> ~1-2h
  #45 Build matmul_i8_32c kernel [BLOCKED #41,#42,#44]
       -> 8 cols x 4 rows fan-out (extend asym3_dequant_layer_8c
          template)
       -> ~6-12h
  #46 Standalone concurrent-split GEMM bench [BLOCKED #43,#45]
       -> the go/no-go decision point
       -> threshold: >=1.4x lift -> proceed; <1.4x -> escalate
       -> ~2-4h
  #47 Lever 1 (command-list dispatch) impl [BLOCKED #41,
                                              conditional on verdict]
       -> ~4-8h if ABI exists; NONE if not
  #48 Lever 2 (persistent NPU kernel) scoping [BLOCKED #44,#47]
       -> spec doc only; deferred work
       -> ~2-4h spec; 40-80h to build

### Sequence to follow

Step 1 (parallel, ~3h total): #41 + #42 + #43 simultaneously
Step 2 (~1-2h): #44 (after #42 lands)
Step 3 (~1h): #47 (only if #41 found a list ABI)
Step 4 (~6-12h): #45 (after #41/#42/#44)
Step 5 (~2-4h): #46 (the decision point)
Step 6: branch on #46 outcome:
  >=1.4x lift -> engine integration + 27B prefill bench (new task)
  <1.4x lift  -> #48 spec persistent kernel OR escalate prefill

## Key files / locations

### Existing NPU GEMM bench bins (already build clean on hipx)

  crates/hipx/src/bin/matmul_i8_2048.rs (binary: hipx-matmul-i8-2048)
  crates/hipx/src/bin/matmul_i8_1024.rs (binary: hipx-matmul-i8-1024)
  crates/hipx/src/bin/matmul_bf16_1024.rs (binary: hipx-matmul-bf16-1024)
  crates/hipx/src/bin/matmul_512.rs

### Existing NPU GEMM kernel binaries

  kernels/aie2p/matmul_i8_2048_4c/build/main.pdi (4c, M=K=N=2048)
  kernels/aie2p/matmul_i8_1024_4c/build/main.pdi (4c, M=K=N=1024)
  kernels/aie2p/matmul_i8_512_4c/build/
  kernels/aie2p/matmul_bf16_1024_4c/build/
  kernels/aie2p/matmul_bf16_512_4c/build/

These were ported from mlir-aie programming_examples (compiled
artifacts only; source not in our tree). For #45 we need to
either build matmul_i8_32c by extending an existing 4c MLIR or
authoring fresh from the asym3_dequant_layer_8c template.

### Multi-core MLIR template (proven by dequant work last session)

  kernels/aie2p/asym3_dequant_layer/        (1c reference)
  kernels/aie2p/asym3_dequant_layer_2c/     (2 cols)
  kernels/aie2p/asym3_dequant_layer_4c/     (4 cols)
  kernels/aie2p/asym3_dequant_layer_8c/     (8 cols, full column,
                                              best-perf reference)

Use the 8c MLIR as the topology starting point; extend to 8x4=32
cores by adding rows 3, 4, 5 to each column. The matmul reference
in /home/kaden/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/
shows full 4c x 4-row = 16-tile matmul; we extend to 8c x 4-row.

### Hipx dispatch surface (for porting / measuring)

  crates/hipx/src/cmd.rs    - submit_exec_cmd, config_cus
  crates/hipx/src/ioctl.rs  - SYNC_BO, EXEC_CMD, SYNCOBJ_TIMELINE_WAIT
  crates/hipx/src/fence.rs  - timeline_wait
  crates/hipx/src/bo.rs     - BO management

### XDNA driver source

  /home/kaden/xdna-driver/include/uapi/drm/amdxdna_accel.h
  (target for #41 ABI scan)

### MLIR-AIE / IRON reference

  /home/kaden/mlir-aie/python/iron/                 (target for #42)
  /home/kaden/mlir-aie/programming_examples/        (test.py harness
                                                     for #44 IRON)

### iGPU side

  crates/rdna-compute/src/dispatch.rs - existing iGPU GEMM wrappers
                                         (look for matmul_int8 /
                                          gemm_i8 entries)
  tests/speed-baselines/gfx1151.txt   - prefill numbers to beat

### Reusable patterns from last session

  Verifier dual-reference pattern: see
    crates/hipx/src/bin/verify_asym3_score_layer.rs
    (matching ref + iGPU-shape independent ref; PASS requires both)

  A/B bench pattern (use as #46 model): see
    crates/engine/examples/hipfire_x_stage_1_5_ab.rs
    Output format: bench/stage-1.5-ab-<ts>.txt with
    median/p95 + per-step breakdown + lift_pct.

## Operating environment

  Local branch:  npu-roadmap/2026-05-02 (tip: c759666)
  Worktree:      /home/kaden/ClaudeCode/autorocm/hipfire/.worktrees/strix-halo
  Workstation:   k9lin (gfx1100, 7900 XTX) - main dev box
  NPU machine:   hipx (Strix Halo, gfx1151 + AIE-2P NPU2)
                 - kernel build target (mlir-aie + Peano installed)
                 - /home/kaden/hipfire on hipx side

  Sync pattern:  edit on k9lin -> rsync to hipx -> build there
                 -> run -> rsync artifacts back. Or commit + push
                 -> hipx pulls. Both flows work; rsync is faster
                 for kernel-only edits.

  NPU build:     hand-rolled clang++ Peano + native aiecc, NO
                 Python in kernel build path. See
                 kernels/aie2p/asym3_dequant_layer_8c/build.sh
                 for the canonical pattern.

  hipcc PATH:    /opt/rocm/bin/hipcc on hipx; needs explicit
                 `export PATH=/opt/rocm/bin:$PATH` before running
                 engine examples that JIT-compile iGPU kernels.

  Cargo features: --features deltanet,npu for engine examples that
                  use both KvCache types AND NpuRuntime.

## Operating mode (carry over from prior contract)

  - Always commit on npu-roadmap/2026-05-02. No em-dashes.
  - Three-strike escalation: if a stage takes more than 3
    substantial attempts, write to MANUAL_REVIEW.md and move on.
  - No Python in kernel build (hand-rolled clang++ Peano +
    native aiecc).
  - Codex stop-time review hook is active; expect codex callbacks
    catching common issues (NaN gates, fmt cleanliness, build with
    documented features, etc.). Fix-and-continue.
  - When you complete a task, mark it completed via TaskUpdate,
    then claim the next unblocked task by ID order.
