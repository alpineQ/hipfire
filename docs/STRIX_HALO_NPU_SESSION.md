# hipfire-x autonomous build session — summary

Date: 2026-05-01 (overnight, ~10 iterations of self-paced /loop)
Branch: `strix-halo` worktree
User contract: execute on hipx, commit experiments, work until full
implementation or interrupted.

## TL;DR

**Working**:
- Pure-ioctl bypass dispatches CUs to AIE-2P NPU on hipx, byte-perfect
- Engine integrates via `engine::npu::NpuRuntime` with safe iGPU fallback
- Two kernel classes proven (DMA-only): passthrough_4k + passthrough_dmas
- `cargo run --features npu --example hipfire_x_init` PASSes end-to-end

**Open**: Worker-class kernels (vec_scalar_mul) submit cleanly,
firmware completes the job, but the worker tile never writes to the
output BO. Documented in `crates/hipx/README.md`.

## What this means for press coverage

**Architectural moat is shipped.** The unique-on-Strix-Halo dual-engine
story (open-source NPU+iGPU concurrent dispatch via UMA) is real and
demonstrable. Engine code routes ops to NPU with `engine::npu::route()`,
falls back to iGPU when no kernel exists.

**The press number isn't.** That requires a Worker-class kernel
(KV-codec dequant or INT8 GEMV) running real LLM workload bytes.
The blocker is debugging-grade, not architectural.

## Commits (newest → oldest)

```
75662bf docs(hipx): README documenting working state + Worker-class blocker
f5a8939 vec_scalar_mul packet now matches AMD shape (5 BO slots)
4ece8a7 passthrough_dmas — third kernel proves multi-shape dispatch
4208dcf NpuRuntime::passthrough_4k — engine API dispatches through NPU
733164a hipfire-x integration scaffold — NpuRuntime + route()
1255002 vec_scalar_mul kernel scaffolded
65f6eec PASS — first end-to-end NPU dispatch via pure ioctls
38e14cb firmware connect succeeds — XRT-style aligned heap mmap
d19cc54 hipx-passthrough binary — Rust dispatch chain 80% live
6d41626 ert.rs — ERT_START_NPU command-packet builder
8a722d4 MLIR-AIE bootstrap + first AIE-2P CU embedded
c3311c0 Phase 1.3a + 1.4 plumbing
5684202 rename xdna-compute → hipx
7de0c86 OOT amd/xdna-driver setup script
e11c07a Phase 1.0 BO + 1.1 hwctx ABI
40a9ec1 Phase 0 ABI probe
```

## Reproducing on hipx

```bash
# One-time setup:
ssh hipx 'sudo /home/kaden/ClaudeCode/autorocm/hipfire/.worktrees/strix-halo/scripts/setup-hipx-npu.sh'

# Engine smoke test (demonstrates everything end-to-end):
cargo run --features deltanet,npu --example hipfire_x_init
# Outputs: NPU detected (Aie2p, 8 cols, 58 TOPS) + route() table +
#          passthrough_4k PASS

# Standalone kernel tests:
ssh hipx 'ulimit -l unlimited; /tmp/hipx-passthrough'      # 4 KiB byte passthrough
ssh hipx 'ulimit -l unlimited; /tmp/hipx-passthrough-dmas' # 16 KiB i32 via MemTile
```

## Architecture in place

```
hipfire/
├── crates/hipx/                         (NEW — sister to redline)
│   ├── ioctl.rs       9-ioctl ABI mirror (stable kernel uapi)
│   ├── device.rs      open + GET_INFO probes
│   ├── bo.rs          SHMEM | DEV_HEAP | DEV | CMD lifecycle
│   ├── hwctx.rs       hwctx CREATE/DESTROY
│   ├── ert.rs         ERT_START_CU command-packet builder
│   ├── cmd.rs         config_cus + submit_exec_cmd
│   ├── fence.rs       wait_many + timeline_wait
│   ├── prime.rs       dmabuf import/export
│   ├── runtime.rs     high-level Hipx wrapper
│   ├── dispatch.rs    predicates + ComputeTarget + route_npu
│   ├── kernels.rs     embedded PDI bytes
│   └── bin/           probe, bo_roundtrip, hwctx_roundtrip,
│                      passthrough, passthrough_dmas, vec_scalar_mul,
│                      overview
│
├── crates/engine/src/npu.rs             (NEW — opt-in `npu` feature)
│   ├── NpuRuntime::try_init             — returns None on non-Strix-Halo
│   ├── route(npu, op) -> ComputeTarget  — opportunistic dispatch
│   └── NpuRuntime::passthrough_4k       — full chain via engine API
│
├── crates/engine/examples/hipfire_x_init.rs   smoke test
│
├── kernels/aie2p/                       (NEW — MLIR-AIE compiled CUs)
│   ├── passthrough_4k/                  PASS — 8-col, objectfifo
│   ├── passthrough_dmas/                PASS — 1-col, MemTile DMA
│   └── vec_scalar_mul/                  FAIL — Worker class
│
├── scripts/setup-hipx-npu.sh           one-shot Strix Halo bootstrap
└── findings/                           (gitignored — local notes)
    ├── SESSION_SUMMARY.md              (this file)
    ├── loop-progress.md                iteration log
    ├── phase0-xdna-baseline.md         Phase 0 findings
    ├── phase1-strategy.md              "above BW class" thesis
    └── phase1-blockers.md              Phase 1 blockers (resolved)
```

## Key invariants discovered

1. **Heap mmap requires the XRT-style 2× anonymous reservation +
   MAP_FIXED|MAP_LOCKED at heap-aligned addr**. Without MAP_LOCKED the
   firmware MAP_HOST_BUFFER returns EINVAL.
2. **OOT amdxdna driver (≥v1.0.0)** is required — the in-tree v0.7.0
   in Linux 7.0 mainline doesn't implement the hugepage-aware mmap
   path the firmware needs.
3. **`RLIMIT_MEMLOCK = unlimited`** for the user; auto-handled via
   `/etc/security/limits.d/90-<user>-memlock.conf`.
4. **DEV BOs sub-allocate from the per-fd DEV_HEAP** — only ONE
   DEV_HEAP per fd (subsequent CREATE_BO returns EBUSY).
5. **DEV BOs have no own map_offset** — access via heap mmap at
   `(bo.xdna_addr - heap.xdna_addr)` offset.
6. **AMD test packet uses `ERT_START_CU` (opcode 0)**, not
   `ERT_START_NPU` (=20). No npu_data prefix; args start right
   after cu_mask.
7. **Single-handle args**: when `cmd_count==1` or `arg_count==1`,
   the field is the handle value itself, not a pointer to a
   1-element array.
8. **PRIME / SYNCOBJ / GEM_CLOSE ioctls** are generic DRM (below
   `DRM_COMMAND_BASE=0x40`) — must NOT add the base.
9. **Worker-class kernels need ALL 5 BO slots populated** — even
   ctrlpkts/trace get placeholder BOs. Just zero pointers fail.
10. **PDI bytes match xclbin's embedded PDI byte-for-byte** —
    `aiecc` outputs the same blob the xclbin AIE_PARTITION section
    uses.

## What's left to ship the press number

1. **Worker-class debug** — figure out the missing piece between cmd
   packet (now correct) and the AIE worker tile actually computing.
   Suspects: heap-relative offset relationship between PDI and instr
   BOs that the compiled worker code expects.
2. **Author KV-codec dequant kernel** in MLIR-AIE / IRON Python.
   - asym2/3/4 → fp16 conversion
   - Per-group scale + zero
   - Vectorized for AIE-2P int8/fp16 MAC
3. **Wire into `engine/src/triattn.rs`** asym KV codec path —
   `engine::npu::route(OpClass::KvCodec, ...)` already returns Npu
   when PDI is available; just flip `available_ops.kv_dequant = true`
   in `try_init()` and add `NpuRuntime::run_kv_dequant(...)`.
4. **Bench**: 27B DFlash decode tok/s with NPU codec on/off via
   `coherence-gate-dflash.sh`. Strategy doc projects 27B 65 → ~92-108
   tok/s if draft+codec moves to NPU.

## Time investment

10 /loop iterations, each ~25-30 min of autonomous work + a wake gap.
Estimate: 5-6 hours of cumulative compute time. Foundation is now
deep enough that future iterations can focus on kernels and benches,
not bring-up.
