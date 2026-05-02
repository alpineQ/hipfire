# Phase 1 strategy — "above the bandwidth class" on Strix Halo

Date: 2026-05-01
Worktree: `strix-halo` (gfx1151 + aie2p)
Premise: Phase 0 confirmed the amdxdna ABI works; now decide *what to
build with it* before grinding more plumbing.

## The bandwidth ceiling is real

LPDDR5X 8000 MT/s × 256-bit bus = **256 GB/s peak** on Strix Halo.
Practical iGPU sustained BW measured at ~180-220 GB/s. Hipfire 9B
decode on this machine is 46.0 tok/s (committed floor); 9B model at
mq4 ≈ 4.5 GB; per-token iGPU BW = 4.5 × 46 = 207 GB/s. **The iGPU is
already saturating LPDDR5X for decode.**

Adding a second compute engine (NPU) does **not** double available BW.
Both engines share the same DRAM channels. The question for "above BW
class" is: can NPU+iGPU concurrent execution either (a) pull on
disjoint cache-resident regions, (b) run a workload phase that's
compute-bound and would otherwise have stalled the iGPU, or (c)
amortize a shared weight load across two engines doing different work
on it.

## Workloads that can break the ceiling

| Strategy | Mechanism | Fit on hipfire | Ceiling lift |
|---|---|---|---|
| Concurrent prefill+decode | NPU prefills next batch while iGPU decodes current | Limited — interactive serving doesn't usually have a "next batch" queued | Pipeline-only, not BW-breaking |
| **Spec-decode draft offload** | NPU runs draft model concurrent with iGPU verify; engines pull from disjoint weight regions | **Direct fit — DFlash is hipfire's flagship decode path** | **+50-65% verify rate (math below)** |
| MoE expert offload | Some experts on NPU | A3B MoE supported, but each expert is BW-bound itself | Marginal |
| KV-tier on NPU SRAM | Hot KV slot kept in NPU's tile memory | NPU has only ~24 KB on-chip — too small for production KV | Not viable on AIE-2P |
| Specialized integer fast paths | NPU INT8 throughput exceeds iGPU INT8 WMMA | Only meaningful at compute-bound shapes (large batch prefill) | Mostly redundant with iGPU MMQ at pp2048 |

**Spec-decode draft offload is the clear win.** The math below shows
why it can lift the BW ceiling, not just pipeline around it.

## Spec-decode draft offload — the concrete arithmetic

### Today's hipfire 27B DFlash on Strix Halo

From `tests/speed-baselines/gfx1151.txt`:
- 27B target weight @ mq4 ≈ **13.5 GB**
- 27B's draft @ mq4 = **919 MB** (`qwen35-27b-dflash-mq4.hfq`)
- Decode 65.83 tok/s, τ=8.85, accept rate 0.583

Per accepted output token, the iGPU does:
- `1/τ` ≈ **0.113 verify passes** on the 27B target (batched over the τ
  candidates) → 0.113 × 13.5 GB = **1.53 GB**
- **1 draft pass** on the 0.92 GB draft model → **0.92 GB**
- Total iGPU BW per accepted token: **~2.45 GB**
- At 65.83 tok/s × 2.45 GB = **161 GB/s effective BW** (~75% of
  practical peak).

The draft pass is currently consuming **38% of iGPU BW per accepted
token**. That's the bandwidth we can reclaim.

### After NPU draft offload

iGPU only does verify: 1.53 GB per accepted token. At 200 GB/s
practical sustained = 7.65 ms per token = **131 tok/s ceiling** for
the iGPU side alone.

NPU side simultaneously must produce 1 draft pass (0.92 GB read) per
accepted token, in less than 7.65 ms wall-clock. NPU practical BW on
LPDDR5X is conservatively ~50 GB/s when iGPU is also pulling (the
controller schedules across both clients). That's 18 ms for one draft
pass — **slower than verify**, so NPU draft becomes the limiter.

So the realistic ceiling is **draft-pass-rate-limited**, somewhere in
the **100-110 tok/s range** for 27B DFlash on Strix Halo. Compared to
today's 65.83 → **+50-65% lift, ~92-108 tok/s.**

### 9B DFlash equivalent

- 9B target @ mq4 ≈ 4.5 GB; 9B draft = 557 MB
- Today: 151 tok/s, τ=13.15
- Per token: 0.076 × 4.5 + 1 × 0.557 = **0.90 GB iGPU**
- After offload: iGPU does 0.342 GB → could hit 580 tok/s if
  draft-side keeps up. NPU draft of 0.557 GB at ~50 GB/s = 11.1 ms =
  90 tok/s NPU ceiling.
- Realistic with overlap: **~200-280 tok/s**, +30-85% over 151.

### Why this beats raw iGPU MMQ tuning

The MMQ work in master closed the *prefill* gap (354 → 1100 tok/s
pp2048 vs llama.cpp 1076). Decode is where hipfire still has headroom,
and decode is BW-bound. **No iGPU-side kernel optimization can break
the BW ceiling.** This is why the NPU is uniquely valuable on Strix
Halo: it's a second BW client, not a second compute pile.

## Why Strix Halo specifically

The same plan does *not* work on a discrete GPU:
- dGPU + dNPU would communicate over PCIe (hundreds of µs/MB latency,
  bandwidth-bottlenecked at PCIe 4 × 16 = 32 GB/s). Sharing KV cache
  between draft and target across PCIe = catastrophic.
- On Strix Halo, both engines are PCIe **endpoints inside the SoC**
  pulling from the **same UMA pool**. KV handoff between draft and
  target is a dmabuf import — no copy.

This is the architectural moat: nobody else can do this. llama.cpp,
vLLM, mlc-llm, exllamav2 — none of them have an NPU draft path
because most production hardware is dGPU. **Strix Halo is the first
mass-market platform where this is even possible.**

## Plan B: what if MLIR-AIE doesn't ship a working PDI for AIE-2P?

Risk register:

| Risk | Severity | Mitigation |
|---|---|---|
| MLIR-AIE 1.3 nominally targets AIE-2P but PDIs don't actually load on `npu_7` | High — gates everything | Use a known-good PDI (FastFlowLM ships them; Riallto / IRON examples) as smoke test before authoring our own |
| AMD Ryzen AI SDK installer rejects STX-H (#366) | Low — installer-only; the underlying XRT/xdna-plugin path works (FastFlowLM proves) | Build XRT from source if needed; we already plan to bypass XRT entirely |
| Direct ioctl path needs more state than docs reveal (PDI parsing, CU config layout) | Medium | Mine XRT source as reference (`amd/xdna-driver` userspace tree). Their EXEC_CMD sequence is the ground truth |
| dmabuf cross-import iGPU↔NPU has unexpected friction (PASID, IOMMU) | Medium | Probe early (Phase 1.4); PRIME spec mandates this should work |
| NPU sustained BW lower than 50 GB/s when iGPU is hot | Low-Medium | Microbench in Phase 1.5; if true, lift expectation to +30% rather than +50% |

## Phase 1 success criteria

| Phase | Deliverable | Pass condition |
|---|---|---|
| 1.0 | BO alloc + map round-trip | Pattern survives SYNC_BO TO/FROM_DEVICE |
| 1.1 | hwctx create + destroy | syncobj_handle returned non-zero, destroy clean |
| 1.2 | Scout MLIR-AIE / IRON PDI for AIE-2P | One known-good PDI in hand (theirs or ours) |
| 1.3 | No-op CU dispatch | EXEC_CMD returns; QUERY_HW_CONTEXTS shows command_completions = 1 |
| 1.4 | dmabuf import iGPU↔NPU | NPU reads pattern written by iGPU compute |
| 1.5 | INT8 GEMV microbench | Measure (tok/s draft size, GB/s) under: NPU-only, iGPU-only, NPU+iGPU concurrent |

After Phase 1.5 we know whether the spec-decode draft offload thesis
is real before writing a Rust port of `dflash.rs` for the NPU.

## What we don't build yet (intentional)

- A general "NPU backend for any tensor op". Decode is BW-bound; only
  draft offload is sized right. Don't generalize too early.
- An iGPU↔NPU scheduler. After Phase 1.5 we'll know what shape of
  pipelining works; building a scheduler before that is premature.
- Kernel compilation at hipfire build-time. Phase 1.2 settles what's
  shippable; if we end up with one PDI per draft model class, baking
  PDI bytes is fine.

## Out of scope while in `strix-halo` worktree

(Unchanged from worktree scope rule.) gfx1100/1030/1010/1011/1200/1201
work happens in feat branches off master; this worktree is gfx1151 +
aie2p only.

## Where this leaves the iGPU MMQ tool-call bug

User explicitly de-prioritized: "we need to change the game then fix
the tool call issue". The MMQ revert (commit `d1506d0`, opt-in only)
is the safe state. Auto-MMQ stays disabled until the i8 WMMA
correctness bug at large batch + tool-call distribution is bisected.
That's a separate worktree's problem; it doesn't gate NPU work.
