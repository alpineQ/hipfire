# Manual Review Queue (session/strix-halo-2026-05-02)

PR triage notes from movie-night managerial session. Items here need
Kaden's review or hardware-blocked validation before merge.

## PR #127 — HARDWARE-GATED

- Title: feat(gfx906): wave64 FP16 hybrid prefill kernels (+90% speedup)
- Recommendation: merge after Kaden validates on V620/MI50 hardware
- What was tested: nothing executable; this session's hardware is gfx1151,
  not gfx906/gfx908.
- What was reviewed:
  - dispatch.rs predicate `is_gcn5_wave64()` matches `"gfx906" | "gfx908"`
    exactly. Fallback path preserved for all other arches.
  - Branch order across 4 dispatch sites: wave64 check goes BEFORE
    `should_use_mmq` and respects existing `HIPFIRE_FP16=0` opt-out.
  - Kernel C++ pattern (qkv variant): `warp_id = tid >> 5`,
    `gid = blockIdx.x * 2 + warp_id`, bounds check `if (gid >= total_m)
    return`. Standard 2-rows-per-block wave64 split. Lane math correct.
- Concerns:
  - gfx908 (MI100) included in the gate but no perf data in the PR body;
    only gfx906 measurements (74 -> 141 tk/s). gfx908 might be safe but
    is not validated as a win.
  - No correctness check beyond the coherence battery (which only
    hard-fails on panics, zero tokens, or timeouts; output divergence
    is not a hard fail per CLAUDE.md). A logit-divergence comparison
    against the FP16 wave32 path on gfx906 would be a stronger signal.
  - No env-level escape hatch specific to this hybrid path; if it
    regresses, only `HIPFIRE_FP16=0` (which kills the entire FP16
    prefill path) is available.
  - BATCH_TILE=8 hardcoded in 4 kernel files; not tunable per-arch.
- Link: https://github.com/Kaden-Schutt/hipfire/pull/127

## PR #123 — REVIEW-NEEDED (DRAFT, author: fivetide)

- Title: fix(thinking): hard-suppress thinking when thinking=off
- Recommendation: approach is correct (max_think_tokens=1 cap, NOT
  directive injection); ready-for-review when author satisfied. The
  design space has been searched in this repo before; six commits
  across 24h tried various directive-injection placements and all
  broke a different prompt shape.
- What was tested: nothing executed; DRAFT, did not check out.
- What was reviewed:
  - cli/index.ts: `run()` and `serve()` both add the
    `if (cfg.thinking === "off") max_think_tokens = 1` branch with
    correct precedence over per-model numeric cap.
  - 99-bottles measurement (1545 -> 418 tokens, 0 -> 20 bottle
    mentions on Qwen3.6-35B-A3B) is convincing.
  - Coherence-gate 4/4 OK in PR body.
- Concerns:
  - Cap=1 may truncate `<think>` mid-marker if a model tokenizes
    that text into multiple tokens. Worth verifying across the
    registry's `thinking=off` defaults.
  - No regression test added (a 99-bottles canary in
    coherence-gate.sh would protect this).
- Link: https://github.com/Kaden-Schutt/hipfire/pull/123

## PR #124 — REVIEW-NEEDED (DRAFT, author: fivetide)

- Title: fix(dflash): wire max_think_tokens through DFlash spec-decode
- Recommendation: approach is sound; address the O(N^2) decode
  comment (or confirm AR path has same shape, accept), then mark
  ready. Complements #123 by adding the same cap to the spec-decode
  path. Should sequence after or with #123.
- What was tested: nothing executed; DRAFT, did not check out.
- What was reviewed:
  - daemon.rs:1333 generate_dflash() now takes max_think_tokens.
  - Per-token `tokenizer.decode_bytes(&streamed_tokens)` -> rfind
    on `<think>`/`</think>` markers. Works because Qwen tokenizes
    those as bare ASCII.
  - Force-close emits `</think>\n` token JSON and breaks the
    spec-cycle. Honestly disclosed limitation: no answer tokens
    after force-close because DFlash can't splice mid-cycle.
- Concerns:
  - O(N^2) decode cost on long generations (decode_bytes on the
    full streamed buffer per emitted token).
  - String-literal marker matching could mis-fire if the model
    emits literal `<think>` text in answer tokens (low probability
    for Qwen).
  - No regression test added.
- Link: https://github.com/Kaden-Schutt/hipfire/pull/124

## PR #125 — REVIEW-NEEDED (DRAFT, author: fivetide)

- Title: fix(daemon): add runtime n-gram loop detector to AR generation
- Recommendation: approach is correct and conservative
  (env-disable available); address the per-token rebuild cost if
  cheap (or accept the ~1-2% overhead at default settings),
  validate against a code-completion prompt to confirm no false
  positives outside the coherence battery's 4 prompts, then mark
  ready. Complements #123/#124 as the answer-phase counterpart to
  the think cap.
- What was tested: nothing executed; DRAFT, did not check out.
- What was reviewed:
  - daemon.rs AR loop: 4-gram HashMap rebuilt per token over a
    256-token sliding window, threshold=8 repeats forces EOS.
  - Both threshold and window are env-tunable. Threshold=0 disables.
  - Operates on token IDs (no decode overhead).
- Concerns:
  - HashMap rebuilt from scratch per token is O(window) when O(1)
    is achievable via incremental update. ~256 hashmap ops per
    token at default settings.
  - Coherence-gate validated 0.8b/4b/9b only; code-completion
    prompts at temp=0 might have higher 4-gram baseline density.
  - AR-only; DFlash doesn't get the same protection (though
    DFlash's spec-cycle structure may make answer-phase loops
    rarer).
- Link: https://github.com/Kaden-Schutt/hipfire/pull/125

## PR #115 — SCOPE-OR-POLICY (DRAFT, author: Kaden himself)

- Title: feat(mq-lloyd): Lloyd-Max codebooks for MQ3 / MQ2 — help
  wanted to clear ship gates
- Recommendation: parked pending Kaden's direction. He authored this
  draft and the title explicitly says "help wanted to clear ship
  gates" — this signals he knows what the gates are; speculative
  review without that context is not high-leverage.
- What was tested: nothing.
- What was reviewed: only metadata (7000+/-175 in a single branch).
- Concerns: None applicable; this is Kaden's PR to drive.
- Link: https://github.com/Kaden-Schutt/hipfire/pull/115
