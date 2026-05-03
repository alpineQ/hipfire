# Release notes draft — post v0.1.9-alpha.1 (2026-05-02)

Unreleased; collected from commits on `master` since
`24bf788 chore(release): bump to 0.1.9-alpha.1` through
`b9d1016 feat(cli): hipfire chat (#129)`.

Suggested next bump: `v0.1.9-alpha.2`.

## Highlights

### Strix Halo OOM fix on large models

`hipfire pull qwen3.5:122b-a10b` (and other models that exceed half of
unified-memory size) now loads correctly on Ryzen AI MAX+ 395 / 128 GB
GTT systems. Previously, the mmap-based weight loader doubled physical
RAM use during load (page cache + GPU copy), OOMing at layer 42 of 48
on the 122B MoE.

The new `pread`-based loader keeps the page-cache footprint bounded at
sub-MB during load. Validated on hipx (32 GB Strix Halo, 9B model):
master path evicted 137 MB of existing cache to fit the load; PR path
grew cache by 932 KB. Identical generation output (46.7-46.8 tok/s).

Non-Unix platforms fall back to the existing mmap path unchanged.

PR #128, commit `36ca314`.

### Interactive chat TUI

`hipfire chat <model>` is the new multi-turn interactive front-end.
Streaming tokens, multi-line input (CTRL+O for newline, bracketed
paste handles clipboards), basic markdown, slash commands
(`/help`, `/clear`, `/stats`, `/trim`, `/set`, `/exit`), input history
with in-progress draft preservation, and a clean ephemeral daemon
lifecycle that does not clobber a long-lived `serve -d`.

103 unit tests cover the pure helpers; three independent adversarial
reviews (Claude, Gemini, GLM-5) flagged 27 issues across the
implementation, all addressed before merge.

PR #129 (closes #63), commit `b9d1016`.

### Prompt normalization rule table

Prompts are now normalized via an explicit rule table covering CRLF,
NBSP, and trailing whitespace. Output is byte-identical to v0.1.9-alpha.1
on already-clean prompts; Windows-line-ending and locale-influenced
prompts now produce the same generation as Unix-line-ending prompts.

Per the existing whitespace-variance discipline (see CLAUDE.md prompt
normalization section), the previous prompt-collapse mechanism stays
default-on (HIPFIRE_NORMALIZE_PROMPT=1).

A Codex-flagged regression on prompts containing non-trailing tabs
was fixed in a follow-up commit (80330c3): tabs inside content are now
preserved verbatim rather than collapsed.

PR #122 (closes #40), commit `049bc17`. Codex follow-up commit `80330c3`.

## Other changes

- `4b89ad7` docs: CREDITS.md added (foundational sources, papers,
  contributors).
- `80b4d06` fix: refresh-credits abort splice on jq failure (Codex
  follow-up to credits work).

## Full commit list

```
b9d1016 feat(cli): hipfire chat — interactive TUI (closes #63) (#129)
36ca314 fix(load): pread-based weight loading to prevent OOM on unified-memory APUs (#128)
80330c3 fix(prompt-norm): preserve non-trailing tabs verbatim (Codex follow-up to #122)
049bc17 feat(prompt-norm): CRLF/NBSP/trailing-ws rule table (closes #40) (#122)
80b4d06 fix(refresh-credits): abort splice on jq failure (Codex follow-up)
4b89ad7 docs: add CREDITS.md (foundational sources, papers, contributors)
```

## Open PRs at draft time (for next release after this)

- #127 feat(gfx906): wave64 FP16 hybrid prefill kernels (HARDWARE-GATED)
- #125 fix(daemon): runtime n-gram loop detector (DRAFT)
- #124 fix(dflash): wire max_think_tokens through DFlash (DRAFT)
- #123 fix(thinking): hard-suppress thinking when thinking=off (DRAFT)
- #115 feat(mq-lloyd): Lloyd-Max codebooks for MQ3/MQ2 (DRAFT)

PRs #123/#124/#125 form a cohesive "kill runaway generation" series
addressing the documented Qwen3.6-A3B repetition-loop pathology;
they should likely sequence into the same release once the authors
mark them ready.

## Validation status at draft time

- Coherence gate: clean on the four canonical prompts (#128 and #129
  both passed via pre-commit + post-merge spot-check).
- Speed gate: not run as part of this draft; gfx1100 baselines unchanged
  by these commits (no kernel changes); gfx1151 baselines unchanged for
  the same reason.
- A/B regression: not applicable to these commits (no perf-affecting
  changes in the engine path).
