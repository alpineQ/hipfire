# Movie-Night Managerial Session — 2026-05-02

## PRs reviewed

| PR | Title | Verdict | SHA / Link |
|----|-------|---------|------------|
| 128 | pread weight loading (Strix Halo OOM fix) | MERGED | 36ca314 |
| 129 | hipfire chat TUI | MERGED | b9d1016 |
| 127 | gfx906 wave64 FP16 hybrid | DEFERRED (HARDWARE-GATED) | https://github.com/Kaden-Schutt/hipfire/pull/127 |

## PRs merged

- #128 admin-merged (closed before session, recap above): pread + FADV_DONTNEED
  load path for unified-memory APUs. A/B on hipx 9B (4.95 GiB) load: master path
  evicted 137 MB of existing cache; PR path grew cache by 932 KB. Identical gen
  output (46.7-46.8 tok/s). Mergebase 36ca314.
- #129 admin-merged this session: hipfire chat TUI. 103 unit tests pass, both
  bundles build clean, dispatch + non-TTY guard work. Diff to existing code is
  4 fn exports + one PID-file gate (default-off) + one new dispatch case. Risk
  surface bounded to the new `chat` subcommand. Mergebase b9d1016.

## PRs escalated to MANUAL_REVIEW

- #127 gfx906 wave64 FP16 hybrid prefill: HARDWARE-GATED. Code reads cleanly,
  predicate gate is correctly scoped, kernel pattern is standard. Concerns
  documented in MANUAL_REVIEW.md. No merge attempted.

## Anomalies / observations

- (placeholder; will fill as session progresses)
