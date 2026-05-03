# Movie-Night Managerial Session — 2026-05-02

PR triage and managerial work while Kaden was at the movies. Session
ran on the `strix-halo` worktree, post-rebase onto origin/master.

## PRs reviewed

| PR | Title | Verdict | Mergebase / Link |
|----|-------|---------|------------------|
| 128 | pread weight loading (Strix Halo OOM fix) | MERGED | 36ca314 |
| 129 | hipfire chat TUI | MERGED | b9d1016 |
| 127 | gfx906 wave64 FP16 hybrid | DEFERRED (HARDWARE-GATED) | https://github.com/Kaden-Schutt/hipfire/pull/127 |
| 123 | hard-suppress thinking when thinking=off | DRAFT, REVIEW-NEEDED | https://github.com/Kaden-Schutt/hipfire/pull/123 |
| 124 | wire max_think_tokens through DFlash | DRAFT, REVIEW-NEEDED | https://github.com/Kaden-Schutt/hipfire/pull/124 |
| 125 | runtime n-gram loop detector | DRAFT, REVIEW-NEEDED | https://github.com/Kaden-Schutt/hipfire/pull/125 |
| 115 | Lloyd-Max codebooks for MQ3/MQ2 | SCOPE-OR-POLICY (Kaden's draft) | https://github.com/Kaden-Schutt/hipfire/pull/115 |

## PRs merged

- **#128 (admin)** during prior turn before this contract started, but
  recap: pread + FADV_DONTNEED load path for unified-memory APUs. A/B
  on hipx 9B (4.95 GiB) load: master path evicted 137 MB of existing
  cache; PR path grew cache by 932 KB. Identical generation output
  (46.7-46.8 tok/s). Mergebase 36ca314.
- **#129 (admin)** during this session: hipfire chat TUI. 103 unit
  tests pass (PR body said 57; suite has grown). Both bundles build
  clean with bun target. Diff to existing code is 4 fn exports + one
  PID-file gate (default-off) + one new dispatch case. Risk surface
  bounded to the new `chat` subcommand. Mergebase b9d1016.

Both admin merges had concrete justifications recorded in the merge
commit body (per the contract's "admin merge requires concrete
justification" rule).

## PRs reviewed read-only

- **#127** gfx906 wave64 FP16 hybrid prefill kernels. Substantive
  technical review posted. Concerns documented: gfx908 included in
  gate but no perf data; no logit-divergence check beyond coherence
  battery; no env-level escape hatch; BATCH_TILE=8 hardcoded.
  Recommendation: Kaden validate on V620/MI50 hardware before merge.
- **#123** thinking=off hard-suppress. Approach matches the project's
  prior anti-pattern lesson (cap-at-daemon, NOT directive injection;
  see existing memory). Approve-in-spirit pending Kaden's review.
- **#124** DFlash think cap. Approach sound; flagged O(N^2) decode
  cost as a perf concern; no regression test added.
- **#125** n-gram loop detector. Approach sound; flagged
  HashMap-rebuild-per-token as O(window) when O(1) is achievable;
  recommended code-completion smoke test before merge.

## PRs escalated to MANUAL_REVIEW

All non-merged PRs above are documented in MANUAL_REVIEW.md with
classification, recommendation, what was tested, what was reviewed,
and concerns.

## Issues triaged

Scanned `gh issue list --state open --limit 50`. The Strix-Halo and
gfx1151-relevant ones are actively being worked by contributors with
the right hardware:
- #87 auto-MMQ regression: nwoolmer is running MMQ-screen logs against
  recent master commits.
- #119 ROCm 7.2 / clang 22 compiler regression: fix is on
  `origin/strix-halo` per Kaden's existing comment (and the rebased
  state I just pushed). Master does not yet carry the fix
  (compiler.rs grep on master shows no `__clang_hip_runtime_wrapper`
  -include and no max/min device override). Whether to merge to
  master is Kaden's call.
- #105 CPU+GPU split: Kaden's last comment routes to #77 / #76. Done.

No issues identified that I could productively reproduce + comment on
without redundant noise.

## Anomalies / observations

1. **No CI configured.** `.github/workflows/` does not exist on
   master. `gh run list` returns empty. PR `statusCheckRollup` is
   `[]` for all PRs in the queue. Branch protection still requires
   review (so admin merge is needed for solo-merge), but there is
   no automated build/test gate. The `coherence-gate.sh` and
   `speed-gate.sh` run only as pre-commit hooks on the developer
   side.
2. **`origin/strix-halo` carries fixes not yet on master**, including
   the ROCm 7.2.x compiler workarounds (b1ab41f, aabd535, 850c69b
   on the rebased branch) that resolve #119 for the Strix Halo
   reporter. Whether these should fast-forward to master is a
   directional call.
3. **Stale branch audit clean.** Oldest non-master remote branch is
   `origin/gemma4` at 3 weeks. Nothing meets the >30-days threshold.
   `origin/fix/111-tool-call-attractor-block` (17h, 2 commits ahead
   of master) appears to predate the squash-merge of PR #121 at
   3f39668 and could be deleted by Kaden, but is not stale enough
   to flag.

## Telemetry

- `MANUAL_REVIEW.md` populated (5 PRs).
- `docs/release-notes-2026-05-02.md` drafts the next release
  candidate (post v0.1.9-alpha.1, suggested bump v0.1.9-alpha.2).
- This file.

## Branch

`session/strix-halo-2026-05-02` carries the telemetry. Master is
unchanged by this session except for the two merges above (which
went through `gh pr merge` and were force-pulled to `origin/master`
by GitHub).
