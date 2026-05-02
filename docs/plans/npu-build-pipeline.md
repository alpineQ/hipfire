# NPU kernel build pipeline (hand-rolled, no Python)

This doc captures the cross-cutting decision (per the npu-roadmap
contract) on how kernel rebuilds tie into the cargo workflow.

## Status quo (commit `937cb3e`)

- Per-kernel directory at `kernels/aie2p/<name>/` with three sources:
  - `aie.mlir` — hand-written MLIR.
  - `<name>_kernel.cc` — C++ kernel using `aie_api`.
  - `build.sh` — shell driver invoking the native `aiecc` and Peano `clang++`.
- Output artifacts in `kernels/aie2p/<name>/build/`:
  - `main.pdi` — firmware PDI.
  - `insts.bin` — NPU instruction stream.
  - `<name>.xclbin` — wrapper format (zip-like, but XRT-specific).
- Embedding into `crates/hipx/src/kernels.rs` is via `include_bytes!`
  pointing at `build/main.pdi` and `build/insts.bin` directly.

## Decision: stay with `bash build.sh` for now; do NOT wire `build.rs`

Reasoning:

1. **Build environment is hipx-only.** The kernels build only on the
   Strix Halo dev box where mlir-aie + Peano are installed at
   `/home/kaden/mlir-aie/...`. Forcing `cargo build` on every other
   machine to attempt this would fail loudly. A `build.rs` would have
   to feature-gate or path-detect, which is fragile.

2. **Rebuild cost is small but nontrivial.** Each kernel rebuild is
   ~5-15 seconds of `clang++` + `aiecc`. Cargo dirty-detection on
   `kernels/aie2p/.../*.cc` and `*.mlir` would have to integrate with
   `build.rs` and shell out, which trades clarity for sub-minute saves.
   Manual `bash build.sh` after kernel edits is fine.

3. **The committed PDI is the source of truth for verification.** Once
   stage 1.2 lands and the PDI is `include_bytes!`'d, any change to
   the kernel sources requires a deliberate rebuild + commit of the
   binary artifact. That's a desirable friction; nobody should change
   kernels casually.

## Planned conventions

- **One rebuild script per kernel** (`build.sh` in the kernel dir).
  Pattern set by `kernels/aie2p/asym3_dequant_256/build.sh`.

- **One driver script for "rebuild all"** at
  `kernels/aie2p/build_all.sh` (TODO; populated as more kernels move
  off the legacy IRON-Python path).

- **Manual commit gate.** PDI + insts.bin live under the kernel's
  `build/` dir, gitignored. To embed via `include_bytes!`, copy or
  symlink them to a stable path like `kernels/aie2p/<name>/asym3_dequant_256.pdi`
  that is committed. This separates the rebuild artifacts from the
  shipped binary blobs and keeps the diff legible (the commit shows a
  single PDI byte change, not the whole `build/` tree).

- **Future optimisation**: if rebuild churn becomes a daily problem
  for any kernel, revisit `build.rs` with a `cfg!(target_os = "linux")
  && Path::new("/opt/mlir-aie").exists()` guard. Not today.

## Rebuild idempotency check

The native `aiecc` is deterministic: same MLIR + same kernel `.o`
produces a byte-identical PDI. If a rebuild produces a different PDI,
something in the toolchain has changed; treat that as a separate
investigation, not an expected rebuild.

## What the contract says, vs what we did

The contract calls this a "ship it, refine later" decision. We chose
"ship it without `build.rs`". The escalation trigger ("if build
automation gets gnarly") fired immediately on consideration, and we
stopped before adding complexity. Re-evaluate when the kernel count
hits 5+ or rebuild churn becomes a measured problem.
