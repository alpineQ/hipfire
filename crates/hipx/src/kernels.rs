//! Kernel artifact registry. Analog of `rdna-compute::kernels` for
//! the NPU side.
//!
//! On the iGPU side, hipfire ships a `kernels/*.hip` directory with
//! HIP source, compiles to `.hsaco` ELF at build time, and embeds the
//! ELF bytes as `pub const FOO_SRC: &[u8] = include_bytes!(...)`. The
//! engine then uploads + dispatches at runtime.
//!
//! On the NPU side, the artifact format is **PDI** (Platform Device
//! Image) — a binary blob produced by an AIE compiler (MLIR-AIE +
//! Peano). Same shape as `.hsaco`: produced offline, embedded at
//! build time, sent to the firmware as a CU configuration BO.
//!
//! Today this module is a stub — no kernels embedded yet. Phase 1.3
//! lands the first PDI (likely a passthrough/memcpy CU as the
//! syncobj round-trip test). Phase 1.5 lands an INT8 GEMV.
//!
//! ```text
//! kernels/aie2p/
//! ├── README.md            — toolchain notes
//! ├── passthrough.mlir     — IR source
//! └── ...
//! kernels/aie2p/build/
//! ├── passthrough.pdi      — built artifact (embedded via include_bytes!)
//! └── ...
//! ```

/// Marker — present so the module compiles and has somewhere for
/// future `pub const FOO_PDI: &[u8] = include_bytes!(...)` lines.
pub const _PLACEHOLDER: &[u8] = &[];
