//! engine: GGUF model loading and LLaMA inference on RDNA GPUs.
//!
//! When built with `--features npu`, the engine also exposes
//! `engine::npu` — a wrapper over `hipx` that gates NPU offload
//! behind runtime capability detection (only Strix Halo has an NPU
//! today). NPU paths are always opportunistic; iGPU stays
//! correctness-load-bearing.

#[cfg(feature = "npu")]
pub mod npu;

pub mod gguf;
pub mod hfq;
pub mod llama;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
pub mod qwen35_vl;
#[cfg(feature = "deltanet")]
pub mod speculative;
#[cfg(feature = "deltanet")]
pub mod dflash;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod image;
pub mod tokenizer;
