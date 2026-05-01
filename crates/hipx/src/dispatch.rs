//! NPU dispatch predicates — analog of `rdna-compute::dispatch` for
//! the AIE-2P side. Engine code uses these to pick the right kernel /
//! tile shape / column count for the current SoC. **No per-arch forks
//! ever.** Add new SoCs here as predicate cases.

use crate::device::NpuInfo;

/// Coarse SoC family classification. Strix Halo (gfx1151 + aie2p) is
/// the only entry today; future Krackan / Strix Point / etc. land
/// here as new variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuFamily {
    /// AIE-2P / NPU5 — Strix Halo (Ryzen AI MAX+ 395). 8 cols × 4 core
    /// rows = 32 tiles, ~58 TOPS INT8 at 1.27 GHz.
    Aie2p,
    /// Anything not yet classified. Engine should refuse to dispatch
    /// to NPU when this is returned.
    Unknown,
}

/// Classify by the AIE major version returned from
/// `DRM_AMDXDNA_QUERY_AIE_VERSION`. This is the version of the AIE
/// architecture, not the firmware build.
pub fn classify(info: &NpuInfo) -> NpuFamily {
    match info.aie_version {
        // AIE-2P reports 1.x on the npu_7 firmware lineage we have on
        // hipx (verified 2026-05-01). Earlier AIE-2 / AIE-ML chips
        // report similar values; refining the family detection will
        // need a `pci_device_id`-based predicate added when we have
        // hardware to test.
        (1, _) => NpuFamily::Aie2p,
        _ => NpuFamily::Unknown,
    }
}

/// Whether the SoC has a usable NPU at all. Used by the engine to
/// gate any NPU codepath behind a runtime probe.
pub fn has_npu(info: &NpuInfo) -> bool {
    !matches!(classify(info), NpuFamily::Unknown)
}

/// Peak INT8 TOPS reported by firmware. Used to size workload routing
/// (e.g. "use NPU when GEMM is > N TOPS-equivalent").
pub fn npu_int8_tops(info: &NpuInfo) -> u64 {
    info.npu_tops_max
}

/// Total available column count in the AIE array.
pub fn cols_available(info: &NpuInfo) -> u16 {
    info.aie_cols
}

/// Core tile rows per column (the compute-tile rows; mem + shim rows
/// don't run kernels). Used to compute `num_tiles` for hwctx requests.
pub fn core_rows(info: &NpuInfo) -> u16 {
    info.core_tiles.1
}

/// Whether this NPU is on a UMA SoC where a dmabuf imported from
/// amdgpu (the iGPU's render node) can be ingested without a copy.
/// Strix Halo is the canonical case; this is the architectural moat
/// for the spec-decode draft offload thesis. Every NPU we can target
/// today is on a UMA SoC, so this is constant-true; left as a
/// predicate so future dGPU+NPU hardware can return false.
pub fn npu_uma_with_igpu(info: &NpuInfo) -> bool {
    has_npu(info)
}

/// Whether the NPU's INT8 throughput actually beats the iGPU's i8
/// WMMA path at the GEMM shapes hipfire's MMQ kernels run. Decided by
/// peak TOPS comparison; refine with measured numbers post-Phase 1.5.
pub fn npu_beats_igpu_int8(info: &NpuInfo, igpu_int8_tops_estimate: u64) -> bool {
    npu_int8_tops(info) > igpu_int8_tops_estimate
}

/// Per-op routing decision. Engine code calls `compute_target(op,
/// shape, runtime_caps)` to learn where to dispatch.
///
/// Returning `Igpu` is always safe (current default for everything).
/// Returning `Npu` requires the caller to have a working hipx runtime
/// AND an embedded PDI for that op. With no PDI, the dispatch should
/// fall back at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTarget {
    Igpu,
    Npu,
}

/// Op-class predicate. Per-op routing rules live here so engine call
/// sites stay agnostic. New op classes get a new variant + a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// Asym KV codec (asym2/3/4 dequant). The first NPU-resident op
    /// per the rising-tide plan — frees iGPU BW from codec work
    /// without contesting attention compute.
    KvCodec,
    /// INT8 GEMM matmul. Routed to NPU when batch is large enough
    /// for AIE-2P's compute throughput to win over BW costs.
    Int8Gemm { m: u32, n: u32, k: u32 },
    /// Embedding/classifier sidecar (BGE/E5-class small models).
    /// Always NPU when supported because iGPU BW is the constraint
    /// during simultaneous LLM inference.
    EmbeddingSidecar,
    /// Vision encoder (qwen35_vl, multimodal). NPU-friendly: conv-
    /// heavy, INT8-quant tolerant, fits in NPU workspace.
    VisionEncoder,
    /// Sampler (top-k/top-p, repetition penalty). Marginal; only
    /// route when NPU is otherwise idle.
    Sampler { vocab: u32 },
    /// Anything else — always iGPU.
    Other,
}

/// Engine-facing routing function. Pure of side effects; deterministic
/// given (op, info, npu_pdi_available). The "PDI available" flag tells
/// us whether we have a kernel artifact for this op — without one,
/// even if the hardware would be a good fit, we must fall back.
pub fn compute_target(op: OpClass, info: &NpuInfo, npu_pdi_available: bool) -> ComputeTarget {
    if !has_npu(info) || !npu_pdi_available {
        return ComputeTarget::Igpu;
    }
    match op {
        OpClass::KvCodec | OpClass::EmbeddingSidecar | OpClass::VisionEncoder => {
            ComputeTarget::Npu
        }
        OpClass::Int8Gemm { m, n, k } => {
            // GEMM beats iGPU only at compute-bound shapes. For each
            // 1 byte of weight read we want at least ~10 ops/cycle on
            // the NPU side. Threshold tuned against measured numbers
            // in Phase 1.5; conservative default until we have data.
            let arith_intensity = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
            if arith_intensity > 8 * 1024 * 1024 {
                ComputeTarget::Npu
            } else {
                ComputeTarget::Igpu
            }
        }
        OpClass::Sampler { vocab } => {
            // Sampler is usually <1% wall-clock. Only route to NPU
            // when vocab is large enough to amortize the dispatch
            // cost (~50us roundtrip).
            if vocab >= 65536 {
                ComputeTarget::Npu
            } else {
                ComputeTarget::Igpu
            }
        }
        OpClass::Other => ComputeTarget::Igpu,
    }
}
