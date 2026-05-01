//! Strix Halo NPU offload — engine-side wrapper around `hipx`.
//!
//! Compiled only when the `npu` feature is enabled. Providing a single
//! place where the engine asks "is the NPU available, and which ops can
//! I send to it?" lets every call site stay agnostic about what
//! hardware is present at runtime.
//!
//! # Lifecycle
//!
//! ```ignore
//! let npu = engine::npu::NpuRuntime::try_init();
//! match &npu {
//!     Some(rt) => println!("hipfire-x active: {} TOPS INT8", rt.tops_int8()),
//!     None => println!("hipfire-x: no compatible NPU on this system"),
//! }
//!
//! // Per-op: ask `route()` which engine should run this op. Returns
//! // ComputeTarget::Npu only when we have a kernel for that op AND
//! // shape AND the runtime is initialized.
//! let target = engine::npu::route(&npu, op, shape);
//! ```
//!
//! # Engine-call-site semantics
//!
//! The engine calls into NpuRuntime::run_<op>() once a route has been
//! chosen. If the kernel fails (driver dropped, fence timeout), the
//! method returns `Err`, and the call site falls back to the iGPU
//! path. This makes NPU offload an *opportunistic* speedup, never a
//! correctness-load-bearing path.

use hipx::dispatch::{
    classify, compute_target, has_npu, npu_int8_tops, ComputeTarget, NpuFamily, OpClass,
};
use hipx::Hipx;

/// One-per-process NPU runtime. Engine code holds this behind an
/// `Option<NpuRuntime>` so non-Strix-Halo machines see `None`.
pub struct NpuRuntime {
    hipx: Hipx,
    family: NpuFamily,
    /// Set of op classes for which we have a working PDI loaded. Used
    /// by `route()` to gate dispatch — a route to NPU is meaningless
    /// without a kernel for that op.
    available_ops: AvailableOps,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct AvailableOps {
    /// 4 KiB byte passthrough (proved end-to-end via hipx-passthrough).
    /// Useful as a smoke test; not a real engine op.
    pub passthrough_4k: bool,
    /// KV-codec asym dequant (asym2/3/4 → fp16). The actual workload-
    /// moving kernel; lights up when MLIR-AIE source for it is built.
    pub kv_dequant: bool,
    /// INT8 GEMV / GEMM for spec-decode draft. Future.
    pub int8_gemm: bool,
}

impl NpuRuntime {
    /// Best-effort init. Returns `None` on:
    /// - non-Strix-Halo systems (no `/dev/accel/accel0`)
    /// - missing OOT amdxdna driver
    /// - `RLIMIT_MEMLOCK` too low (heap mmap fails)
    /// - any other ABI / setup mismatch
    ///
    /// Safe to call once at engine init; subsequent dispatch decisions
    /// just check `Option::is_some()`.
    pub fn try_init() -> Option<Self> {
        let hipx = match Hipx::open() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[hipfire-x] init skipped: {e}");
                return None;
            }
        };
        let family = classify(&hipx.info);
        if !has_npu(&hipx.info) {
            return None;
        }
        // Kernel availability is compile-time today (we embed PDI bytes
        // in hipx::kernels). Phase 2 will swap this for runtime probes
        // when we add dynamic PDI loading.
        let available_ops = AvailableOps {
            passthrough_4k: true,
            kv_dequant: false,
            int8_gemm: false,
        };
        Some(Self {
            hipx,
            family,
            available_ops,
        })
    }

    pub fn family(&self) -> NpuFamily {
        self.family
    }

    pub fn tops_int8(&self) -> u64 {
        npu_int8_tops(&self.hipx.info)
    }

    pub fn cols(&self) -> u16 {
        self.hipx.info.aie_cols
    }

    pub fn available_ops(&self) -> &AvailableOps {
        &self.available_ops
    }
}

/// Per-op routing decision. Pure of side effects; only depends on
/// the runtime's reported capabilities and the op shape.
pub fn route(npu: &Option<NpuRuntime>, op: OpClass) -> ComputeTarget {
    let Some(rt) = npu else {
        return ComputeTarget::Igpu;
    };
    let pdi_available = match op {
        OpClass::KvCodec => rt.available_ops.kv_dequant,
        OpClass::Int8Gemm { .. } => rt.available_ops.int8_gemm,
        // Other op classes have no NPU kernel yet.
        OpClass::EmbeddingSidecar | OpClass::VisionEncoder | OpClass::Sampler { .. } => false,
        OpClass::Other => false,
    };
    compute_target(op, &rt.hipx.info, pdi_available)
}
