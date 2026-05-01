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

/// Smoke test: runs the embedded passthrough_4k kernel end-to-end.
/// Allocates a 4 KiB input buffer, fills it with the user's bytes,
/// dispatches the CU, returns the output. Used to verify the engine
/// → hipx → kernel chain end-to-end on a real machine without
/// depending on a real LLM kernel being authored yet.
///
/// Returns `Err` if the dispatch path fails (driver dropped, fence
/// timeout, etc.) — caller can fall back to a CPU memcpy.
impl NpuRuntime {
    pub fn passthrough_4k(&mut self, input: &[u8; 4096]) -> Result<[u8; 4096], hipx::XdnaError> {
        use hipx::cmd::{config_cus, submit_exec_cmd};
        use hipx::ert::ErtBuilder;
        use hipx::fence::timeline_wait;
        use hipx::hwctx::HwctxBuilder;
        use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
        use hipx::kernels::{
            passthrough_4k_args as args, PASSTHROUGH_4K_COLUMNS,
            PASSTHROUGH_4K_INSTS, PASSTHROUGH_4K_OPS_PER_CYCLE, PASSTHROUGH_4K_PDI,
        };
        use std::time::Duration;

        if !self.available_ops.passthrough_4k {
            return Err(hipx::XdnaError {
                code: 0,
                message: "passthrough_4k kernel not available".into(),
            });
        }

        // hwctx
        let mut b = HwctxBuilder::default();
        b.num_columns = PASSTHROUGH_4K_COLUMNS;
        b.max_opc = PASSTHROUGH_4K_OPS_PER_CYCLE;
        let ctx = self.hipx.create_hwctx(&b)?;

        // PDI BO + bind CU
        let pdi_bo = self.hipx.alloc_dev(PASSTHROUGH_4K_PDI.len())?;
        unsafe {
            let buf = self.hipx.dev_slice(&pdi_bo)?;
            buf[..PASSTHROUGH_4K_PDI.len()].copy_from_slice(PASSTHROUGH_4K_PDI);
        }
        pdi_bo.sync(SYNC_TO_DEVICE)?;
        let _cu = config_cus(self.hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])?;

        // npu_insts BO
        let instr_bo = self.hipx.alloc_dev(PASSTHROUGH_4K_INSTS.len())?;
        unsafe {
            let buf = self.hipx.dev_slice(&instr_bo)?;
            buf[..PASSTHROUGH_4K_INSTS.len()].copy_from_slice(PASSTHROUGH_4K_INSTS);
        }
        let _ = instr_bo.sync(SYNC_TO_DEVICE);
        let ninstr_dwords = (PASSTHROUGH_4K_INSTS.len() / 4) as u32;

        // input
        let mut input_bo = self.hipx.alloc_shmem(4096)?;
        {
            let buf = input_bo.map()?;
            buf[..4096].copy_from_slice(input);
        }
        let _ = input_bo.sync(SYNC_TO_DEVICE);
        let input_va = input_bo.host_ptr().unwrap() as u64;

        // output
        let mut output_bo = self.hipx.alloc_shmem(4096)?;
        {
            let buf = output_bo.map()?;
            for b in buf[..4096].iter_mut() {
                *b = 0;
            }
        }
        let _ = output_bo.sync(SYNC_TO_DEVICE);
        let output_va = output_bo.host_ptr().unwrap() as u64;

        // cmd packet
        let mut cmd_bo = self.hipx.alloc_cmd(4096)?;
        {
            let cbuf = cmd_bo.map()?;
            let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
            eb.set_cu_mask(0x1);
            eb.set_arg_u64(args::OPCODE, 3);
            eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
            eb.set_arg_u32(args::NINSTR, ninstr_dwords);
            eb.set_arg_u64(args::BO0, input_va);
            eb.set_arg_u64(args::BO1, output_va);
            let _ = eb.finalize(0x3C);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);

        // submit
        let seq = submit_exec_cmd(
            self.hipx.device.fd,
            &ctx,
            &[&cmd_bo],
            &[&instr_bo, &input_bo, &output_bo],
        )?;

        // wait — same multi-point fallback as the standalone binary;
        // 100ms safety sleep for when timeline_wait EINVALs.
        for point in [seq, seq + 1, seq.saturating_add(2)] {
            if timeline_wait(
                self.hipx.device.fd,
                ctx.syncobj_handle,
                point,
                Duration::from_secs(5),
            )
            .is_ok()
            {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(100));

        let _ = output_bo.sync(SYNC_FROM_DEVICE);
        let outp = output_bo.map()?;
        let mut out = [0u8; 4096];
        out.copy_from_slice(&outp[..4096]);
        Ok(out)
    }
}
