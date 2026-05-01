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
    /// Lazily-initialized matvec kernel state — created on first call
    /// to `matvec_i16()` and reused across subsequent dispatches.
    /// Holds the bound CU, instruction stream, hwctx, and pre-built
    /// cmd packet so steady-state dispatch is just BO refill + submit.
    matvec_288: Option<Matvec288Kernel>,
}

/// Persistent state for the i16 288×288 matvec kernel. All BOs,
/// the hwctx, the bound CU, and the cmd packet are allocated once
/// and reused for every dispatch — the engine pays the ~1.5 ms
/// first-call setup once, then every subsequent call is the bare
/// dispatch latency (~530 µs steady-state at the time of writing).
struct Matvec288Kernel {
    ctx: hipx::hwctx::Hwctx,
    _cu: hipx::cmd::CuBinding,
    instr_bo: hipx::Bo,
    a_bo: hipx::Bo,
    b_bo: hipx::Bo,
    c_bo: hipx::Bo,
    bo3_bo: hipx::Bo,
    bo4_bo: hipx::Bo,
    cmd_bo: hipx::Bo,
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
            matvec_288: None,
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

        // wait — point=seq matches aie2_ctx.c's
        // drm_syncobj_add_point(syncobj, chain, fence, seq).
        timeline_wait(
            self.hipx.device.fd,
            ctx.syncobj_handle,
            seq,
            Duration::from_secs(5),
        )?;

        let _ = output_bo.sync(SYNC_FROM_DEVICE);
        let outp = output_bo.map()?;
        let mut out = [0u8; 4096];
        out.copy_from_slice(&outp[..4096]);
        Ok(out)
    }

    /// Split the matvec dispatch into submit + wait phases so the
    /// caller can run iGPU work concurrently with the NPU job.
    /// Returns the firmware sequence number to wait on.
    pub fn matvec_i16_288x288_submit(
        &mut self,
        a: &[i16],
        b: &[i16],
    ) -> Result<u64, hipx::XdnaError> {
        self.matvec_i16_288x288_inner(a, b, /*wait*/ false, &mut [])
            .map(|(seq, _)| seq)
    }

    /// Wait on a previously-submitted matvec sequence and copy the
    /// result into `c`. Pair with `matvec_i16_288x288_submit`.
    pub fn matvec_i16_288x288_wait(
        &mut self,
        seq: u64,
        c: &mut [i32],
    ) -> Result<(), hipx::XdnaError> {
        use hipx::fence::timeline_wait;
        use hipx::ioctl::SYNC_FROM_DEVICE;
        use std::time::Duration;

        let Some(kern) = self.matvec_288.as_mut() else {
            return Err(hipx::XdnaError {
                code: 0,
                message: "matvec kernel not initialized".into(),
            });
        };
        timeline_wait(
            self.hipx.device.fd,
            kern.ctx.syncobj_handle,
            seq,
            Duration::from_secs(5),
        )?;
        let _ = kern.c_bo.sync(SYNC_FROM_DEVICE);
        let outp = kern.c_bo.map()?;
        for (i, slot) in c.iter_mut().enumerate() {
            let bytes: [u8; 4] = outp[i * 4..i * 4 + 4].try_into().unwrap();
            *slot = i32::from_le_bytes(bytes);
        }
        Ok(())
    }

    /// 288×288 i16 → i32 GEMV. Lazily initializes the kernel on first
    /// call (allocates BOs, binds CU, builds cmd packet); subsequent
    /// calls just refill the input BOs and dispatch. `a` is M×K row-
    /// major i16, `b` is K i16; the result is M i32 written into `c`.
    ///
    /// Returns Err on dispatch failure — the caller should fall back
    /// to an iGPU/CPU GEMV path.
    pub fn matvec_i16_288x288(
        &mut self,
        a: &[i16],
        b: &[i16],
        c: &mut [i32],
    ) -> Result<(), hipx::XdnaError> {
        let (seq, _) = self.matvec_i16_288x288_inner(a, b, /*wait*/ true, c)?;
        let _ = seq;
        Ok(())
    }

    fn matvec_i16_288x288_inner(
        &mut self,
        a: &[i16],
        b: &[i16],
        wait: bool,
        c: &mut [i32],
    ) -> Result<(u64, ()), hipx::XdnaError> {
        use hipx::cmd::{config_cus, submit_exec_cmd};
        use hipx::ert::{reset_state, ErtBuilder};
        use hipx::fence::timeline_wait;
        use hipx::hwctx::HwctxBuilder;
        use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
        use hipx::kernels::{
            matvec_288x288_args as args, MATVEC_288X288_COLUMNS, MATVEC_288X288_INSTS,
            MATVEC_288X288_K, MATVEC_288X288_M, MATVEC_288X288_OPS_PER_CYCLE, MATVEC_288X288_PDI,
        };
        use std::time::Duration;

        let m = MATVEC_288X288_M;
        let k = MATVEC_288X288_K;
        if a.len() != m * k {
            return Err(hipx::XdnaError {
                code: 0,
                message: format!("matvec a.len={} != {}", a.len(), m * k),
            });
        }
        if b.len() != k {
            return Err(hipx::XdnaError {
                code: 0,
                message: format!("matvec b.len={} != {k}", b.len()),
            });
        }
        if wait && c.len() != m {
            return Err(hipx::XdnaError {
                code: 0,
                message: format!("matvec c.len={} != {m}", c.len()),
            });
        }

        // Lazy first-call init.
        if self.matvec_288.is_none() {
            let mut hb = HwctxBuilder::default();
            hb.num_columns = MATVEC_288X288_COLUMNS;
            hb.max_opc = MATVEC_288X288_OPS_PER_CYCLE;
            let ctx = self.hipx.create_hwctx(&hb)?;

            let pdi_bo = self.hipx.alloc_dev(MATVEC_288X288_PDI.len())?;
            unsafe {
                let buf = self.hipx.dev_slice(&pdi_bo)?;
                buf[..MATVEC_288X288_PDI.len()].copy_from_slice(MATVEC_288X288_PDI);
            }
            let _ = pdi_bo.sync(SYNC_TO_DEVICE);
            let cu = config_cus(self.hipx.device.fd, &ctx, vec![pdi_bo], &[0u8])?;

            let _pad = self.hipx.alloc_dev(32 * 1024)?;
            std::mem::forget(_pad);

            let instr_bo = self.hipx.alloc_dev(MATVEC_288X288_INSTS.len())?;
            unsafe {
                let buf = self.hipx.dev_slice(&instr_bo)?;
                buf[..MATVEC_288X288_INSTS.len()].copy_from_slice(MATVEC_288X288_INSTS);
            }
            let _ = instr_bo.sync(SYNC_TO_DEVICE);
            let ninstr_dwords = (MATVEC_288X288_INSTS.len() / 4) as u32;

            let mut a_bo = self.hipx.alloc_shmem(m * k * 2)?;
            let mut b_bo = self.hipx.alloc_shmem(k * 2)?;
            let mut c_bo = self.hipx.alloc_shmem(m * 4)?;
            let mut bo3_bo = hipx::Bo::alloc_shmem_exact(self.hipx.device.fd, 8)?;
            let mut bo4_bo = hipx::Bo::alloc_shmem_exact(self.hipx.device.fd, 1)?;
            // Populate host_ptr by mapping each BO once during init.
            // Subsequent map() calls just return the cached address.
            let _ = a_bo.map()?;
            let _ = b_bo.map()?;
            let _ = c_bo.map()?;
            let _ = bo3_bo.map()?;
            let _ = bo4_bo.map()?;

            let a_va = a_bo.host_ptr().unwrap() as u64;
            let b_va = b_bo.host_ptr().unwrap() as u64;
            let c_va = c_bo.host_ptr().unwrap() as u64;
            let bo3_va = bo3_bo.host_ptr().unwrap() as u64;
            let bo4_va = bo4_bo.host_ptr().unwrap() as u64;

            let mut cmd_bo = self.hipx.alloc_cmd(4096)?;
            {
                let cbuf = cmd_bo.map()?;
                let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
                eb.set_cu_mask(0x1);
                eb.set_arg_u64(args::OPCODE, 3);
                eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
                eb.set_arg_u32(args::NINSTR, ninstr_dwords);
                eb.set_arg_u64(args::A, a_va);
                eb.set_arg_u64(args::B, b_va);
                eb.set_arg_u64(args::C, c_va);
                eb.set_arg_u64(args::BO3, bo3_va);
                eb.set_arg_u64(args::BO4, bo4_va);
                let _ = eb.finalize(0x3C);
            }
            let _ = cmd_bo.sync(SYNC_TO_DEVICE);

            self.matvec_288 = Some(Matvec288Kernel {
                ctx,
                _cu: cu,
                instr_bo,
                a_bo,
                b_bo,
                c_bo,
                bo3_bo,
                bo4_bo,
                cmd_bo,
            });
            self.available_ops.int8_gemm = true; // first GEMM-class kernel live
        }

        let kern = self.matvec_288.as_mut().unwrap();

        // Refill A, B, reset C sentinel
        {
            let abuf = kern.a_bo.map()?;
            for (i, &v) in a.iter().enumerate() {
                abuf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        let _ = kern.a_bo.sync(SYNC_TO_DEVICE);
        {
            let bbuf = kern.b_bo.map()?;
            for (i, &v) in b.iter().enumerate() {
                bbuf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        let _ = kern.b_bo.sync(SYNC_TO_DEVICE);
        {
            let cbuf = kern.c_bo.map()?;
            for byte in cbuf[..m * 4].iter_mut() {
                *byte = 0;
            }
        }
        let _ = kern.c_bo.sync(SYNC_TO_DEVICE);

        // reset cmd packet state to NEW (firmware skips packets still
        // marked COMPLETED from prior dispatch)
        {
            let cbuf = kern.cmd_bo.map()?;
            reset_state(&mut cbuf[..4]);
        }
        let _ = kern.cmd_bo.sync(SYNC_TO_DEVICE);

        let seq = submit_exec_cmd(
            self.hipx.device.fd,
            &kern.ctx,
            &[&kern.cmd_bo],
            &[
                &kern.instr_bo,
                &kern.a_bo,
                &kern.b_bo,
                &kern.c_bo,
                &kern.bo3_bo,
                &kern.bo4_bo,
            ],
        )?;
        if !wait {
            return Ok((seq, ()));
        }
        timeline_wait(
            self.hipx.device.fd,
            kern.ctx.syncobj_handle,
            seq,
            Duration::from_secs(5),
        )?;

        let _ = kern.c_bo.sync(SYNC_FROM_DEVICE);
        let outp = kern.c_bo.map()?;
        for (i, slot) in c.iter_mut().enumerate() {
            let bytes: [u8; 4] = outp[i * 4..i * 4 + 4].try_into().unwrap();
            *slot = i32::from_le_bytes(bytes);
        }
        Ok((seq, ()))
    }
}
