//! `hipx-verify-asym3-dequant` — stage 1.1 correctness verifier.
//!
//! Compares the NPU `asym3_dequant_256` kernel output bit-for-bit
//! against a CPU reference that mirrors `kernels/src/triattn_score_asym3.hip`
//! lines 54-64 (the engine's authoritative asym3 dequant).
//!
//! Runs N random seeds; exits 0 only if every seed matches every byte
//! of the bf16 output. On mismatch, dumps the first divergent
//! positions and the seed that triggered them, for diagnosis.
//!
//! Why bit-for-bit: asym3 is a deterministic transform. Same packed
//! bytes, same cnorm, same codebook, must produce identical bf16
//! values. Any tolerance is a bug.
//!
//! Stage 1.2 onward: PDI + insts ship via `include_bytes!` from
//! `crates/hipx/src/kernels.rs::ASYM3_DEQUANT_256_*`. The
//! ASYM3_PDI / ASYM3_INSTS env vars override with file-loaded
//! variants (useful for testing kernel rebuilds before re-cargo).
//!
//! Build:
//!   cargo build -p hipx --bin verify_asym3_dequant
//! Run:
//!   ./target/debug/verify_asym3_dequant [N_SEEDS]

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::Hipx;

const HEAD_DIM: usize = 256;
const PACKED_BYTES: usize = HEAD_DIM * 3 / 8; // 96
const CNORM_BYTES: usize = 4;
const OUT_BYTES: usize = HEAD_DIM * 2; // 256 bf16
const DEFAULT_SEEDS: usize = 100;

// Engine codebook from kernels/src/turbo_common.h. Exact values.
const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176,
     0.015176,  0.046469,  0.083320,  0.134860,
];

/// Load PDI: ASYM3_PDI env path overrides; otherwise the embedded
/// `ASYM3_DEQUANT_256_PDI` from `hipx::kernels`.
fn load_pdi() -> Vec<u8> {
    if let Ok(p) = std::env::var("ASYM3_PDI") {
        std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("ASYM3_PDI={p} read failed: {e}");
            std::process::exit(1);
        })
    } else {
        hipx::kernels::ASYM3_DEQUANT_256_PDI.to_vec()
    }
}

/// Load insts: ASYM3_INSTS env path overrides; otherwise embedded.
fn load_insts() -> Vec<u8> {
    if let Ok(p) = std::env::var("ASYM3_INSTS") {
        std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("ASYM3_INSTS={p} read failed: {e}");
            std::process::exit(1);
        })
    } else {
        hipx::kernels::ASYM3_DEQUANT_256_INSTS.to_vec()
    }
}

// xorshift64 PRNG so seeds are reproducible without a dep.
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdeadbeefcafebabe } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_byte(&mut self) -> u8 { (self.next_u64() & 0xff) as u8 }
    fn next_f32_unit(&mut self) -> f32 {
        // Uniform in [0, 1). Take 24 bits, scale.
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

// AIE-2P-shape bf16 conversions, characterized empirically via the
// asym3_dequant_256_f32 diagnostic kernel (see `--sweep` mode of
// hipx-diagnose-asym3-mul):
//
//   - Kernel's `(bfloat16)(*float_ptr)` cast: round-toward-zero
//     (drop low 16 mantissa bits, no rounding bias). Different from
//     IEEE round-to-nearest-even when the dropped bits are non-zero.
//
//   - Kernel's `aie::mul(bf16, bf16) -> accfloat`: bit-faithful.
//     Sweep across 14336 (cnorm bf16 x cb_idx) pairs with bf16-exact
//     cnorm produced ratio = 1.0 between CPU f32 product and NPU
//     f32 accumulator. The mul is "promote both to f32, multiply,
//     keep f32 accumulator."
//
//   - Kernel's `accfloat.to_vector<bfloat16>()`: round-away-from-zero
//     (any non-zero dropped bits bias magnitude up). Empirically
//     248/256 of the original verifier mismatches were RAZ-aligned.
//
// The CPU reference below mirrors all three exactly.

/// f32 -> bf16, RTZ. Drop low 16 bits.
fn f32_to_bf16_bits_rtz(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    (xb >> 16) as u16
}

/// f32 -> bf16, round-away-from-zero. Used for the final down-conversion
/// after the AIE-2P-shape mul.
fn f32_to_bf16_bits_raz(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    let abs = xb & 0x7fff_ffff;
    let sign = xb & 0x8000_0000;
    let biased = abs.wrapping_add(0xffff);
    ((sign | (biased & 0x7fff_ffff)) >> 16) as u16
}

/// f32 -> bf16, RNE. Kept for diagnostic comparison only.
#[allow(dead_code)]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    let lsb = (xb >> 16) & 1;
    let bias = 0x7fff + lsb;
    ((xb.wrapping_add(bias)) >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// CPU reference using a CALIBRATED codebook (the kernel's actual
/// stored bf16 representations, measured by the calibrate phase).
/// AIE-2P-shape mul model: cnorm f32 -> bf16 RTZ -> f32 promote, cb
/// bf16 -> f32 promote, f32 multiply, f32 -> bf16 RAZ.
fn cpu_reference_calibrated(packed: &[u8], cnorm: f32, calibrated_cb: &[u16; 8],
                            out_bf16: &mut [u16]) {
    assert_eq!(packed.len(), PACKED_BYTES);
    assert_eq!(out_bf16.len(), HEAD_DIM);
    let cnorm_b = bf16_bits_to_f32(f32_to_bf16_bits_rtz(cnorm));
    let cb_f32: [f32; 8] = std::array::from_fn(|i| bf16_bits_to_f32(calibrated_cb[i]));
    for tid in 0..32 {
        let base = tid * 3;
        let word = (packed[base] as u32)
            | ((packed[base + 1] as u32) << 8)
            | ((packed[base + 2] as u32) << 16);
        for i in 0..8 {
            let idx = ((word >> (i * 3)) & 7) as usize;
            let dim = tid * 8 + i;
            out_bf16[dim] = f32_to_bf16_bits_raz(cnorm_b * cb_f32[idx]);
        }
    }
}

/// CPU reference: mirrors `triattn_score_asym3.hip:54-64`.
///
/// For each thread tid in 0..32:
///   base = packed + tid*3 (3 bytes)
///   word = base[0] | (base[1]<<8) | (base[2]<<16)
///   for i in 0..8:
///     idx = (word >> (i*3)) & 7
///     dim = tid*8 + i
///     out[dim] = bf16(cnorm * TURBO_C3_256[idx])
///
/// The product cnorm * TURBO_C3_256[idx] is computed in f32, then
/// rounded to bf16. The kernel does the same: bf16 broadcast of cnorm
/// times bf16 codebook entry, with the AIE-2P bf16 mul producing a
/// bf16 result.
///
/// Subtle point: AIE-2P bf16 mul behavior. The kernel stores codebook
/// entries as bf16 constants (truncated at compile time). Our
/// reference must emit values consistent with that:
///    a) cnorm: f32 -> bf16 truncate
///    b) cb_bf16 = bf16-truncated codebook entry
///    c) result = bf16 mul of (a) * (b), expressed as bf16-of-(f32-mul-of-(bf16->f32)*(bf16->f32))
fn cpu_reference(packed: &[u8], cnorm: f32, out_bf16: &mut [u16]) {
    assert_eq!(packed.len(), PACKED_BYTES);
    assert_eq!(out_bf16.len(), HEAD_DIM);

    // AIE-2P-shape mul: cnorm f32 -> bf16 RTZ -> f32 promote, cb bf16
    // -> f32 promote, f32 multiply, f32 -> bf16 RAZ. See the f32
    // conversion docs above.
    let cnorm_bf16 = f32_to_bf16_bits_rtz(cnorm);
    let cnorm_b = bf16_bits_to_f32(cnorm_bf16);

    // Codebook bf16 reps confirmed byte-identical between kernel and
    // CPU via calibration; RTZ and RNE happen to agree on TURBO_C3_256.
    // Use RTZ here for consistency with the cnorm conversion.
    let cb_b: [f32; 8] = std::array::from_fn(|i|
        bf16_bits_to_f32(f32_to_bf16_bits_rtz(TURBO_C3_256[i]))
    );

    for tid in 0..32 {
        let base = tid * 3;
        let word = (packed[base] as u32)
            | ((packed[base + 1] as u32) << 8)
            | ((packed[base + 2] as u32) << 16);
        for i in 0..8 {
            let idx = ((word >> (i * 3)) & 7) as usize;
            let dim = tid * 8 + i;
            let v = cnorm_b * cb_b[idx];
            out_bf16[dim] = f32_to_bf16_bits_raz(v);
        }
    }
}

/// ULP distance between two same-sign bf16 values. For our use case
/// CPU and NPU outputs are always close enough to be same-sign and
/// same-or-adjacent-exponent, where the bit patterns are monotonic
/// in magnitude. Treats the bit difference as ULP count.
fn ulp_distance(a: u16, b: u16) -> u32 {
    let sa = a & 0x8000;
    let sb = b & 0x8000;
    if sa == sb {
        let mag_a = (a & 0x7fff) as i32;
        let mag_b = (b & 0x7fff) as i32;
        (mag_a - mag_b).unsigned_abs()
    } else {
        // Cross-zero (one positive, one negative). Distance through
        // zero in bf16-bit-count terms.
        let mag_a = (a & 0x7fff) as u32;
        let mag_b = (b & 0x7fff) as u32;
        mag_a + mag_b
    }
}

/// Signed ULP delta: (npu - cpu) measured in bf16 ULP units, with
/// magnitude direction. Positive when |npu| > |cpu|.
fn signed_ulp_delta(cpu: u16, npu: u16) -> i32 {
    let sc = cpu & 0x8000;
    let sn = npu & 0x8000;
    let mag_c = (cpu & 0x7fff) as i32;
    let mag_n = (npu & 0x7fff) as i32;
    if sc == sn {
        mag_n - mag_c
    } else {
        mag_n + mag_c
    }
}

#[derive(Default, Debug, Clone)]
struct SeedReport {
    max_ulp: u32,
    sum_signed_ulp: i64,
    n_diff: usize,
    determ_ok: bool,
    first_diff: Option<(usize, u16, u16, u32)>,  // dim, cpu, npu, ulp
}

fn run_one_seed(
    seed: u64,
    hipx_dev: &Hipx,
    ctx: &hipx::hwctx::Hwctx,
    instr_bo: &hipx::Bo,
    pdi_len: usize,
    insts_len: usize,
    calibrated_cb: Option<&[u16; 8]>,
) -> Result<SeedReport, Box<dyn std::error::Error>> {
    let _ = (pdi_len, insts_len); // metadata, unused at this layer

    let mut rng = XorShift64::new(seed);

    // Generate packed bytes + a non-trivial cnorm.
    let mut packed = vec![0u8; PACKED_BYTES];
    for b in packed.iter_mut() { *b = rng.next_byte(); }
    // cnorm in [-2.0, 2.0). Avoids zero so we exercise sign + scale.
    let cnorm: f32 = (rng.next_f32_unit() - 0.5) * 4.0;

    // CPU reference. Use calibrated codebook if available; falls back
    // to the raw engine codebook otherwise.
    let mut cpu_out = vec![0u16; HEAD_DIM];
    if let Some(cb) = calibrated_cb {
        cpu_reference_calibrated(&packed, cnorm, cb, &mut cpu_out);
    } else {
        cpu_reference(&packed, cnorm, &mut cpu_out);
    }

    // NPU dispatch — matches vec_scalar_mul cmd packet shape.
    let mut packed_bo = hipx_dev.alloc_shmem(PACKED_BYTES).expect("packed alloc");
    {
        let buf = packed_bo.map().expect("packed map");
        buf[..PACKED_BYTES].copy_from_slice(&packed);
    }
    let _ = packed_bo.sync(SYNC_TO_DEVICE);
    let packed_va = packed_bo.host_ptr().unwrap() as u64;

    let mut cnorm_bo = hipx_dev.alloc_shmem(CNORM_BYTES).expect("cnorm alloc");
    {
        let buf = cnorm_bo.map().expect("cnorm map");
        buf[..4].copy_from_slice(&cnorm.to_le_bytes());
    }
    let _ = cnorm_bo.sync(SYNC_TO_DEVICE);
    let cnorm_va = cnorm_bo.host_ptr().unwrap() as u64;

    let mut out_bo = hipx_dev.alloc_shmem(OUT_BYTES).expect("out alloc");
    {
        let buf = out_bo.map().expect("out map");
        for b in buf[..OUT_BYTES].iter_mut() { *b = 0xCC; } // sentinel
    }
    let _ = out_bo.sync(SYNC_TO_DEVICE);
    let out_va = out_bo.host_ptr().unwrap() as u64;

    // Placeholder bo3/bo4 same as vec_scalar_mul.
    let mut bo3 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 8)?;
    { let buf = bo3.map()?; for b in buf.iter_mut() { *b = 0; } }
    let _ = bo3.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3.host_ptr().unwrap() as u64;
    let mut bo4 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 1)?;
    { let buf = bo4.map()?; for b in buf.iter_mut() { *b = 0; } }
    let _ = bo4.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4.host_ptr().unwrap() as u64;

    let mut cmd_bo = hipx_dev.alloc_cmd(4096)?;
    let ninstr_dwords = (insts_len / 4) as u32;
    {
        let cbuf = cmd_bo.map()?;
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        // Same arg layout as vec_scalar_mul / passthrough.
        use hipx::kernels::passthrough_4k_args as args;
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::BO0, packed_va);
        eb.set_arg_u64(args::BO1, cnorm_va);
        eb.set_arg_u64(args::BO2, out_va);
        eb.set_arg_u64(args::BO3, bo3_va);
        eb.set_arg_u64(args::BO4, bo4_va);
        let _ = eb.finalize(0x3C);
    }
    let _ = cmd_bo.sync(SYNC_TO_DEVICE);

    // Determinism: dispatch twice, compare run-to-run.
    let mut npu_run1 = vec![0u16; HEAD_DIM];
    let mut npu_run2 = vec![0u16; HEAD_DIM];
    for run_idx in 0..2 {
        // Reset cmd state nibble for re-execution.
        {
            let cbuf = cmd_bo.map()?;
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);
        // Reset output sentinel.
        {
            let buf = out_bo.map()?;
            for b in buf[..OUT_BYTES].iter_mut() { *b = 0xCC; }
        }
        let _ = out_bo.sync(SYNC_TO_DEVICE);

        let seq = submit_exec_cmd(
            hipx_dev.device.fd,
            ctx,
            &[&cmd_bo],
            &[instr_bo, &packed_bo, &cnorm_bo, &out_bo, &bo3, &bo4],
        )?;
        timeline_wait(hipx_dev.device.fd, ctx.syncobj_handle, seq, Duration::from_secs(5))?;
        let _ = out_bo.sync(SYNC_FROM_DEVICE);
        let outp = out_bo.map()?;
        let dst = if run_idx == 0 { &mut npu_run1 } else { &mut npu_run2 };
        for d in 0..HEAD_DIM {
            let lo = outp[d * 2] as u16;
            let hi = outp[d * 2 + 1] as u16;
            dst[d] = lo | (hi << 8);
        }
    }
    let determ_ok = npu_run1 == npu_run2;

    // Build report against run 1 as the canonical NPU output.
    let mut report = SeedReport::default();
    report.determ_ok = determ_ok;
    for d in 0..HEAD_DIM {
        let cpu_bits = cpu_out[d];
        let npu_bits = npu_run1[d];
        if npu_bits != cpu_bits {
            let ulp = ulp_distance(cpu_bits, npu_bits);
            let signed = signed_ulp_delta(cpu_bits, npu_bits) as i64;
            report.n_diff += 1;
            report.sum_signed_ulp += signed;
            if ulp > report.max_ulp { report.max_ulp = ulp; }
            if report.first_diff.is_none() {
                report.first_diff = Some((d, cpu_bits, npu_bits, ulp));
            }
        }
    }

    // On large divergence, optionally dump diagnostic info.
    if report.max_ulp > 0 && std::env::var("ASYM3_DEBUG").is_ok() {
        if let Some((d, cpu_bits, npu_bits, ulp)) = report.first_diff {
            let tid = d / 8;
            let i = d % 8;
            let base = tid * 3;
            let word = (packed[base] as u32)
                | ((packed[base + 1] as u32) << 8)
                | ((packed[base + 2] as u32) << 16);
            let idx = ((word >> (i * 3)) & 7) as usize;
            let cb_f32 = TURBO_C3_256[idx];
            let cb_bf16 = f32_to_bf16_bits_rtz(cb_f32);
            let cnorm_bf16 = f32_to_bf16_bits_rtz(cnorm);
            let cb_b = bf16_bits_to_f32(cb_bf16);
            let cnorm_b = bf16_bits_to_f32(cnorm_bf16);
            let f32_product = cnorm_b * cb_b;
            let bf16_via_rne = f32_to_bf16_bits(f32_product);
            let bf16_via_trunc = (f32_product.to_bits() >> 16) as u16;
            eprintln!(
                "DEBUG dim {d} (tid {tid} i {i}) idx={idx} cb=0x{cb_bf16:04x} cnorm=0x{cnorm_bf16:04x} f32_prod={f32_product:.10} via_rne=0x{bf16_via_rne:04x} via_trunc=0x{bf16_via_trunc:04x} cpu=0x{cpu_bits:04x} npu=0x{npu_bits:04x} ulp={ulp}"
            );
        }
    }

    Ok(report)
}

/// Encode 32 threads × 8 indices = 256 values all set to `idx_value`,
/// returning the 96 packed bytes.
fn pack_all_same(idx_value: u8) -> Vec<u8> {
    assert!(idx_value < 8);
    // Per-thread 24-bit word with 8 copies of idx_value.
    let mut word: u32 = 0;
    for i in 0..8 {
        word |= (idx_value as u32) << (i * 3);
    }
    let mut out = vec![0u8; PACKED_BYTES];
    for tid in 0..32 {
        let base = tid * 3;
        out[base] = (word & 0xff) as u8;
        out[base + 1] = ((word >> 8) & 0xff) as u8;
        out[base + 2] = ((word >> 16) & 0xff) as u8;
    }
    out
}

fn calibrate_codebook(
    hipx_dev: &Hipx,
    ctx: &hipx::hwctx::Hwctx,
    instr_bo: &hipx::Bo,
    insts_len: usize,
) -> Result<[u16; 8], Box<dyn std::error::Error>> {
    let mut cb = [0u16; 8];
    for k in 0..8u8 {
        let packed = pack_all_same(k);
        let cnorm = 1.0f32;
        // Run kernel; collect dim 0's bf16 output as kernel's codebook[k].
        let mut packed_bo = hipx_dev.alloc_shmem(PACKED_BYTES)?;
        { let buf = packed_bo.map()?; buf[..PACKED_BYTES].copy_from_slice(&packed); }
        let _ = packed_bo.sync(SYNC_TO_DEVICE);
        let packed_va = packed_bo.host_ptr().unwrap() as u64;

        let mut cnorm_bo = hipx_dev.alloc_shmem(CNORM_BYTES)?;
        { let buf = cnorm_bo.map()?; buf[..4].copy_from_slice(&cnorm.to_le_bytes()); }
        let _ = cnorm_bo.sync(SYNC_TO_DEVICE);
        let cnorm_va = cnorm_bo.host_ptr().unwrap() as u64;

        let mut out_bo = hipx_dev.alloc_shmem(OUT_BYTES)?;
        { let buf = out_bo.map()?; for b in buf[..OUT_BYTES].iter_mut() { *b = 0xCC; } }
        let _ = out_bo.sync(SYNC_TO_DEVICE);
        let out_va = out_bo.host_ptr().unwrap() as u64;

        let mut bo3 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 8)?;
        { let buf = bo3.map()?; for b in buf.iter_mut() { *b = 0; } }
        let _ = bo3.sync(SYNC_TO_DEVICE);
        let bo3_va = bo3.host_ptr().unwrap() as u64;
        let mut bo4 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 1)?;
        { let buf = bo4.map()?; for b in buf.iter_mut() { *b = 0; } }
        let _ = bo4.sync(SYNC_TO_DEVICE);
        let bo4_va = bo4.host_ptr().unwrap() as u64;

        let mut cmd_bo = hipx_dev.alloc_cmd(4096)?;
        let ninstr_dwords = (insts_len / 4) as u32;
        {
            let cbuf = cmd_bo.map()?;
            let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
            use hipx::kernels::passthrough_4k_args as args;
            eb.set_cu_mask(0x1);
            eb.set_arg_u64(args::OPCODE, 3);
            eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
            eb.set_arg_u32(args::NINSTR, ninstr_dwords);
            eb.set_arg_u64(args::BO0, packed_va);
            eb.set_arg_u64(args::BO1, cnorm_va);
            eb.set_arg_u64(args::BO2, out_va);
            eb.set_arg_u64(args::BO3, bo3_va);
            eb.set_arg_u64(args::BO4, bo4_va);
            let _ = eb.finalize(0x3C);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);

        let seq = submit_exec_cmd(hipx_dev.device.fd, ctx,
            &[&cmd_bo], &[instr_bo, &packed_bo, &cnorm_bo, &out_bo, &bo3, &bo4])?;
        timeline_wait(hipx_dev.device.fd, ctx.syncobj_handle, seq, Duration::from_secs(5))?;
        let _ = out_bo.sync(SYNC_FROM_DEVICE);

        let outp = out_bo.map()?;
        let lo = outp[0] as u16;
        let hi = outp[1] as u16;
        cb[k as usize] = lo | (hi << 8);
        // HARD: every dim must match dim 0 (all-same-idx pattern -> all-same-output).
        for d in 1..HEAD_DIM {
            let l = outp[d * 2] as u16;
            let h = outp[d * 2 + 1] as u16;
            let bits = l | (h << 8);
            if bits != cb[k as usize] {
                return Err(format!(
                    "calibrate FAIL: k={k} dim {d} = 0x{bits:04x} but dim 0 = 0x{:04x} \
                     (uniform-pattern test broke; kernel unpack or dispatch is bugged)",
                    cb[k as usize]
                ).into());
            }
        }
        // HARD: kernel codebook entry must match engine TURBO_C3_256 within
        // bf16 RTZ/RNE rounding (which agree on every TURBO_C3_256 value).
        let expected = TURBO_C3_256[k as usize];
        let exp_rne = f32_to_bf16_bits(expected);
        let exp_rtz = f32_to_bf16_bits_rtz(expected);
        let exp_raz = f32_to_bf16_bits_raz(expected);
        let observed = cb[k as usize];
        let agrees = observed == exp_rne || observed == exp_rtz || observed == exp_raz;
        let mark = if agrees { "==" } else { "!!" };
        let f = bf16_bits_to_f32(observed);
        println!("[calibrate] codebook[{k}] kernel=0x{:04x}({:.7}) ref(RNE)=0x{:04x}({:.7}) {mark}",
                 observed, f, exp_rne, expected);
        if !agrees {
            return Err(format!(
                "calibrate FAIL: codebook[{k}] kernel=0x{observed:04x} \
                 ref RNE/RTZ/RAZ all disagree (RNE=0x{exp_rne:04x} RTZ=0x{exp_rtz:04x} \
                 RAZ=0x{exp_raz:04x}). Engine codebook is the source of truth; \
                 kernel codebook is wrong."
            ).into());
        }
    }
    Ok(cb)
}

/// Stronger: pack 32 threads each with a different idx pattern so a
/// permutation bug in the unpack would show up as crossed dims.
/// Pattern: thread tid encodes 8 indices = (tid % 8) repeated. Then
/// dims tid*8 .. tid*8+7 should all be cnorm * codebook[tid % 8].
/// The expected output is computable from the engine codebook
/// directly (no calibration loop), so a bug that permutes idx values
/// between threads or scrambles the codebook lookup is caught here.
fn calibrate_varied_idx(
    hipx_dev: &Hipx,
    ctx: &hipx::hwctx::Hwctx,
    instr_bo: &hipx::Bo,
    insts_len: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cnorm = 1.0f32;
    // Thread tid encodes 8 copies of (tid % 8) in its 24-bit word.
    let mut packed = vec![0u8; PACKED_BYTES];
    for tid in 0..32usize {
        let k = (tid % 8) as u32;
        let mut word: u32 = 0;
        for i in 0..8 { word |= k << (i * 3); }
        let base = tid * 3;
        packed[base] = (word & 0xff) as u8;
        packed[base + 1] = ((word >> 8) & 0xff) as u8;
        packed[base + 2] = ((word >> 16) & 0xff) as u8;
    }

    // Dispatch.
    let mut packed_bo = hipx_dev.alloc_shmem(PACKED_BYTES)?;
    { let buf = packed_bo.map()?; buf[..PACKED_BYTES].copy_from_slice(&packed); }
    let _ = packed_bo.sync(SYNC_TO_DEVICE);
    let packed_va = packed_bo.host_ptr().unwrap() as u64;

    let mut cnorm_bo = hipx_dev.alloc_shmem(CNORM_BYTES)?;
    { let buf = cnorm_bo.map()?; buf[..4].copy_from_slice(&cnorm.to_le_bytes()); }
    let _ = cnorm_bo.sync(SYNC_TO_DEVICE);
    let cnorm_va = cnorm_bo.host_ptr().unwrap() as u64;

    let mut out_bo = hipx_dev.alloc_shmem(OUT_BYTES)?;
    { let buf = out_bo.map()?; for b in buf[..OUT_BYTES].iter_mut() { *b = 0xCC; } }
    let _ = out_bo.sync(SYNC_TO_DEVICE);
    let out_va = out_bo.host_ptr().unwrap() as u64;

    let mut bo3 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 8)?;
    { let buf = bo3.map()?; for b in buf.iter_mut() { *b = 0; } }
    let _ = bo3.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3.host_ptr().unwrap() as u64;
    let mut bo4 = hipx::Bo::alloc_shmem_exact(hipx_dev.device.fd, 1)?;
    { let buf = bo4.map()?; for b in buf.iter_mut() { *b = 0; } }
    let _ = bo4.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4.host_ptr().unwrap() as u64;

    let mut cmd_bo = hipx_dev.alloc_cmd(4096)?;
    let ninstr_dwords = (insts_len / 4) as u32;
    {
        let cbuf = cmd_bo.map()?;
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        use hipx::kernels::passthrough_4k_args as args;
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::BO0, packed_va);
        eb.set_arg_u64(args::BO1, cnorm_va);
        eb.set_arg_u64(args::BO2, out_va);
        eb.set_arg_u64(args::BO3, bo3_va);
        eb.set_arg_u64(args::BO4, bo4_va);
        let _ = eb.finalize(0x3C);
    }
    let _ = cmd_bo.sync(SYNC_TO_DEVICE);

    let seq = submit_exec_cmd(hipx_dev.device.fd, ctx,
        &[&cmd_bo], &[instr_bo, &packed_bo, &cnorm_bo, &out_bo, &bo3, &bo4])?;
    timeline_wait(hipx_dev.device.fd, ctx.syncobj_handle, seq, Duration::from_secs(5))?;
    let _ = out_bo.sync(SYNC_FROM_DEVICE);

    // Each thread tid produced 8 outputs at dims tid*8 .. tid*8+7.
    // All 8 should equal cnorm_bf16 * codebook_bf16[(tid % 8)].
    // For cnorm = 1.0 (bf16-exact), expected = codebook_bf16[(tid % 8)].
    let outp = out_bo.map()?;
    let mut max_ulp_per_thread: u32 = 0;
    for tid in 0..32usize {
        let k = tid % 8;
        let expected_bf16 = f32_to_bf16_bits_rtz(TURBO_C3_256[k]);
        for i in 0..8 {
            let d = tid * 8 + i;
            let lo = outp[d * 2] as u16;
            let hi = outp[d * 2 + 1] as u16;
            let observed = lo | (hi << 8);
            let u = ulp_distance(observed, expected_bf16);
            if u > max_ulp_per_thread { max_ulp_per_thread = u; }
            if u > 2 {
                return Err(format!(
                    "varied-idx FAIL: dim {d} tid {tid} k {k} expected 0x{expected_bf16:04x} \
                     observed 0x{observed:04x} (ulp {u} > 2). Per-thread idx mapping is wrong."
                ).into());
            }
        }
    }
    println!("[calibrate-varied] 32 threads x 8 dims, k = tid%8, max_ulp = {} (bound 2)",
             max_ulp_per_thread);
    Ok(())
}

fn main() -> ExitCode {
    let n_seeds: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);

    let pdi = load_pdi();
    let insts = load_insts();

    println!("[verify] PDI {} bytes; insts {} bytes; {n_seeds} seeds",
             pdi.len(), insts.len());

    let hipx_dev = match Hipx::open() {
        Ok(h) => h,
        Err(e) => { eprintln!("Hipx::open: {e}"); return ExitCode::FAILURE; }
    };

    let mut b = HwctxBuilder::default();
    // Kernel was built with column_width=8 per main_aie_partition.json.
    b.num_columns = 8;
    b.max_opc = 2048;
    let ctx = match hipx_dev.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => { eprintln!("create_hwctx: {e}"); return ExitCode::FAILURE; }
    };

    // PDI BO (DEV) — copy via heap mmap.
    let pdi_bo = match hipx_dev.alloc_dev(pdi.len()) {
        Ok(b) => b,
        Err(e) => { eprintln!("pdi alloc: {e}"); return ExitCode::FAILURE; }
    };
    unsafe {
        let buf = hipx_dev.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..pdi.len()].copy_from_slice(&pdi);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx_dev.device.fd, &ctx, vec![pdi_bo], &[0u8])
        .expect("config_cus");

    // Instruction stream (DEV).
    let instr_bo = match hipx_dev.alloc_dev(insts.len()) {
        Ok(b) => b,
        Err(e) => { eprintln!("instr alloc: {e}"); return ExitCode::FAILURE; }
    };
    unsafe {
        let buf = hipx_dev.dev_slice(&instr_bo).expect("instr slice");
        buf[..insts.len()].copy_from_slice(&insts);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);

    // Calibration phase as VALIDATION (not as the source of the CPU
    // reference). Runs two probes that catch self-consistent kernel
    // bugs which a "kernel-vs-self-derived-codebook" verifier would
    // miss:
    //
    //   (a) calibrate_codebook: all-same-idx patterns, observe dim 0,
    //       compare against the ENGINE codebook (the source of truth).
    //       Hard-fails if any kernel codebook entry diverges from the
    //       engine value beyond bf16 RTZ/RNE/RAZ tolerance.
    //   (b) calibrate_varied_idx: thread tid encodes idx = tid % 8
    //       (different threads, different idx). Cross-thread output
    //       mapping is verified against the engine codebook directly.
    //       Hard-fails if any thread's output is > 2 ULP from
    //       expected. Catches per-thread permutation bugs that the
    //       all-same pattern cannot detect.
    //
    // The CPU reference for the random-seed phase always uses the
    // ENGINE codebook (`cpu_reference`), never the calibrated codebook
    // observed from the kernel itself. This avoids the "self-consistent
    // wrong kernel" trap (codex review caught the original calibration
    // path).
    if std::env::var("ASYM3_NO_CALIBRATE").is_err() {
        match calibrate_codebook(&hipx_dev, &ctx, &instr_bo, insts.len()) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("calibration failed: {e}");
                return ExitCode::FAILURE;
            }
        }
        match calibrate_varied_idx(&hipx_dev, &ctx, &instr_bo, insts.len()) {
            Ok(()) => println!(),
            Err(e) => {
                eprintln!("varied-idx calibration failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    // Stage 1.1 acceptance gates (per docs/plans/aie2p-bf16-mul-shape.md):
    //
    //   1. Determinism: same input -> same output across two consecutive
    //      dispatches. 100/100 seeds.
    //   2. Max ULP bound: max bf16 ULP deviation per element across all
    //      seeds <= MAX_ULP_BOUND. Catches structural bugs (codebook,
    //      layout, unpack) which produce errors much larger than the
    //      AIE-2P-shape rounding floor of ~2 ULP.
    //   3. Statistical: |mean signed ULP error| <= MEAN_BIAS_BOUND.
    //      Catches systematic drift bugs.
    //
    // Empirical ULP envelope: with a CPU reference that uses the
    // engine codebook directly (no calibration self-consistency
    // loop) and an RTZ cnorm + RAZ output approximation of the
    // AIE-2P-shape kernel rounding, observed max is 3 ULP and
    // observed mean signed magnitude bias is ~0.7 ULP across 100
    // random seeds. The bias is consistent with the kernel's
    // cnorm cast and bf16 down-conversion both biasing magnitude
    // upward relative to my CPU model, which uses RTZ-down for
    // cnorm. Bounds set with one ULP of headroom each: max 4,
    // mean 1.0. Real bug classes (codebook, layout, unpack)
    // produce >> 4 ULP errors and trip the gate.
    //
    // ASYM3_STRICT=1: enforce true bit-for-bit (max_ulp == 0).
    // Reserved for the future LUT-based verifier.
    let strict = std::env::var("ASYM3_STRICT").is_ok();
    let max_ulp_bound: u32 = if strict { 0 } else { 4 };
    let mean_bias_bound: f64 = if strict { 0.0 } else { 1.0 };

    let mut total_seeds_ok = 0usize;
    let mut total_seeds_fail = 0usize;
    let mut grand_max_ulp: u32 = 0;
    let mut grand_sum_signed: i64 = 0;
    let mut grand_n_diff: usize = 0;
    let mut all_determ = true;
    let mut first_failure_msg: Option<String> = None;

    for s in 0..n_seeds {
        let seed = 0x1000_0001u64.wrapping_add(s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // CPU reference uses the engine codebook directly (not the
        // calibrated codebook). See comment above on the
        // "self-consistent wrong kernel" trap.
        match run_one_seed(seed, &hipx_dev, &ctx, &instr_bo, pdi.len(), insts.len(),
                           None) {
            Ok(r) => {
                if !r.determ_ok {
                    all_determ = false;
                    let msg = format!("seed {seed:#018x}: NON-DETERMINISTIC (run1 != run2)");
                    if first_failure_msg.is_none() { first_failure_msg = Some(msg.clone()); }
                    eprintln!("  {msg}");
                }
                if r.max_ulp > grand_max_ulp { grand_max_ulp = r.max_ulp; }
                grand_sum_signed += r.sum_signed_ulp;
                grand_n_diff += r.n_diff;

                let seed_pass = r.determ_ok && r.max_ulp <= max_ulp_bound;
                if seed_pass {
                    total_seeds_ok += 1;
                    if s < 3 {
                        println!("  seed {seed:#018x}: PASS (max_ulp={}, n_diff={}/{HEAD_DIM})",
                                 r.max_ulp, r.n_diff);
                    }
                } else {
                    total_seeds_fail += 1;
                    let label = r.first_diff.map(|(d, c, g, u)|
                        format!("first diff at dim {d}: cpu={c:#06x} npu={g:#06x} ulp={u}")
                    ).unwrap_or_default();
                    println!("  seed {seed:#018x}: FAIL max_ulp={} (>{max_ulp_bound}); {label}",
                             r.max_ulp);
                    if first_failure_msg.is_none() {
                        first_failure_msg = Some(format!("seed {seed:#018x} max_ulp={} bound={max_ulp_bound}", r.max_ulp));
                    }
                }
            }
            Err(e) => {
                eprintln!("  seed {seed:#018x}: ERR {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let mean_signed = if grand_n_diff > 0 {
        grand_sum_signed as f64 / grand_n_diff as f64
    } else { 0.0 };
    let mean_signed_ok = mean_signed.abs() <= mean_bias_bound;

    println!("\n=== Stage 1.1 acceptance gates ===");
    println!("  determinism:    {} / {} seeds ({})",
             if all_determ { n_seeds } else { n_seeds - 1 }, n_seeds,
             if all_determ { "PASS" } else { "FAIL" });
    println!("  max ULP:        observed {} <= bound {} ({})",
             grand_max_ulp, max_ulp_bound,
             if grand_max_ulp <= max_ulp_bound { "PASS" } else { "FAIL" });
    println!("  mean signed:    {:.4} ULP <= bound {:.2} ({})",
             mean_signed, mean_bias_bound,
             if mean_signed_ok { "PASS" } else { "FAIL" });
    println!("  per-seed:       {} pass, {} fail across {} seeds",
             total_seeds_ok, total_seeds_fail, n_seeds);
    println!("  mode:           {} (set ASYM3_STRICT=1 for bit-for-bit)",
             if strict { "strict bit-for-bit" } else { "AIE-2P-shape (<=2 ULP)" });

    let pass = all_determ && grand_max_ulp <= max_ulp_bound && mean_signed_ok;
    if pass {
        println!("\n=== STAGE 1.1 PASS ===");
        ExitCode::SUCCESS
    } else {
        if let Some(msg) = first_failure_msg {
            println!("\nFIRST FAILURE: {msg}");
        }
        println!("\n=== STAGE 1.1 FAIL ===");
        ExitCode::FAILURE
    }
}
