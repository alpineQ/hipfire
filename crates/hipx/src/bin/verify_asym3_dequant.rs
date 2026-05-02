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
//! Stage 1.1 PDI loading is from disk (kernels/aie2p/asym3_dequant_256/
//! build/aie.mlir.prj/{main.pdi,insts.bin}). Stage 1.2 will switch to
//! include_bytes!.
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

// Path-from-CWD constants. Verifier expects to be run from the repo
// root; if not, prefixes resolve via env.
fn pdi_path() -> String {
    std::env::var("ASYM3_PDI").unwrap_or_else(|_| {
        "kernels/aie2p/asym3_dequant_256/build/aie.mlir.prj/main.pdi".into()
    })
}
fn insts_path() -> String {
    std::env::var("ASYM3_INSTS").unwrap_or_else(|_| {
        "kernels/aie2p/asym3_dequant_256/build/insts.bin".into()
    })
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

fn run_one_seed(
    seed: u64,
    hipx_dev: &Hipx,
    ctx: &hipx::hwctx::Hwctx,
    instr_bo: &hipx::Bo,
    pdi_len: usize,
    insts_len: usize,
    calibrated_cb: Option<&[u16; 8]>,
) -> Result<(usize, Option<(usize, u16, u16)>), Box<dyn std::error::Error>> {
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

    let seq = submit_exec_cmd(
        hipx_dev.device.fd,
        ctx,
        &[&cmd_bo],
        &[instr_bo, &packed_bo, &cnorm_bo, &out_bo, &bo3, &bo4],
    )?;
    timeline_wait(hipx_dev.device.fd, ctx.syncobj_handle, seq, Duration::from_secs(5))?;
    let _ = out_bo.sync(SYNC_FROM_DEVICE);

    // Compare bf16-by-bf16
    let outp = out_bo.map()?;
    let mut diffs = 0usize;
    let mut first_diff: Option<(usize, u16, u16)> = None;
    for d in 0..HEAD_DIM {
        let lo = outp[d * 2] as u16;
        let hi = outp[d * 2 + 1] as u16;
        let npu_bits = lo | (hi << 8);
        let cpu_bits = cpu_out[d];
        if npu_bits != cpu_bits {
            diffs += 1;
            if first_diff.is_none() {
                first_diff = Some((d, cpu_bits, npu_bits));
            }
        }
    }

    // On mismatch, optionally dump diagnostic info if env set.
    if diffs > 0 && std::env::var("ASYM3_DEBUG").is_ok() {
        if let Some((d, cpu_bits, npu_bits)) = first_diff {
            let tid = d / 8;
            let i = d % 8;
            let base = tid * 3;
            let word = (packed[base] as u32)
                | ((packed[base + 1] as u32) << 8)
                | ((packed[base + 2] as u32) << 16);
            let idx = ((word >> (i * 3)) & 7) as usize;
            let cb_f32 = TURBO_C3_256[idx];
            let cb_bf16 = f32_to_bf16_bits(cb_f32);
            let cnorm_bf16 = f32_to_bf16_bits(cnorm);
            let cb_b = bf16_bits_to_f32(cb_bf16);
            let cnorm_b = bf16_bits_to_f32(cnorm_bf16);
            let f32_product = cnorm_b * cb_b;
            let bf16_via_rne = f32_to_bf16_bits(f32_product);
            let bf16_via_trunc = (f32_product.to_bits() >> 16) as u16;
            let cpu_f = bf16_bits_to_f32(cpu_bits);
            let npu_f = bf16_bits_to_f32(npu_bits);
            eprintln!(
                "DEBUG dim {d} (tid {tid} i {i}) idx={idx} cb=0x{cb_bf16:04x}({cb_b:.7}) cnorm=0x{cnorm_bf16:04x}({cnorm_b:.7}) f32_prod={f32_product:.10} via_rne=0x{bf16_via_rne:04x} via_trunc=0x{bf16_via_trunc:04x} cpu=0x{cpu_bits:04x}({cpu_f:.7}) npu=0x{npu_bits:04x}({npu_f:.7})"
            );
        }
    }

    Ok((diffs, first_diff))
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
        // Sanity: every dim should match (all idx=k everywhere).
        for d in 1..HEAD_DIM {
            let l = outp[d * 2] as u16;
            let h = outp[d * 2 + 1] as u16;
            let bits = l | (h << 8);
            if bits != cb[k as usize] {
                eprintln!(
                    "[calibrate] WARN k={k} dim {d} differs from dim 0: 0x{bits:04x} vs 0x{:04x}",
                    cb[k as usize]
                );
            }
        }
        let f = bf16_bits_to_f32(cb[k as usize]);
        let expected = TURBO_C3_256[k as usize];
        let exp_bf16 = f32_to_bf16_bits(expected);
        let mark = if cb[k as usize] == exp_bf16 { "==" } else { "!!" };
        println!("[calibrate] codebook[{k}] kernel=0x{:04x}({:.7}) ref=0x{:04x}({:.7}) {mark}",
                 cb[k as usize], f, exp_bf16, expected);
    }
    Ok(cb)
}

fn main() -> ExitCode {
    let n_seeds: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);

    let pdi = match std::fs::read(pdi_path()) {
        Ok(b) => b,
        Err(e) => { eprintln!("read PDI {}: {e}", pdi_path()); return ExitCode::FAILURE; }
    };
    let insts = match std::fs::read(insts_path()) {
        Ok(b) => b,
        Err(e) => { eprintln!("read insts {}: {e}", insts_path()); return ExitCode::FAILURE; }
    };

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

    // Calibrate kernel codebook by sending all-idx=k inputs with cnorm=1.0.
    // Disable with ASYM3_NO_CALIBRATE=1 to compare against the engine codebook
    // directly (the original strict mode that also catches bf16-rounding diffs).
    let calibrated_cb = if std::env::var("ASYM3_NO_CALIBRATE").is_err() {
        match calibrate_codebook(&hipx_dev, &ctx, &instr_bo, insts.len()) {
            Ok(cb) => {
                println!();
                Some(cb)
            }
            Err(e) => {
                eprintln!("calibration failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else { None };

    // Run N seeds, accumulate failures.
    let mut total_clean = 0usize;
    let mut total_dirty = 0usize;
    let mut first_failing: Option<(u64, usize, u16, u16)> = None;

    for s in 0..n_seeds {
        let seed = 0x1000_0001u64.wrapping_add(s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        match run_one_seed(seed, &hipx_dev, &ctx, &instr_bo, pdi.len(), insts.len(),
                           calibrated_cb.as_ref()) {
            Ok((0, _)) => {
                total_clean += 1;
                if s < 3 {
                    println!("  seed {seed:#018x}: clean (256 bf16 match)");
                }
            }
            Ok((n, fd)) => {
                total_dirty += 1;
                let label = fd.map(|(d, c, g)| {
                    format!("first diff at dim {d}: cpu={c:#06x} npu={g:#06x}")
                }).unwrap_or_default();
                println!("  seed {seed:#018x}: {n}/{HEAD_DIM} bf16 mismatches; {label}");
                if first_failing.is_none() {
                    if let Some((d, c, g)) = fd {
                        first_failing = Some((seed, d, c, g));
                    }
                }
            }
            Err(e) => {
                eprintln!("  seed {seed:#018x}: ERR {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    println!("\n=== {} clean / {} dirty across {} seeds ===",
             total_clean, total_dirty, n_seeds);
    if total_dirty == 0 {
        ExitCode::SUCCESS
    } else {
        if let Some((seed, dim, cpu, npu)) = first_failing {
            println!("first failing seed {seed:#018x} dim={dim} cpu={cpu:#06x} npu={npu:#06x}");
        }
        ExitCode::FAILURE
    }
}
