//! `hipx-diagnose-asym3-mul` — characterize AIE-2P bf16 mul rounding.
//!
//! Runs the f32 variant of the asym3 dequant kernel
//! (kernels/aie2p/asym3_dequant_256_f32/) which returns the
//! `aie::mul` accumulator as fp32. Compares against the CPU's
//! pure-fp32 product of bf16-promoted-to-f32 inputs.
//!
//! Outputs three pieces of info per dim:
//!
//!   - CPU f32 product (cnorm bf16 -> f32 * cb bf16 -> f32, x86 fp32 mul)
//!   - NPU f32 output (the AIE-2P accumulator pre-bf16-conversion)
//!   - NPU bf16 from the regular kernel
//!
//! If CPU f32 == NPU f32 byte-for-byte, the mul is lossless on bf16
//! inputs and we just need to model the to_vector<bfloat16>()
//! rounding.
//!
//! If CPU f32 != NPU f32, the mul itself uses limited internal
//! precision (e.g. bf16-mantissa internal multiplier) and we must
//! model the mul, not just the conversion.
//!
//! Run from repo root (PDI paths default to that):
//!   cargo run -p hipx --bin diagnose_asym3_mul

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::Hipx;

const HEAD_DIM: usize = 256;
const PACKED_BYTES: usize = HEAD_DIM * 3 / 8;
const OUT_F32_BYTES: usize = HEAD_DIM * 4;
const OUT_BF16_BYTES: usize = HEAD_DIM * 2;

const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176,
     0.015176,  0.046469,  0.083320,  0.134860,
];

fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let xb = x.to_bits();
    if (xb & 0x7fff_ffff) > 0x7f80_0000 {
        return ((xb >> 16) | 0x0040) as u16;
    }
    let lsb = (xb >> 16) & 1;
    let bias = 0x7fff + lsb;
    ((xb.wrapping_add(bias)) >> 16) as u16
}
fn f32_to_bf16_bits_rtz(x: f32) -> u16 { (x.to_bits() >> 16) as u16 }
fn f32_to_bf16_bits_raz(x: f32) -> u16 {
    // Round away from zero: bias positive results up, negative down.
    let xb = x.to_bits();
    let abs = xb & 0x7fff_ffff;
    let sign = xb & 0x8000_0000;
    let biased = abs + 0xffff;  // any non-zero low 16 bits round up in magnitude
    (((sign | (biased & 0x7fff_ffff)) >> 16)) as u16
}
fn bf16_bits_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

fn dispatch_f32(
    hipx_dev: &Hipx,
    ctx: &hipx::hwctx::Hwctx,
    instr_bo: &hipx::Bo,
    insts_len: usize,
    packed: &[u8],
    cnorm: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut packed_bo = hipx_dev.alloc_shmem(PACKED_BYTES)?;
    { let buf = packed_bo.map()?; buf[..PACKED_BYTES].copy_from_slice(packed); }
    let _ = packed_bo.sync(SYNC_TO_DEVICE);
    let packed_va = packed_bo.host_ptr().unwrap() as u64;

    let mut cnorm_bo = hipx_dev.alloc_shmem(4)?;
    { let buf = cnorm_bo.map()?; buf[..4].copy_from_slice(&cnorm.to_le_bytes()); }
    let _ = cnorm_bo.sync(SYNC_TO_DEVICE);
    let cnorm_va = cnorm_bo.host_ptr().unwrap() as u64;

    let mut out_bo = hipx_dev.alloc_shmem(OUT_F32_BYTES)?;
    { let buf = out_bo.map()?; for b in buf[..OUT_F32_BYTES].iter_mut() { *b = 0xCC; } }
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
    let mut result = vec![0f32; HEAD_DIM];
    for d in 0..HEAD_DIM {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&outp[d * 4..d * 4 + 4]);
        result[d] = f32::from_le_bytes(bytes);
    }
    Ok(result)
}

fn dispatch_bf16(
    hipx_dev: &Hipx,
    ctx: &hipx::hwctx::Hwctx,
    instr_bo: &hipx::Bo,
    insts_len: usize,
    packed: &[u8],
    cnorm: f32,
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let mut packed_bo = hipx_dev.alloc_shmem(PACKED_BYTES)?;
    { let buf = packed_bo.map()?; buf[..PACKED_BYTES].copy_from_slice(packed); }
    let _ = packed_bo.sync(SYNC_TO_DEVICE);
    let packed_va = packed_bo.host_ptr().unwrap() as u64;

    let mut cnorm_bo = hipx_dev.alloc_shmem(4)?;
    { let buf = cnorm_bo.map()?; buf[..4].copy_from_slice(&cnorm.to_le_bytes()); }
    let _ = cnorm_bo.sync(SYNC_TO_DEVICE);
    let cnorm_va = cnorm_bo.host_ptr().unwrap() as u64;

    let mut out_bo = hipx_dev.alloc_shmem(OUT_BF16_BYTES)?;
    { let buf = out_bo.map()?; for b in buf[..OUT_BF16_BYTES].iter_mut() { *b = 0xCC; } }
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
    let mut result = vec![0u16; HEAD_DIM];
    for d in 0..HEAD_DIM {
        let lo = outp[d * 2] as u16;
        let hi = outp[d * 2 + 1] as u16;
        result[d] = lo | (hi << 8);
    }
    Ok(result)
}

fn unpack_idx(packed: &[u8], dim: usize) -> usize {
    let tid = dim / 8;
    let i = dim % 8;
    let base = tid * 3;
    let word = (packed[base] as u32)
        | ((packed[base + 1] as u32) << 8)
        | ((packed[base + 2] as u32) << 16);
    ((word >> (i * 3)) & 7) as usize
}

fn open_with(
    pdi_path: &str,
    insts_path: &str,
) -> Result<(Hipx, hipx::hwctx::Hwctx, hipx::Bo, usize, hipx::cmd::CuBinding),
            Box<dyn std::error::Error>> {
    let pdi = std::fs::read(pdi_path)?;
    let insts = std::fs::read(insts_path)?;
    let hipx_dev = Hipx::open()?;
    let mut b = HwctxBuilder::default();
    b.num_columns = 8;
    b.max_opc = 2048;
    let ctx = hipx_dev.create_hwctx(&b)?;

    let pdi_bo = hipx_dev.alloc_dev(pdi.len())?;
    unsafe {
        let buf = hipx_dev.dev_slice(&pdi_bo).unwrap();
        buf[..pdi.len()].copy_from_slice(&pdi);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let cu = config_cus(hipx_dev.device.fd, &ctx, vec![pdi_bo], &[0u8])?;

    let instr_bo = hipx_dev.alloc_dev(insts.len())?;
    unsafe {
        let buf = hipx_dev.dev_slice(&instr_bo).unwrap();
        buf[..insts.len()].copy_from_slice(&insts);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);

    Ok((hipx_dev, ctx, instr_bo, insts.len(), cu))
}

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Use a single seed and dump 16 dims worth of comparison.
    let cnorm: f32 = 0.7654321;
    // Pick a packed pattern that hits every codebook entry at least
    // once across the first 16 dims.
    let packed = {
        let mut p = vec![0u8; PACKED_BYTES];
        let pattern: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        // Encode tid 0 indices = pattern (so dims 0..7 hit codebook 0..7).
        // tid 1 same again (dims 8..15).
        let mut word: u32 = 0;
        for i in 0..8 { word |= (pattern[i] as u32) << (i * 3); }
        for tid in 0..2 {
            let base = tid * 3;
            p[base] = (word & 0xff) as u8;
            p[base + 1] = ((word >> 8) & 0xff) as u8;
            p[base + 2] = ((word >> 16) & 0xff) as u8;
        }
        p
    };

    eprintln!("=== diagnose: cnorm={cnorm} pattern dims 0..15 = idx 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7 ===");

    // Run the f32 diagnostic kernel.
    let f32_pdi = "kernels/aie2p/asym3_dequant_256_f32/build/main.pdi";
    let f32_insts = "kernels/aie2p/asym3_dequant_256_f32/build/insts.bin";
    let (h1, ctx1, instr1, ilen1, _cu1) = open_with(f32_pdi, f32_insts)?;
    let f32_out = dispatch_f32(&h1, &ctx1, &instr1, ilen1, &packed, cnorm)?;
    drop((ctx1, h1)); // Allow second hwctx
    // We can't keep both hwctxs alive trivially; let's just open + close.

    // Run the regular bf16 kernel.
    let bf16_pdi = "kernels/aie2p/asym3_dequant_256/build/main.pdi";
    let bf16_insts = "kernels/aie2p/asym3_dequant_256/build/insts.bin";
    let (h2, ctx2, instr2, ilen2, _cu2) = open_with(bf16_pdi, bf16_insts)?;
    let bf16_out = dispatch_bf16(&h2, &ctx2, &instr2, ilen2, &packed, cnorm)?;
    drop((ctx2, h2));

    println!("\n{:>4} {:>3} {:>10} {:>14} {:>14} {:>14} {:>10} {:>10} {:>10} {:>10}",
             "dim", "idx", "cb_bf16", "cpu_f32", "npu_f32_acc", "diff_f32", "rne", "rtz", "raz", "npu_bf16");
    for d in 0..16 {
        let idx = unpack_idx(&packed, d);
        let cb_bf16 = f32_to_bf16_bits_rne(TURBO_C3_256[idx]);
        let cb_b = bf16_bits_to_f32(cb_bf16);
        let cnorm_bf16 = f32_to_bf16_bits_rne(cnorm);
        let cnorm_b = bf16_bits_to_f32(cnorm_bf16);
        let cpu_f32 = cnorm_b * cb_b;
        let diff = f32_out[d] - cpu_f32;
        let rne = f32_to_bf16_bits_rne(f32_out[d]);
        let rtz = f32_to_bf16_bits_rtz(f32_out[d]);
        let raz = f32_to_bf16_bits_raz(f32_out[d]);
        println!("{:4} {:3} 0x{:04x} {:14.10} {:14.10} {:14.4e} 0x{:04x}     0x{:04x}     0x{:04x}     0x{:04x}",
                 d, idx, cb_bf16, cpu_f32, f32_out[d], diff, rne, rtz, raz, bf16_out[d]);
    }

    // Cross-tab summary: how many of dim 0..15 match each rounding mode of NPU's f32 output
    let mut hits_rne = 0;
    let mut hits_rtz = 0;
    let mut hits_raz = 0;
    let mut hits_cpu_rne = 0;
    for d in 0..HEAD_DIM {
        if f32_to_bf16_bits_rne(f32_out[d]) == bf16_out[d] { hits_rne += 1; }
        if f32_to_bf16_bits_rtz(f32_out[d]) == bf16_out[d] { hits_rtz += 1; }
        if f32_to_bf16_bits_raz(f32_out[d]) == bf16_out[d] { hits_raz += 1; }
        // For comparison: would CPU's f32-mul+RNE match NPU's bf16 output?
        let idx = unpack_idx(&packed, d);
        let cb_b = bf16_bits_to_f32(f32_to_bf16_bits_rne(TURBO_C3_256[idx]));
        let cnorm_b = bf16_bits_to_f32(f32_to_bf16_bits_rne(cnorm));
        if f32_to_bf16_bits_rne(cnorm_b * cb_b) == bf16_out[d] { hits_cpu_rne += 1; }
    }
    println!("\nSummary across all 256 dims:");
    println!("  npu_bf16 == bf16_rne(npu_f32_acc):  {}/256", hits_rne);
    println!("  npu_bf16 == bf16_rtz(npu_f32_acc):  {}/256", hits_rtz);
    println!("  npu_bf16 == bf16_raz(npu_f32_acc):  {}/256", hits_raz);
    println!("  npu_bf16 == bf16_rne(cpu_f32_prod): {}/256", hits_cpu_rne);

    Ok(())
}
