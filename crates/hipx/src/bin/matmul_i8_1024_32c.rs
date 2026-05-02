//! `hipx-matmul-i8-1024-32c` - 1024x1024x1024 INT8 matmul on AIE-2P NPU2,
//! full 8-col x 4-row = 32-tile fan-out, COLUMN-MAJOR B layout.
//!
//! 8x more compute than the 512^3 32c variant; dispatch overhead amortizes
//! proportionally so this is where 32-core compute throughput becomes
//! visible. Input verifier uses the same Knuth-hash pseudo-random pattern
//! as the 512 variant to defeat tile-remap shift symmetry.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::kernels::{
    matmul_i8_1024_32c_args as args, MATMUL_I8_1024_32C_COLUMNS, MATMUL_I8_1024_32C_INSTS,
    MATMUL_I8_1024_32C_K, MATMUL_I8_1024_32C_M, MATMUL_I8_1024_32C_N,
    MATMUL_I8_1024_32C_OPS_PER_CYCLE, MATMUL_I8_1024_32C_PDI,
};
use hipx::Hipx;

const A_BYTES: usize = MATMUL_I8_1024_32C_M * MATMUL_I8_1024_32C_K;
const B_BYTES: usize = MATMUL_I8_1024_32C_K * MATMUL_I8_1024_32C_N;
const C_BYTES: usize = MATMUL_I8_1024_32C_M * MATMUL_I8_1024_32C_N * 4;

fn main() -> ExitCode {
    let hipx = match Hipx::open() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Hipx::open: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[mm8_1024_32c] device: AIE {}.{}, fw {}.{}.{} build {}",
        hipx.info.aie_version.0,
        hipx.info.aie_version.1,
        hipx.info.firmware_version.0,
        hipx.info.firmware_version.1,
        hipx.info.firmware_version.2,
        hipx.info.firmware_version.3
    );

    let mut b = HwctxBuilder::default();
    b.num_columns = MATMUL_I8_1024_32C_COLUMNS;
    b.max_opc = MATMUL_I8_1024_32C_OPS_PER_CYCLE;
    let ctx = match hipx.create_hwctx(&b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create_hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[mm8_1024_32c] hwctx={} cols={} M=K=N={} (32 cores, INT8, B col-major)",
        ctx.handle, MATMUL_I8_1024_32C_COLUMNS, MATMUL_I8_1024_32C_M
    );

    let pdi_bo = hipx
        .alloc_dev(MATMUL_I8_1024_32C_PDI.len())
        .expect("pdi alloc");
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..MATMUL_I8_1024_32C_PDI.len()].copy_from_slice(MATMUL_I8_1024_32C_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _cu = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8]).expect("config_cus");

    let instr_bo = hipx
        .alloc_dev(MATMUL_I8_1024_32C_INSTS.len())
        .expect("instr alloc");
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..MATMUL_I8_1024_32C_INSTS.len()].copy_from_slice(MATMUL_I8_1024_32C_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (MATMUL_I8_1024_32C_INSTS.len() / 4) as u32;

    // Pseudo-random inputs (Knuth multiplicative hash) for full per-position
    // uniqueness; defeats tile-remap shift symmetry. See 512_32c bin for the
    // long-form rationale.
    fn pseudo_i8(idx: u64) -> i8 {
        let h = idx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h >> 61) as i8) - 4
    }
    let mut a_host = vec![0i8; MATMUL_I8_1024_32C_M * MATMUL_I8_1024_32C_K];
    for r in 0..MATMUL_I8_1024_32C_M {
        for kk in 0..MATMUL_I8_1024_32C_K {
            let idx = (r * MATMUL_I8_1024_32C_K + kk) as u64;
            a_host[r * MATMUL_I8_1024_32C_K + kk] = pseudo_i8(idx);
        }
    }
    // B is column-major: b_host_cm[c * K + kk] = B[kk, c]
    let b_offset = (MATMUL_I8_1024_32C_M * MATMUL_I8_1024_32C_K) as u64;
    let mut b_host_cm = vec![0i8; MATMUL_I8_1024_32C_K * MATMUL_I8_1024_32C_N];
    for kk in 0..MATMUL_I8_1024_32C_K {
        for c in 0..MATMUL_I8_1024_32C_N {
            let logical_idx = b_offset + (kk * MATMUL_I8_1024_32C_N + c) as u64;
            let v = pseudo_i8(logical_idx);
            b_host_cm[c * MATMUL_I8_1024_32C_K + kk] = v;
        }
    }

    let mut a_bo = hipx.alloc_shmem(A_BYTES).expect("A alloc");
    {
        let buf = a_bo.map().expect("A map");
        buf[..A_BYTES].copy_from_slice(unsafe {
            std::slice::from_raw_parts(a_host.as_ptr() as *const u8, A_BYTES)
        });
    }
    let _ = a_bo.sync(SYNC_TO_DEVICE);
    let a_va = a_bo.host_ptr().unwrap() as u64;

    let mut b_bo = hipx.alloc_shmem(B_BYTES).expect("B alloc");
    {
        let buf = b_bo.map().expect("B map");
        buf[..B_BYTES].copy_from_slice(unsafe {
            std::slice::from_raw_parts(b_host_cm.as_ptr() as *const u8, B_BYTES)
        });
    }
    let _ = b_bo.sync(SYNC_TO_DEVICE);
    let b_va = b_bo.host_ptr().unwrap() as u64;

    let mut c_bo = hipx.alloc_shmem(C_BYTES).expect("C alloc");
    {
        let buf = c_bo.map().expect("C map");
        for byte in buf[..C_BYTES].iter_mut() {
            *byte = 0xCC;
        }
    }
    let _ = c_bo.sync(SYNC_TO_DEVICE);
    let c_va = c_bo.host_ptr().unwrap() as u64;

    let mut bo3_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 1).expect("bo3 alloc");
    {
        let buf = bo3_bo.map().expect("bo3 map");
        for b in buf.iter_mut() {
            *b = 0;
        }
    }
    let _ = bo3_bo.sync(SYNC_TO_DEVICE);
    let bo3_va = bo3_bo.host_ptr().unwrap() as u64;
    let mut bo4_bo = hipx::Bo::alloc_shmem_exact(hipx.device.fd, 4).expect("bo4 alloc");
    {
        let buf = bo4_bo.map().expect("bo4 map");
        for b in buf.iter_mut() {
            *b = 0;
        }
    }
    let _ = bo4_bo.sync(SYNC_TO_DEVICE);
    let bo4_va = bo4_bo.host_ptr().unwrap() as u64;

    println!(
        "[mm8_1024_32c] BOs: A={A_BYTES} ({}KiB) B={B_BYTES} ({}KiB) C={C_BYTES} ({}KiB)",
        A_BYTES / 1024,
        B_BYTES / 1024,
        C_BYTES / 1024
    );

    let mut cmd_bo = hipx.alloc_cmd(4096).expect("cmd alloc");
    {
        let cbuf = cmd_bo.map().expect("cmd map");
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

    let t0 = Instant::now();
    let seq = match submit_exec_cmd(
        hipx.device.fd,
        &ctx,
        &[&cmd_bo],
        &[&instr_bo, &a_bo, &b_bo, &c_bo, &bo3_bo, &bo4_bo],
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("submit: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = timeline_wait(
        hipx.device.fd,
        ctx.syncobj_handle,
        seq,
        Duration::from_secs(30),
    ) {
        eprintln!("timeline_wait: {e}");
        return ExitCode::FAILURE;
    }
    let single_us = t0.elapsed().as_micros();
    let _ = c_bo.sync(SYNC_FROM_DEVICE);

    let outp = c_bo.map().expect("C re-map");

    // Full reference. b_host_cm uses col-major layout; reference uses
    // logical row-major B by indexing via (kk, c).
    let mut c_ref = vec![0i32; MATMUL_I8_1024_32C_M * MATMUL_I8_1024_32C_N];
    for r in 0..MATMUL_I8_1024_32C_M {
        for kk in 0..MATMUL_I8_1024_32C_K {
            let av = a_host[r * MATMUL_I8_1024_32C_K + kk] as i32;
            if av == 0 {
                continue;
            }
            let row_base = r * MATMUL_I8_1024_32C_N;
            for c in 0..MATMUL_I8_1024_32C_N {
                let bv = b_host_cm[c * MATMUL_I8_1024_32C_K + kk] as i32;
                c_ref[row_base + c] += av * bv;
            }
        }
    }

    let mut errors = 0usize;
    let mut first_bad: Option<(usize, usize, i32, i32)> = None;
    for r in 0..MATMUL_I8_1024_32C_M {
        for c in 0..MATMUL_I8_1024_32C_N {
            let off = (r * MATMUL_I8_1024_32C_N + c) * 4;
            let bytes: [u8; 4] = outp[off..off + 4].try_into().unwrap();
            let got = i32::from_le_bytes(bytes);
            let want = c_ref[r * MATMUL_I8_1024_32C_N + c];
            if got != want {
                if first_bad.is_none() {
                    first_bad = Some((r, c, want, got));
                }
                errors += 1;
            }
        }
    }
    let total_elems = MATMUL_I8_1024_32C_M * MATMUL_I8_1024_32C_N;
    if errors == 0 {
        let macs = 2.0 * MATMUL_I8_1024_32C_M as f64
            * MATMUL_I8_1024_32C_K as f64
            * MATMUL_I8_1024_32C_N as f64;
        let gops = macs / (single_us as f64 / 1e6) / 1e9;
        println!(
            "[mm8_1024_32c] CORRECTNESS PASS - all {total_elems} output elements match"
        );
        println!(
            "[mm8_1024_32c] single dispatch: {single_us} us -> {gops:.2} GOp/s on {} GMACs",
            macs as u64 / 1_000_000_000
        );
    } else {
        println!(
            "[mm8_1024_32c] CORRECTNESS FAIL: {errors}/{total_elems} mismatches; first {first_bad:?}"
        );
        return ExitCode::FAILURE;
    }

    let n_iter: u32 = 30;
    let mut total_us: u128 = 0;
    let mut max_us: u128 = 0;
    for _ in 0..n_iter {
        {
            let cbuf = cmd_bo.map().expect("cmd reset");
            hipx::ert::reset_state(&mut cbuf[..4]);
        }
        let _ = cmd_bo.sync(SYNC_TO_DEVICE);
        let t = Instant::now();
        let seq = submit_exec_cmd(
            hipx.device.fd,
            &ctx,
            &[&cmd_bo],
            &[&instr_bo, &a_bo, &b_bo, &c_bo, &bo3_bo, &bo4_bo],
        )
        .expect("submit");
        timeline_wait(
            hipx.device.fd,
            ctx.syncobj_handle,
            seq,
            Duration::from_secs(30),
        )
        .expect("wait");
        let us = t.elapsed().as_micros();
        total_us += us;
        if us > max_us {
            max_us = us;
        }
    }
    let mean_us = total_us / n_iter as u128;
    let macs = 2.0 * MATMUL_I8_1024_32C_M as f64
        * MATMUL_I8_1024_32C_K as f64
        * MATMUL_I8_1024_32C_N as f64;
    let tops = macs / (mean_us as f64 / 1e6) / 1e12;
    println!(
        "[mm8_1024_32c] perf: mean={mean_us}us max={max_us}us -> {tops:.2} TOp/s ({n_iter} iters, {} GMACs)",
        (macs / 2.0 / 1e9) as u64
    );

    ExitCode::SUCCESS
}
