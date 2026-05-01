//! Third-kernel smoke test: passthrough_dmas. Single-column partition
//! that forwards 4096 × i32 (16 KiB) from input to output through a
//! MemTile DMA. No compute core involved — this is purely a DMA-
//! routing test, but it differs from passthrough_4k in *partition
//! shape* (1 column vs 8) and in *MemTile usage*.

use std::process::ExitCode;
use std::time::Duration;

use hipx::cmd::{config_cus, submit_exec_cmd};
use hipx::ert::ErtBuilder;
use hipx::fence::timeline_wait;
use hipx::hwctx::HwctxBuilder;
use hipx::ioctl::{SYNC_FROM_DEVICE, SYNC_TO_DEVICE};
use hipx::kernels::{
    passthrough_dmas_args as args, PASSTHROUGH_DMAS_COLUMNS, PASSTHROUGH_DMAS_INSTS,
    PASSTHROUGH_DMAS_OPS_PER_CYCLE, PASSTHROUGH_DMAS_PDI,
};
use hipx::Hipx;

const N_ELEMS: usize = 4096;
const BUF_BYTES: usize = N_ELEMS * 4; // i32

fn main() -> ExitCode {
    let hipx = Hipx::open().expect("Hipx::open");
    let mut b = HwctxBuilder::default();
    b.num_columns = PASSTHROUGH_DMAS_COLUMNS;
    b.max_opc = PASSTHROUGH_DMAS_OPS_PER_CYCLE;
    let ctx = hipx.create_hwctx(&b).expect("create_hwctx");
    println!("[ptd] hwctx={} cols={}", ctx.handle, PASSTHROUGH_DMAS_COLUMNS);

    let pdi_bo = hipx.alloc_dev(PASSTHROUGH_DMAS_PDI.len()).expect("pdi alloc");
    unsafe {
        let buf = hipx.dev_slice(&pdi_bo).expect("pdi slice");
        buf[..PASSTHROUGH_DMAS_PDI.len()].copy_from_slice(PASSTHROUGH_DMAS_PDI);
    }
    let _ = pdi_bo.sync(SYNC_TO_DEVICE);
    let _ = config_cus(hipx.device.fd, &ctx, vec![pdi_bo], &[0u8]).expect("config_cus");

    let instr_bo = hipx.alloc_dev(PASSTHROUGH_DMAS_INSTS.len()).expect("instr alloc");
    unsafe {
        let buf = hipx.dev_slice(&instr_bo).expect("instr slice");
        buf[..PASSTHROUGH_DMAS_INSTS.len()].copy_from_slice(PASSTHROUGH_DMAS_INSTS);
    }
    let _ = instr_bo.sync(SYNC_TO_DEVICE);
    let ninstr_dwords = (PASSTHROUGH_DMAS_INSTS.len() / 4) as u32;

    let mut input_bo = hipx.alloc_shmem(BUF_BYTES).expect("input alloc");
    {
        let buf = input_bo.map().expect("input map");
        for i in 0..N_ELEMS {
            let v = (i as i32).wrapping_mul(31337);
            buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let _ = input_bo.sync(SYNC_TO_DEVICE);
    let input_va = input_bo.host_ptr().unwrap() as u64;

    let mut output_bo = hipx.alloc_shmem(BUF_BYTES).expect("output alloc");
    {
        let buf = output_bo.map().expect("output map");
        for b in buf[..BUF_BYTES].iter_mut() { *b = 0xEE; }
    }
    let _ = output_bo.sync(SYNC_TO_DEVICE);
    let output_va = output_bo.host_ptr().unwrap() as u64;

    let mut cmd_bo = hipx.alloc_cmd(4096).expect("cmd alloc");
    {
        let cbuf = cmd_bo.map().expect("cmd map");
        let mut eb = ErtBuilder::new_start_cu(&mut cbuf[..256]);
        eb.set_cu_mask(0x1);
        eb.set_arg_u64(args::OPCODE, 3);
        eb.set_arg_u64(args::INSTR_PTR, instr_bo.xdna_addr);
        eb.set_arg_u32(args::NINSTR, ninstr_dwords);
        eb.set_arg_u64(args::INPUT, input_va);
        eb.set_arg_u64(args::OUTPUT, output_va);
        let _ = eb.finalize(0x3C);
    }
    let _ = cmd_bo.sync(SYNC_TO_DEVICE);

    let seq = submit_exec_cmd(
        hipx.device.fd, &ctx, &[&cmd_bo],
        &[&instr_bo, &input_bo, &output_bo],
    ).expect("submit");
    println!("[ptd] submitted seq={seq}");

    for point in [seq, seq + 1, seq.saturating_add(2)] {
        if timeline_wait(hipx.device.fd, ctx.syncobj_handle, point,
                         Duration::from_secs(5)).is_ok() { break; }
    }
    std::thread::sleep(Duration::from_millis(100));

    let _ = output_bo.sync(SYNC_FROM_DEVICE);
    let outp = output_bo.map().expect("re-map");

    let mut errors = 0;
    let mut first_bad = None;
    for i in 0..N_ELEMS {
        let want = (i as i32).wrapping_mul(31337);
        let got_bytes: [u8; 4] = outp[i * 4..i * 4 + 4].try_into().unwrap();
        let got = i32::from_le_bytes(got_bytes);
        if got != want {
            if first_bad.is_none() { first_bad = Some((i, want, got)); }
            errors += 1;
        }
    }
    if errors == 0 {
        println!("[ptd] PASS — {N_ELEMS} × i32 forwarded through MemTile DMA");
        ExitCode::SUCCESS
    } else {
        println!("[ptd] FAIL: {errors}/{N_ELEMS} mismatches; first {first_bad:?}");
        eprintln!("first 4 i32: {:?}",
                  (0..4).map(|i| {
                      let b: [u8;4] = outp[i*4..i*4+4].try_into().unwrap();
                      i32::from_le_bytes(b)
                  }).collect::<Vec<_>>());
        ExitCode::FAILURE
    }
}
