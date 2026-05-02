//! iGPU INT8 GEMM throughput at production prefill shapes.
//!
//! Measures `gemm_hfq4g256_mmq_set` (Q8_1 activations x HFQ4 weights, i8 WMMA
//! accumulate) on shapes representative of Qwen3-27B prefill:
//!   - 4608 x 4608 : q/k/v/o projections
//!   - 36864 x 4608 : fused FFN gate+up
//!   - 4608 x 18432 : FFN down
//! Batch sizes span the MMQ threshold (32 below, 128/256/512 above).
//!
//! Reports effective TOp/s (2 * M * K * batch / time). This is the iGPU
//! ground floor for the prefill viability decision on Strix Halo (gfx1151).
//!
//! Build:
//!   cargo build --release --example bench_int8_gemm_prefill
//! Run:
//!   ./target/release/examples/bench_int8_gemm_prefill
//!
//! Output is also written to bench/prefill-igpu-int8-<ts>.txt by the caller
//! (this binary just emits to stderr/stdout in a parseable format).

use std::time::SystemTime;

fn make_hfq4g256_weights(m: usize, k: usize) -> Vec<u8> {
    // HFQ4-G256: 136 bytes per 256 elements. Per-group:
    //   [4B f32 scale][4B f32 zero][128B nibbles (256 4-bit values)]
    assert!(k % 256 == 0, "k must be 256-multiple, got {k}");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    let total = m * row_bytes;
    let mut data = vec![0u8; total];
    for r in 0..m {
        for g in 0..groups_per_row {
            let off = r * row_bytes + g * 136;
            data[off..off + 4].copy_from_slice(&0.01f32.to_le_bytes());
            data[off + 4..off + 8].copy_from_slice(&(-0.005f32).to_le_bytes());
            // Nibbles: alternating 0x53 (== 3 and 5).
            for i in 0..128 {
                data[off + 8 + i] = 0x53;
            }
        }
    }
    data
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    let arch = gpu.hip.get_arch(0).unwrap_or_else(|_| "unknown".to_string());
    eprintln!("# bench_int8_gemm_prefill");
    eprintln!("# arch: {arch}");
    eprintln!("# timestamp: {}", SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0));
    eprintln!();
    eprintln!("# columns: shape | batch | iters | ms_total | ms_per_op | tops");
    eprintln!();

    let shapes: &[(usize, usize, &str)] = &[
        (4608, 4608, "27b-attn-proj-4608x4608"),
        (4608, 18432, "27b-ffn-down-4608x18432"),
        (36864, 4608, "27b-ffn-gate+up-36864x4608"),
    ];
    let batches: &[usize] = &[32, 64, 128, 256, 512];
    let n_iters: usize = 100;
    let warmup: usize = 10;

    for &(m, k, name) in shapes {
        let weights = make_hfq4g256_weights(m, k);
        let d_a = gpu.upload_raw(&weights, &[weights.len()]).unwrap();

        for &bs in batches {
            // Activation x: random-ish f32 [batch, k]
            let x_data: Vec<f32> = (0..bs * k).map(|i| 0.001 * (i % 251) as f32).collect();
            let d_x = gpu.upload_f32(&x_data, &[bs, k]).unwrap();
            let d_y = gpu.zeros(&[bs, m], rdna_compute::DType::F32).unwrap();

            // Warmup
            for _ in 0..warmup {
                gpu.gemm_hfq4g256_mmq_set(&d_a, &d_x, &d_y, m, k, bs).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();

            let start = gpu.hip.event_create().unwrap();
            let stop = gpu.hip.event_create().unwrap();

            gpu.hip.event_record(&start, None).unwrap();
            for _ in 0..n_iters {
                gpu.gemm_hfq4g256_mmq_set(&d_a, &d_x, &d_y, m, k, bs).unwrap();
            }
            gpu.hip.event_record(&stop, None).unwrap();
            gpu.hip.event_synchronize(&stop).unwrap();
            let ms_total = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
            let ms_per = ms_total / n_iters as f64;
            let ops_per_iter = 2.0 * m as f64 * k as f64 * bs as f64;
            let tops = ops_per_iter / (ms_per * 1.0e-3) / 1.0e12;

            println!(
                "{name:30} | bs={bs:4} | iters={n_iters} | ms_total={ms_total:8.3} | ms/op={ms_per:8.4} | TOp/s={tops:6.2}"
            );

            gpu.free_tensor(d_x).unwrap();
            gpu.free_tensor(d_y).unwrap();
        }

        gpu.free_tensor(d_a).unwrap();
        eprintln!();
    }
}
