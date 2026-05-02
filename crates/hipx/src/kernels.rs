//! Embedded PDI artifacts. Analog of `rdna-compute::kernels` for the
//! NPU side.
//!
//! Each kernel is a triple `(pdi_bytes, npu_insts_bytes,
//! kernel_meta)` produced by MLIR-AIE → Peano. PDI is the AIE tile
//! program loaded as a CU; npu_insts is the NPU instruction stream
//! (DMA + CU launch micro-ops) the firmware executes; kernel_meta
//! describes argument layout for filling the EXEC_CMD packet.
//!
//! See `kernels/aie2p/<name>/build/` for the source artifacts and
//! `kernels/aie2p/README.md` for the build flow.

/// 4 KB byte-passthrough — the Phase 1.3 syncobj round-trip target.
/// Reads `4096 × u8` from input BO, writes the same to output BO.
/// Uses 1 column, 1 core tile. PDI 2 KB, instruction stream 300 B.
pub const PASSTHROUGH_4K_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/passthrough_4k/build/main.pdi");

pub const PASSTHROUGH_4K_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/passthrough_4k/build/passthrough_insts.bin");

/// DPU kernel ID assigned by aiecc for `MLIR_AIE`. Sent to firmware
/// in the EXEC_CMD packet to identify which CU to launch.
pub const PASSTHROUGH_4K_KERNEL_ID: u32 = 0x901;

/// Operations per cycle declared in the AIE partition manifest. Used
/// as the `max_opc` hint for CREATE_HWCTX.
pub const PASSTHROUGH_4K_OPS_PER_CYCLE: u32 = 2048;

/// Number of AIE columns the partition uses. Used as `num_columns`
/// for HwctxBuilder.
pub const PASSTHROUGH_4K_COLUMNS: u32 = 8;

/// Argument layout (bytes) inside the EXEC_CMD payload. Matches the
/// `ps-kernels[0].arguments` array in `main_kernels.json`.
pub mod passthrough_4k_args {
    pub const OPCODE: usize = 0x00; // u64
    pub const INSTR_PTR: usize = 0x08; // void*
    pub const NINSTR: usize = 0x10; // u32
    pub const BO0: usize = 0x14; // void*
    pub const BO1: usize = 0x1C;
    pub const BO2: usize = 0x24;
    pub const BO3: usize = 0x2C;
    pub const BO4: usize = 0x34;
}

/// vector_scalar_mul int16 — second proof kernel showing real
/// compute (not just passthrough). Multiplies `4096 × i16` by an
/// `i32` scalar, writes 4096 × i16. Single column, single core
/// tile. PDI 3024 B, instruction stream 420 B.
pub const VEC_SCALAR_MUL_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/vec_scalar_mul/build/main.pdi");

pub const VEC_SCALAR_MUL_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/vec_scalar_mul/build/insts_in1_size.bin");

pub const VEC_SCALAR_MUL_KERNEL_ID: u32 = 0x901;
pub const VEC_SCALAR_MUL_OPS_PER_CYCLE: u32 = 2048;
pub const VEC_SCALAR_MUL_COLUMNS: u32 = 8;

/// Same 8-arg DPU layout as passthrough — 5 BO slots after the
/// header (opcode, instr_ptr, ninstr).
pub mod vec_scalar_mul_args {
    pub use super::passthrough_4k_args::*;
    pub const INPUT: usize = BO0;     // 8192 B = 4096 × i16
    pub const SCALE: usize = BO1;     // 4 B = 1 × i32
    pub const OUTPUT: usize = BO2;    // 8192 B = 4096 × i16
}

/// passthrough_dmas — single-column DMA-only forwarding kernel.
/// Differs from passthrough_4k by using a MemTile to forward
/// 4096 × i32 (16 KiB) from input to output via DMA. Single column
/// (column_width=1) instead of full-array (8 cols), so it tests
/// that hipx hwctx works for both partition shapes.
pub const PASSTHROUGH_DMAS_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/passthrough_dmas/build/main.pdi");

pub const PASSTHROUGH_DMAS_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/passthrough_dmas/build/ptd_insts.bin");

pub const PASSTHROUGH_DMAS_KERNEL_ID: u32 = 0x901;
pub const PASSTHROUGH_DMAS_OPS_PER_CYCLE: u32 = 2048;
/// Single-column partition. start_columns = [1, 2, 3, 4] in the
/// manifest; firmware picks one when the hwctx is created.
pub const PASSTHROUGH_DMAS_COLUMNS: u32 = 1;

pub mod passthrough_dmas_args {
    pub use super::passthrough_4k_args::*;
    pub const INPUT: usize = BO0;     // 16384 B = 4096 × i32
    pub const UNUSED1: usize = BO1;   // unused (rt.sequence has _ placeholder)
    pub const OUTPUT: usize = BO2;    // 16384 B = 4096 × i32
}

/// matrix_vector i16xi16→i32 GEMV — third compute proof kernel.
/// Computes c = A · b where A is M×K i16, b is K i16, c is M i32.
/// M=K=288, N=1, tile m=k=32. Full 8-column partition. This is the
/// first "real GEMM-class" kernel and the building block for both
/// INT8 spec-decode draft heads and the asym KV-codec dequant fold.
///
/// Per the kernels.json manifest, the layout is:
/// - bo0 = A matrix (M*K*sizeof(i16) = 288*288*2 = 165 888 B)
/// - bo1 = b vector (K*sizeof(i16) = 576 B)
/// - bo2 = c output (M*sizeof(i32) = 1152 B)
/// - bo3, bo4 = ctrlpkts/trace placeholders
pub const MATVEC_288X288_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matvec_288x288x1/build/main.pdi");

pub const MATVEC_288X288_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matvec_288x288x1/build/insts.bin");

pub const MATVEC_288X288_KERNEL_ID: u32 = 0x901;
pub const MATVEC_288X288_OPS_PER_CYCLE: u32 = 2048;
pub const MATVEC_288X288_COLUMNS: u32 = 8;
pub const MATVEC_288X288_M: usize = 288;
pub const MATVEC_288X288_K: usize = 288;

pub mod matvec_288x288_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;       // 165 888 B  (M*K*i16)
    pub const B: usize = BO1;       //     576 B  (K*i16)
    pub const C: usize = BO2;       //   1 152 B  (M*i32)
}

/// 4-core whole-array i16xi16→i32 matmul, M=K=N=512, tile m=k=n=32.
/// Uses 4 AIE columns (`n_aie_cols=4`) so it pulls the full Strix Halo
/// AIE array into a single dispatch. C = A · B over 16M MACs per call,
/// vs matvec_288x288's 165K MACs — a 100× compute fraction shift.
pub const MATMUL_512_4C_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_512x512x512_4c/build/main.pdi");

pub const MATMUL_512_4C_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_512x512x512_4c/build/insts.bin");

pub const MATMUL_512_4C_KERNEL_ID: u32 = 0x901;
pub const MATMUL_512_4C_OPS_PER_CYCLE: u32 = 2048;
pub const MATMUL_512_4C_COLUMNS: u32 = 8;
pub const MATMUL_512_4C_M: usize = 512;
pub const MATMUL_512_4C_K: usize = 512;
pub const MATMUL_512_4C_N: usize = 512;

pub mod matmul_512_4c_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;     // 524 288 B  (M*K*i16)
    pub const B: usize = BO1;     // 524 288 B  (K*N*i16)
    pub const C: usize = BO2;     // 1 048 576 B (M*N*i32)
}

/// 4-core whole-array i8×i8→i32 matmul, M=K=N=512, tile m=k=n=64.
/// Same shape as MATMUL_512_4C but at i8 precision — half the BO
/// volume on inputs, double the AIE MAC throughput per cycle.
pub const MATMUL_I8_512_4C_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_i8_512_4c/build/main.pdi");

pub const MATMUL_I8_512_4C_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_i8_512_4c/build/insts.bin");

pub const MATMUL_I8_512_4C_KERNEL_ID: u32 = 0x901;
pub const MATMUL_I8_512_4C_OPS_PER_CYCLE: u32 = 2048;
pub const MATMUL_I8_512_4C_COLUMNS: u32 = 8;
pub const MATMUL_I8_512_4C_M: usize = 512;
pub const MATMUL_I8_512_4C_K: usize = 512;
pub const MATMUL_I8_512_4C_N: usize = 512;

pub mod matmul_i8_512_4c_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;     // 262 144 B  (M*K*i8)
    pub const B: usize = BO1;     // 262 144 B  (K*N*i8)
    pub const C: usize = BO2;     // 1 048 576 B (M*N*i32)
}

/// 4-core whole-array i8×i8→i32 matmul, M=K=N=1024, tile m=k=n=64.
/// 8× more MACs than MATMUL_I8_512_4C (2.1 GMACs vs 268 MMACs), so
/// the dispatch overhead amortizes 8× — closer to the AIE peak.
pub const MATMUL_I8_1024_4C_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_i8_1024_4c/build/main.pdi");

pub const MATMUL_I8_1024_4C_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_i8_1024_4c/build/insts.bin");

pub const MATMUL_I8_1024_4C_KERNEL_ID: u32 = 0x901;
pub const MATMUL_I8_1024_4C_OPS_PER_CYCLE: u32 = 2048;
pub const MATMUL_I8_1024_4C_COLUMNS: u32 = 8;
pub const MATMUL_I8_1024_4C_M: usize = 1024;
pub const MATMUL_I8_1024_4C_K: usize = 1024;
pub const MATMUL_I8_1024_4C_N: usize = 1024;

pub mod matmul_i8_1024_4c_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;     // 1 048 576 B  (M*K*i8)
    pub const B: usize = BO1;     // 1 048 576 B  (K*N*i8)
    pub const C: usize = BO2;     // 4 194 304 B  (M*N*i32)
}

/// 4-core whole-array i8×i8→i32 matmul, M=K=N=2048, tile m=k=n=64.
/// 8× more MACs than MATMUL_I8_1024_4C (17 GMACs vs 2.1), so the
/// dispatch overhead amortizes further — useful as the fully-saturating
/// shape for steady-state perf measurement.
pub const MATMUL_I8_2048_4C_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_i8_2048_4c/build/main.pdi");

pub const MATMUL_I8_2048_4C_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_i8_2048_4c/build/insts.bin");

pub const MATMUL_I8_2048_4C_KERNEL_ID: u32 = 0x901;
pub const MATMUL_I8_2048_4C_OPS_PER_CYCLE: u32 = 2048;
pub const MATMUL_I8_2048_4C_COLUMNS: u32 = 8;
pub const MATMUL_I8_2048_4C_M: usize = 2048;
pub const MATMUL_I8_2048_4C_K: usize = 2048;
pub const MATMUL_I8_2048_4C_N: usize = 2048;

pub mod matmul_i8_2048_4c_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;     // 4 194 304 B   (M*K*i8)
    pub const B: usize = BO1;     // 4 194 304 B   (K*N*i8)
    pub const C: usize = BO2;     // 16 777 216 B  (M*N*i32 = 16 MiB)
}

/// 4-core whole-array bf16×bf16→f32 matmul, M=K=N=512, tile 32×32×32.
/// The natural-precision shape for LLM workloads — FP16 hidden states
/// and FP16 weights map directly into bf16 (lossy but production-typical
/// after rounding). A and B are bf16 (2 bytes each), C is f32.
pub const MATMUL_BF16_512_4C_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_bf16_512_4c/build/main.pdi");

pub const MATMUL_BF16_512_4C_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_bf16_512_4c/build/insts.bin");

pub const MATMUL_BF16_512_4C_KERNEL_ID: u32 = 0x901;
pub const MATMUL_BF16_512_4C_OPS_PER_CYCLE: u32 = 2048;
pub const MATMUL_BF16_512_4C_COLUMNS: u32 = 8;
pub const MATMUL_BF16_512_4C_M: usize = 512;
pub const MATMUL_BF16_512_4C_K: usize = 512;
pub const MATMUL_BF16_512_4C_N: usize = 512;

pub mod matmul_bf16_512_4c_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;     // 524 288 B   (M*K*bf16)
    pub const B: usize = BO1;     // 524 288 B   (K*N*bf16)
    pub const C: usize = BO2;     // 1 048 576 B (M*N*f32)
}

/// 4-core whole-array bf16×bf16→f32 matmul, M=K=N=1024, tile 32×32×32.
/// 8× more MACs than the 512^3 BF16 — moves the dispatch overhead
/// fraction from ~50% to ~10%, approaching the per-shape ceiling.
pub const MATMUL_BF16_1024_4C_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_bf16_1024_4c/build/main.pdi");

pub const MATMUL_BF16_1024_4C_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/matmul_bf16_1024_4c/build/insts.bin");

pub const MATMUL_BF16_1024_4C_KERNEL_ID: u32 = 0x901;
pub const MATMUL_BF16_1024_4C_OPS_PER_CYCLE: u32 = 2048;
pub const MATMUL_BF16_1024_4C_COLUMNS: u32 = 8;
pub const MATMUL_BF16_1024_4C_M: usize = 1024;
pub const MATMUL_BF16_1024_4C_K: usize = 1024;
pub const MATMUL_BF16_1024_4C_N: usize = 1024;

pub mod matmul_bf16_1024_4c_args {
    pub use super::passthrough_4k_args::*;
    pub const A: usize = BO0;     // 2 097 152 B   (M*K*bf16)
    pub const B: usize = BO1;     // 2 097 152 B   (K*N*bf16)
    pub const C: usize = BO2;     // 4 194 304 B   (M*N*f32)
}

/// asym3_dequant_256 — single-head asym3 K cache dequant kernel for
/// 27B Gemma (head_dim=256). Maps `[4-byte cnorm | 96-byte packed
/// 3-bit indices]` from `kernels/src/turbo_common.h::TURBO_C3_256`
/// to 256 bf16 values per call. Single-core single-column.
///
/// Stage 1.1 verified to ULP-bounded (<= 2 bf16 ULP per element)
/// against an AIE-2P-shape CPU reference (RTZ cnorm, RAZ output).
/// See docs/plans/aie2p-bf16-mul-shape.md for the hardware-shape
/// characterization.
pub const ASYM3_DEQUANT_256_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/asym3_dequant_256/build/main.pdi");

pub const ASYM3_DEQUANT_256_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/asym3_dequant_256/build/insts.bin");

pub const ASYM3_DEQUANT_256_KERNEL_ID: u32 = 0x901;
pub const ASYM3_DEQUANT_256_OPS_PER_CYCLE: u32 = 2048;
pub const ASYM3_DEQUANT_256_COLUMNS: u32 = 8;
pub const ASYM3_DEQUANT_256_HEAD_DIM: usize = 256;
pub const ASYM3_DEQUANT_256_PACKED_BYTES: usize = 96;
pub const ASYM3_DEQUANT_256_OUT_BYTES: usize = 512;

pub mod asym3_dequant_256_args {
    pub use super::passthrough_4k_args::*;
    pub const PACKED: usize = BO0;  // 96 B   (32 threads x 3 bytes packed indices)
    pub const CNORM: usize = BO1;   // 4 B    (one f32 magnitude factor)
    pub const OUT: usize = BO2;     // 512 B  (256 bf16 dequanted K elements)
}

/// asym3_dequant_layer — per-layer batched asym3 dequant. Stage 1.4
/// MVP variant. Single dispatch covers `N_ITERS` (head, position)
/// pairs sharing the kernel binary; compute is identical to
/// `asym3_dequant_256` per iteration. Hardcoded N_ITERS=32 for the
/// MVP iteration; scale up + multi-core fan-out follows.
///
/// Layout per dispatch:
///   packed: N_ITERS * 96 bytes  (indices, contiguous per iter)
///   cnorm:  N_ITERS * 4  bytes  (one f32 per iter)
///   out:    N_ITERS * 512 bytes (256 bf16 per iter)
pub const ASYM3_DEQUANT_LAYER_PDI: &[u8] =
    include_bytes!("../../../kernels/aie2p/asym3_dequant_layer/build/main.pdi");

pub const ASYM3_DEQUANT_LAYER_INSTS: &[u8] =
    include_bytes!("../../../kernels/aie2p/asym3_dequant_layer/build/insts.bin");

pub const ASYM3_DEQUANT_LAYER_KERNEL_ID: u32 = 0x902;
pub const ASYM3_DEQUANT_LAYER_OPS_PER_CYCLE: u32 = 2048;
pub const ASYM3_DEQUANT_LAYER_COLUMNS: u32 = 8;
pub const ASYM3_DEQUANT_LAYER_HEAD_DIM: usize = 256;
pub const ASYM3_DEQUANT_LAYER_N_ITERS: usize = 32;
pub const ASYM3_DEQUANT_LAYER_PACKED_BYTES: usize =
    ASYM3_DEQUANT_LAYER_N_ITERS * 96;
pub const ASYM3_DEQUANT_LAYER_CNORM_BYTES: usize =
    ASYM3_DEQUANT_LAYER_N_ITERS * 4;
pub const ASYM3_DEQUANT_LAYER_OUT_BYTES: usize =
    ASYM3_DEQUANT_LAYER_N_ITERS * 512;

pub mod asym3_dequant_layer_args {
    pub use super::passthrough_4k_args::*;
    pub const PACKED: usize = BO0;  // N_ITERS * 96 B    (packed 3-bit indices)
    pub const CNORM: usize = BO1;   // N_ITERS * 4  B    (per-iter f32 cnorm)
    pub const OUT: usize = BO2;     // N_ITERS * 512 B   (per-iter 256 bf16)
}
