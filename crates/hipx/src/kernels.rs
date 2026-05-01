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
