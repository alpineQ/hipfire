//! ERT (Embedded Runtime Toolkit) command packet format.
//!
//! When EXEC_CMD submits a CMD BO, the firmware reads it as an
//! `ert_start_kernel_cmd` packet from XRT's `ert.h`. We need to build
//! the same packet shape from Rust without depending on libxrt.
//!
//! For ERT_START_NPU the layout is:
//!   header      [1 dword] state(4):stat(1):unused(5):ext_masks(2):count(11):opcode(5):type(4)
//!   cu_mask     [1 dword] mandatory CU mask (bit N → CU index N)
//!   npu_data    [4 dwords] ert_npu_data: instruction_buffer (u64),
//!                                       instruction_buffer_size (u32),
//!                                       instruction_prop_count (u32)
//!   kernel args [N dwords] per-kernel argspace, layout per
//!                                       `main_kernels.json`
//!
//! `count` is `(total payload dwords - 1)` — i.e. number of dwords
//! after the header, since ert.h uses count for that meaning.

/// ert_cmd_state. Set to NEW before submission; firmware writes
/// COMPLETED (or one of the error states) when done.
pub const ERT_CMD_STATE_NEW: u32 = 1;
pub const ERT_CMD_STATE_COMPLETED: u32 = 4;
pub const ERT_CMD_STATE_ERROR: u32 = 5;
pub const ERT_CMD_STATE_ABORT: u32 = 6;
pub const ERT_CMD_STATE_TIMEOUT: u32 = 7;

/// ert_cmd_opcode (subset relevant to NPU).
pub const ERT_START_CU: u32 = 0;
pub const ERT_START_DPU: u32 = 18;
pub const ERT_START_NPU: u32 = 20;

/// ert_cmd_type. ERT_CU = 3 for CU-targeted commands.
pub const ERT_CU_TYPE: u32 = 3;

/// Build the 32-bit ERT command header.
///
/// `count` is the number of payload dwords after the header (so
/// `header + (count + 1)` total dwords in packet, including header).
pub const fn build_header(opcode: u32, cmd_type: u32, count: u32) -> u32 {
    let state = ERT_CMD_STATE_NEW & 0xF;
    let stat_enabled = 0u32; // bit 4
    let unused = 0u32; // bits 5-9
    let extra_cu_masks = 0u32; // bits 10-11
    let count_field = count & 0x7FF; // bits 12-22
    let opcode_field = opcode & 0x1F; // bits 23-27
    let type_field = cmd_type & 0xF; // bits 28-31
    state
        | (stat_enabled << 4)
        | (unused << 5)
        | (extra_cu_masks << 10)
        | (count_field << 12)
        | (opcode_field << 23)
        | (type_field << 28)
}

/// Read just the `state` nibble from a header word. Useful to tell
/// whether the firmware has marked completion (state == COMPLETED).
pub const fn header_state(header: u32) -> u32 {
    header & 0xF
}

/// `ert_npu_data` — 4 dwords after `cu_mask`. Tells firmware where
/// the NPU instruction stream lives.
#[repr(C, packed)]
#[derive(Default, Debug, Clone, Copy)]
pub struct NpuData {
    pub instruction_buffer: u64,
    pub instruction_buffer_size: u32,
    pub instruction_prop_count: u32,
}

const _ASSERT_NPU_DATA_SIZE: [u8; 16] = [0u8; std::mem::size_of::<NpuData>()];

/// Builder for ert_start_kernel_cmd buffers. Writes into a caller-
/// supplied byte slice; pass that slice to a CMD BO mmap'd region.
pub struct ErtBuilder<'b> {
    buf: &'b mut [u8],
    cursor: usize, // dword cursor (in 4-byte units)
    cu_mask_set: bool,
    npu_data_set: bool,
    arg_offset: usize, // byte offset where kernel args start
}

impl<'b> ErtBuilder<'b> {
    /// Begin a new ERT_START_NPU packet in `buf`. Buffer should be
    /// at least 256 bytes (typical CMD BO is 4 KiB so this is safe).
    /// Header + cu_mask + npu_data are written immediately; kernel
    /// args go in via `set_arg_*`. Call `finalize()` to patch the
    /// final `count` field.
    pub fn new_start_npu(buf: &'b mut [u8]) -> Self {
        // Zero the slice first so unused arg slots are deterministic.
        for b in buf.iter_mut() {
            *b = 0;
        }
        // Write a placeholder header — count gets fixed in finalize.
        let placeholder_header = build_header(ERT_START_NPU, ERT_CU_TYPE, 0);
        buf[0..4].copy_from_slice(&placeholder_header.to_le_bytes());
        Self {
            buf,
            cursor: 1, // dwords past header
            cu_mask_set: false,
            npu_data_set: false,
            arg_offset: 0,
        }
    }

    /// Set the mandatory CU mask. Bit N corresponds to CU index N.
    /// For a single CU bound at index 0, pass `0x1`.
    pub fn set_cu_mask(&mut self, mask: u32) -> &mut Self {
        let off = 4; // dword 1
        self.buf[off..off + 4].copy_from_slice(&mask.to_le_bytes());
        if !self.cu_mask_set {
            self.cursor += 1;
        }
        self.cu_mask_set = true;
        self
    }

    /// Write the ert_npu_data prefix. Must be called once for
    /// ERT_START_NPU. After this, the cursor is positioned at the
    /// start of kernel arg space.
    pub fn set_npu_data(&mut self, instr_addr: u64, instr_bytes: u32) -> &mut Self {
        assert!(
            self.cu_mask_set,
            "set_cu_mask must be called before set_npu_data"
        );
        let off = self.cursor * 4;
        self.buf[off..off + 8].copy_from_slice(&instr_addr.to_le_bytes());
        self.buf[off + 8..off + 12].copy_from_slice(&instr_bytes.to_le_bytes());
        self.buf[off + 12..off + 16].copy_from_slice(&0u32.to_le_bytes());
        self.cursor += 4;
        self.arg_offset = self.cursor * 4;
        self.npu_data_set = true;
        self
    }

    /// Write a u32 kernel arg at the given byte offset within the
    /// kernel arg space (offsets come from `main_kernels.json`).
    pub fn set_arg_u32(&mut self, arg_offset_in_argspace: usize, value: u32) -> &mut Self {
        let off = self.arg_offset + arg_offset_in_argspace;
        self.buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
        self
    }

    /// Write a u64 kernel arg. Layout is little-endian, packed (4-byte
    /// aligned, not 8-byte) — consistent with how aiecc emits arg
    /// offsets in `main_kernels.json`.
    pub fn set_arg_u64(&mut self, arg_offset_in_argspace: usize, value: u64) -> &mut Self {
        let off = self.arg_offset + arg_offset_in_argspace;
        self.buf[off..off + 8].copy_from_slice(&value.to_le_bytes());
        self
    }

    /// Finalize the packet: patch the `count` field of the header
    /// based on `arg_space_bytes`. `count` = total payload dwords
    /// after header - 1, per ert.h convention.
    pub fn finalize(&mut self, arg_space_bytes: usize) -> usize {
        let total_payload_dwords =
            (self.arg_offset + arg_space_bytes) / 4 - 1; // minus the header dword itself? no — the dwords AFTER header
        // count is "(total dwords) - 1" i.e. arg + npu_data + cu_mask
        let total_dwords_after_header = (self.arg_offset + arg_space_bytes) / 4 - 1;
        // ^ above the header is at dword 0. After-header runs from
        //   dword 1 to (arg_offset + args)/4 - 1 inclusive. That count.
        let count = total_dwords_after_header as u32;
        let header = build_header(ERT_START_NPU, ERT_CU_TYPE, count);
        self.buf[0..4].copy_from_slice(&header.to_le_bytes());
        let _ = total_payload_dwords;
        (self.arg_offset + arg_space_bytes) // total bytes written
    }
}
