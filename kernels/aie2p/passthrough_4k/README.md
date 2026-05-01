# passthrough_4k — first AIE-2P kernel

Reads `4096 × u8` from input, writes the same to output. Single
column, single core tile, single mem tile, two shim tiles for
input/output DMA.

This is the Phase 1.3 syncobj round-trip target: smallest possible
"valid CU" we can dispatch to prove the EXEC_CMD path end-to-end.

## Provenance

`source.py` is a verbatim copy of
`mlir-aie/programming_examples/basic/passthrough_pykernel/passthrough_pykernel.py`
(Apache-2.0). Kept here so we have a self-contained build flow.

## Build (re-run when source.py changes)

Prerequisites on hipx:
- `~/mlir-aie/ironenv/` — Python 3.12 venv with mlir_aie + llvm-aie
  wheels installed (see `scripts/setup-hipx-npu.sh`).
- OOT `xrt_plugin-amdxdna` DKMS module loaded (`amdxdna` v1.0.0+).

```bash
ssh hipx '
  export PATH="$HOME/.local/bin:$PATH"
  export VIRTUAL_ENV="$HOME/mlir-aie/ironenv"
  export MLIR_AIE_INSTALL_DIR="$VIRTUAL_ENV/lib/python3.12/site-packages/mlir_aie"
  export PEANO_INSTALL_DIR="$VIRTUAL_ENV/lib/python3.12/site-packages/llvm-aie"
  export PATH="$VIRTUAL_ENV/bin:$MLIR_AIE_INSTALL_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH"
  export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python"
  export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:$PEANO_INSTALL_DIR/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"
  cd /tmp && rm -rf passthrough_build && mkdir -p passthrough_build && cd passthrough_build
  python <source.py> 4096 npu2 > aie.mlir
  aiecc --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
    --no-xchesscc --no-xbridge \
    --xclbin-name=passthrough.xclbin --npu-insts-name=passthrough_insts.bin \
    aie.mlir
'
```

Then `scp` `aie.mlir.prj/main.pdi`, `passthrough_insts.bin`, and
`passthrough.xclbin` back to `build/` here.

## Ground truth (XRT-side, AMD reference)

```
$ ulimit -l unlimited
$ ./passthrough_pykernel -x passthrough.xclbin -i passthrough_insts.bin -k MLIR_AIE
Running...
PASS!
```

Verified 2026-05-01 on hipx. This is the dispatch we want to
replicate via the hipx Rust path (no XRT runtime dependency).

## Kernel metadata (from `main_kernels.json`)

- DPU kernel ID: `0x901`
- Instance: `MLIRAIE`
- Operations/cycle: 2048
- AIE partition: 1 PDI, 8 columns starting at column 0
- Argument layout in EXEC_CMD payload (`crates/hipx/src/kernels.rs`):
  - 0x00: opcode (u64)
  - 0x08: instr pointer (SRAM, the npu_insts BO's xdna_addr)
  - 0x10: ninstr (u32, count of npu_insts dwords)
  - 0x14, 0x1C, 0x24, 0x2C, 0x34: bo0..bo4 host pointers
