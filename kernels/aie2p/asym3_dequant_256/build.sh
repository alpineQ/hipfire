#!/usr/bin/env bash
# asym3_dequant_256 — hand-rolled native build.
#
# No Python, no IRON DSL. Calls only the native C++ binaries shipped
# in the mlir-aie install tree:
#
#   clang  (Peano AIE-2P backend)  — compile asym3_dequant_kernel.cc
#   aiecc  (native, C++)            — lower aie.mlir + link kernel into PDI
#
# Output artifacts in build/:
#   main.pdi            — Partial Device Image (firmware loads this)
#   insts.bin           — NPU instruction stream
#   asym3_dequant.xclbin — packaged form (for xrt-style consumers)
#
# Embedded into hipx via crates/hipx/src/kernels.rs after a successful
# build (see asym3_dequant_kernel binary placeholder).

set -euo pipefail

SRCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRCDIR"

# mlir-aie tree on hipx. Kernel includes (aie_api/aie.hpp) live here.
MLIR_AIE="${MLIR_AIE:-/home/kaden/mlir-aie}"
MLIR_AIE_INC="$MLIR_AIE/ironenv/lib/python3.12/site-packages/mlir_aie/include"
PEANO="$MLIR_AIE/ironenv/lib/python3.12/site-packages/llvm-aie"
AIECC_BIN="$MLIR_AIE/ironenv/lib/python3.12/site-packages/mlir_aie/bin/aiecc"

if [ ! -x "$AIECC_BIN" ]; then
    echo "ERROR: native aiecc not found at $AIECC_BIN" >&2
    echo "       (set MLIR_AIE to the mlir-aie source tree root)" >&2
    exit 1
fi
if [ ! -x "$PEANO/bin/clang" ]; then
    echo "ERROR: Peano clang not found at $PEANO/bin/clang" >&2
    exit 1
fi

mkdir -p build

# Step 1 — Compile the C++ kernel into an AIE-2P object file.
# Flags mirror PEANOWRAP2P_FLAGS from mlir-aie/programming_examples/makefile-common.
echo "[1/2] clang++ --target=aie2p-none-unknown-elf -std=c++20 -c asym3_dequant_kernel.cc"
"$PEANO/bin/clang++" \
    -O2 \
    -std=c++20 \
    --target=aie2p-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
    -DNDEBUG \
    -I "$MLIR_AIE_INC" \
    -c asym3_dequant_kernel.cc \
    -o build/asym3_dequant_kernel.o

# Step 2 — Lower aie.mlir + link kernel object → xclbin (containing PDI).
# Uses canonical aiecc invocation matching mlir-aie programming examples.
# The xclbin is a zip; PDI extracts via a follow-up unzip.
echo "[2/2] aiecc --aie-generate-xclbin --aie-generate-npu-insts aie.mlir"
cd build
# aiecc shells out to Peano's opt/llc to compile the per-tile MLIR
# down to AIE-2P object code. Point it at the Peano install (same
# tree we used for the C++ kernel compile in step 1).
PEANO_INSTALL_DIR="$PEANO" \
"$AIECC_BIN" \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc --no-xbridge \
    --peano="$PEANO" \
    --xclbin-name=asym3_dequant.xclbin \
    --npu-insts-name=insts.bin \
    "$SRCDIR/aie.mlir"

# Extract PDI + JSON manifests from the xclbin (zip format) so they
# can be embedded directly via crates/hipx/src/kernels.rs.
echo
echo "[2.5] unzip PDI + JSON from xclbin"
unzip -o asym3_dequant.xclbin -d _unpacked >/dev/null
cp _unpacked/main.pdi main.pdi 2>/dev/null || true
cp _unpacked/main_aie_partition.json main_aie_partition.json 2>/dev/null || true
cp _unpacked/main_kernels.json main_kernels.json 2>/dev/null || true

echo
echo "Built artifacts:"
ls -la main.pdi insts.bin asym3_dequant.xclbin 2>/dev/null || true
