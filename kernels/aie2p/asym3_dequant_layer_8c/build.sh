#!/usr/bin/env bash
# asym3_dequant_layer_8c - hand-rolled native build (2-core MVP).
# Same toolchain as asym3_dequant_layer.

set -euo pipefail

SRCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRCDIR"

MLIR_AIE="${MLIR_AIE:-/home/kaden/mlir-aie}"
MLIR_AIE_INC="$MLIR_AIE/ironenv/lib/python3.12/site-packages/mlir_aie/include"
PEANO="$MLIR_AIE/ironenv/lib/python3.12/site-packages/llvm-aie"
AIECC_BIN="$MLIR_AIE/ironenv/lib/python3.12/site-packages/mlir_aie/bin/aiecc"

if [ ! -x "$AIECC_BIN" ]; then
    echo "ERROR: native aiecc not found at $AIECC_BIN" >&2
    exit 1
fi
if [ ! -x "$PEANO/bin/clang" ]; then
    echo "ERROR: Peano clang not found at $PEANO/bin/clang" >&2
    exit 1
fi

mkdir -p build

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

echo "[2/2] aiecc --aie-generate-xclbin --aie-generate-npu-insts aie.mlir"
cd build
PEANO_INSTALL_DIR="$PEANO" \
"$AIECC_BIN" \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc --no-xbridge \
    --peano="$PEANO" \
    --xclbin-name=asym3_dequant_layer_8c.xclbin \
    --npu-insts-name=insts.bin \
    "$SRCDIR/aie.mlir"

echo
echo "[2.5] copy main.pdi + manifests"
cp aie.mlir.prj/main.pdi main.pdi
cp aie.mlir.prj/main_aie_partition.json main_aie_partition.json 2>/dev/null || true
cp aie.mlir.prj/main_kernels.json main_kernels.json 2>/dev/null || true

echo
echo "Built artifacts:"
ls -la main.pdi insts.bin asym3_dequant_layer_8c.xclbin 2>/dev/null || true
