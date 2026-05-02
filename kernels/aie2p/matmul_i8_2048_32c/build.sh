#!/usr/bin/env bash
# matmul_i8_2048_32c - hand-rolled native build for 8c x 4-row = 32-tile
# i8 GEMM at M=K=N=2048 with tile (m, k, n) = (64, 64, 32) and
# COLUMN-MAJOR B layout.
#
# MLIR was generated once via mlir-aie's whole_array_placed.py with
# --dev=npu2 -M 2048 -K 2048 -N 2048 -m 64 -k 64 -n 32 --n-aie-cols 8
# --b-col-maj 1 --dtype_in i8 --dtype_out i32 - see aie.mlir in this dir.
#
# Why b-col-maj: row-major B at M=K=N=2048 with our default tile sizes
# emits a DMA descriptor with size=1024 in the K dim (max 1023). The
# col-major B path uses a different DMA pattern that fits inside the
# limit. The host bench must lay out B as B[c, k] = B[c * K + k].
#
# mm.cc + zero.cc copied from mlir-aie/aie_kernels/aie2p/. The kernel
# C++ checks B_COL_MAJ to swap its B index pattern accordingly.

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

DIM_M=64
DIM_K=64
DIM_N=32

echo "[1/2] clang++ --target=aie2p-none-unknown-elf mm.cc -> mm_${DIM_M}x${DIM_K}x${DIM_N}.o"
"$PEANO/bin/clang++" \
    -O2 \
    -std=c++20 \
    --target=aie2p-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
    -DNDEBUG \
    -Di8_i32_ONLY \
    -DDIM_M=${DIM_M} \
    -DDIM_K=${DIM_K} \
    -DDIM_N=${DIM_N} \
    -DVECTORIZED_ONLY \
    -DB_COL_MAJ \
    -I "$MLIR_AIE_INC" \
    -c mm.cc \
    -o build/mm_${DIM_M}x${DIM_K}x${DIM_N}.o

echo "[2/2] aiecc --aie-generate-xclbin --aie-generate-npu-insts aie.mlir"
cd build
PEANO_INSTALL_DIR="$PEANO" \
"$AIECC_BIN" \
    --aie-generate-xclbin \
    --aie-generate-npu-insts \
    --no-compile-host \
    --no-xchesscc --no-xbridge \
    --peano="$PEANO" \
    --xclbin-name=matmul_i8_2048_32c.xclbin \
    --npu-insts-name=insts.bin \
    "$SRCDIR/aie.mlir"

echo
echo "[2.5] copy main.pdi + manifests"
cp aie.mlir.prj/main.pdi main.pdi
cp aie.mlir.prj/main_aie_partition.json main_aie_partition.json 2>/dev/null || true
cp aie.mlir.prj/main_kernels.json main_kernels.json 2>/dev/null || true

echo
echo "Built artifacts:"
ls -la main.pdi insts.bin matmul_i8_2048_32c.xclbin 2>/dev/null || true
