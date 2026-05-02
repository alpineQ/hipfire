#!/usr/bin/env bash
# matmul_i8_512_32c — hand-rolled native build for 8c x 4-row = 32-tile
# i8 GEMM at M=K=N=512 with tile (m, k, n) = (64, 64, 32).
#
# MLIR was generated once via mlir-aie's whole_array_placed.py with
# --dev=npu2 -M 512 -K 512 -N 512 -m 64 -k 64 -n 32 --n-aie-cols 8
# --dtype_in i8 --dtype_out i32 — see aie.mlir in this dir.
#
# Note: M=K=N=1024 hits a DMA descriptor size limit (max 1023 per dim) in
# mlir-aie. Larger shapes need K-dim tiling, addressed in subsequent kernels.
#
# mm.cc + zero.cc were copied from mlir-aie/aie_kernels/aie2p/.

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
    --xclbin-name=matmul_i8_512_32c.xclbin \
    --npu-insts-name=insts.bin \
    "$SRCDIR/aie.mlir"

echo
echo "[2.5] copy main.pdi + manifests"
cp aie.mlir.prj/main.pdi main.pdi
cp aie.mlir.prj/main_aie_partition.json main_aie_partition.json 2>/dev/null || true
cp aie.mlir.prj/main_kernels.json main_kernels.json 2>/dev/null || true

echo
echo "Built artifacts:"
ls -la main.pdi insts.bin matmul_i8_512_32c.xclbin 2>/dev/null || true
