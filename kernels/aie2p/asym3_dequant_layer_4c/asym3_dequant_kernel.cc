//===- asym3_dequant_kernel.cc -----------------------------*- C++ -*-===//
//
// 2-core asym3 dequant kernel for AIE-2P (Strix Halo). Phase A MVP
// of the multi-core layer kernel.
//
// Compute logic per (head, position) is identical to the
// asym3_dequant_layer (single-core) kernel; the multi-core
// rewrite moves the iteration distribution into the MLIR runtime
// sequence + ObjectFifo topology. Each compute tile gets its own
// independent input/output ObjectFifo path (no broadcast or
// fan-in via mem_tile), processing N_ITERS/2 chunks in its share.
//
// See docs/plans/asym3-multicore-plan.md for the architecture
// rationale.
//
//===------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

static const bfloat16 TURBO_C3[8] = {
    bfloat16(-0.134860f), bfloat16(-0.083320f),
    bfloat16(-0.046469f), bfloat16(-0.015176f),
    bfloat16(+0.015176f), bfloat16(+0.046469f),
    bfloat16(+0.083320f), bfloat16(+0.134860f),
};

static inline void
unpack_16_indices(const uint8_t *p, int8_t *out16) {
  uint64_t bits = 0;
  bits |= (uint64_t)p[0] << 0;
  bits |= (uint64_t)p[1] << 8;
  bits |= (uint64_t)p[2] << 16;
  bits |= (uint64_t)p[3] << 24;
  bits |= (uint64_t)p[4] << 32;
  bits |= (uint64_t)p[5] << 40;
  for (int k = 0; k < 16; ++k) {
    out16[k] = (int8_t)((bits >> (3 * k)) & 0x7);
  }
}

static inline ::aie::vector<bfloat16, 16>
lookup_and_scale_16(const ::aie::vector<int8_t, 16> &idx,
                    const ::aie::vector<bfloat16, 16> &cnorm_v) {
  ::aie::vector<bfloat16, 16> result =
      ::aie::broadcast<bfloat16, 16>((bfloat16)0.0f);
  for (int e = 0; e < 8; ++e) {
    auto mask = ::aie::eq(idx, ::aie::broadcast<int8_t, 16>((int8_t)e));
    auto code_e = ::aie::broadcast<bfloat16, 16>(TURBO_C3[e]);
    result = ::aie::select(result, code_e, mask);
  }
  return ::aie::mul(result, cnorm_v).template to_vector<bfloat16>();
}

extern "C" {

// Single (head, position) dequant; same compute as asym3_dequant_layer.
// The 2-core variant invokes this from each of two compute tiles,
// each iterating its own scf.for over its share of N_ITERS chunks.
void asym3_dequant_layer_one(uint8_t *packed, float *cnorm_ptr,
                             bfloat16 *out) {
  bfloat16 cnorm = (bfloat16)(*cnorm_ptr);
  ::aie::vector<bfloat16, 16> cnorm_v =
      ::aie::broadcast<bfloat16, 16>(cnorm);

  for (int c = 0; c < 16; ++c) {
    int8_t idx_buf[16];
    unpack_16_indices(packed + c * 6, idx_buf);
    ::aie::vector<int8_t, 16> idx = ::aie::load_v<16>(idx_buf);
    ::aie::vector<bfloat16, 16> v = lookup_and_scale_16(idx, cnorm_v);
    ::aie::store_v(out + c * 16, v);
  }
}

} // extern "C"
