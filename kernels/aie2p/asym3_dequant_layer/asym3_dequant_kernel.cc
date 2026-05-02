//===- asym3_dequant_kernel.cc -----------------------------*- C++ -*-===//
//
// Per-layer batched asym3 dequant kernel for AIE-2P (Strix Halo).
//
// Same per-(head, position) compute as asym3_dequant_256: decode 256
// packed 3-bit indices to 256 bf16 values via
//   k[d] = cnorm * TURBO_C3_256[idx[d]]
//
// The batching is in the host-side runtime sequence + core scf.for
// loop: a single dispatch carries N (head, position) pairs through
// three ObjectFifos sized at 96 bytes / 4 bytes / 512 bytes per
// element. The C++ kernel itself is unchanged from the 256 variant
// and processes ONE (head, position) pair per call.
//
// Stage 1.4 MVP iteration uses N_ITERS=32 so the dispatch covers a
// 32 (head, position) batch (e.g. 8 heads x 4 positions, or 4 heads
// x 8 positions). Production-shape variants for full 27B Gemma
// layers (8 heads x 4096 positions = 32768 iters) will follow in
// the multi-core scaling pass.
//
// See kernels/aie2p/asym3_dequant_256/asym3_dequant_kernel.cc for
// the per-(head, position) algorithm explanation.
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

// Single (head, position) dequant; same as asym3_dequant_256 but with
// a different exported symbol so the layer-batched MLIR can link
// against it independently of the 256 kernel.
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
