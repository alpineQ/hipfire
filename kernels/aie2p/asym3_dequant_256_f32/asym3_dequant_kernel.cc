//===- asym3_dequant_kernel.cc (f32 diagnostic variant) ----*- C++ -*-===//
//
// Diagnostic sibling of asym3_dequant_256 used to characterize the
// AIE-2P bf16 mul rounding chain. Computes the SAME asym3 dequant
// but stores the multiplication result as fp32 (no final bf16
// round). The CPU verifier compares the fp32 NPU output against
// the CPU f32 product computed from bf16-promoted-to-f32 inputs:
//
//   - if they match: the bf16 mul is "promote bf16 to f32, multiply
//     in f32, store f32" (lossless on bf16 inputs); divergence in
//     the regular kernel must be in the to_vector<bfloat16>
//     conversion.
//   - if they differ: the bf16 mul uses limited internal precision
//     (e.g. 8x8 mantissa multiplier), and we need to model that.
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

extern "C" {

// Same packed 96 B + 1 f32 cnorm input as asym3_dequant_256, but the
// output is 256 * f32 = 1024 bytes. We store the accumulator
// directly via to_vector<float>() — no bf16 down-conversion.
void asym3_dequant_256_f32(uint8_t *packed, float *cnorm_ptr,
                           float *out) {
  bfloat16 cnorm = (bfloat16)(*cnorm_ptr);
  ::aie::vector<bfloat16, 16> cnorm_v =
      ::aie::broadcast<bfloat16, 16>(cnorm);

  for (int c = 0; c < 16; ++c) {
    int8_t idx_buf[16];
    unpack_16_indices(packed + c * 6, idx_buf);
    ::aie::vector<int8_t, 16> idx = ::aie::load_v<16>(idx_buf);

    ::aie::vector<bfloat16, 16> result =
        ::aie::broadcast<bfloat16, 16>((bfloat16)0.0f);
    for (int e = 0; e < 8; ++e) {
      auto mask = ::aie::eq(idx, ::aie::broadcast<int8_t, 16>((int8_t)e));
      auto code_e = ::aie::broadcast<bfloat16, 16>(TURBO_C3[e]);
      result = ::aie::select(result, code_e, mask);
    }

    // Store the mul accumulator as fp32 instead of rounding to bf16.
    auto acc = ::aie::mul(result, cnorm_v);
    ::aie::vector<float, 16> v_f32 = acc.template to_vector<float>();
    ::aie::store_v(out + c * 16, v_f32);
  }
}

} // extern "C"
