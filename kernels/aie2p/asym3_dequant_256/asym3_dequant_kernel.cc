//===- asym3_dequant_kernel.cc -----------------------------*- C++ -*-===//
//
// Per-head asym3 dequant kernel for AIE-2P (Strix Halo).
//
// Decodes 256 packed 3-bit indices to 256 bf16 values via:
//   k[d] = cnorm * TURBO_C3_256[idx[d]]
//
// Codebook (8 entries):
//   {-0.135, -0.083, -0.046, -0.015, +0.015, +0.046, +0.083, +0.135}
//
// Input layout per call (one head, one position):
//   packed: 96 bytes  (256 indices × 3 bits ÷ 8 bits/byte)
//   cnorm:    1 f32   (scaling factor for this head)
//   out:    512 bytes (256 × bf16)
//
// Engine call site (when wired): asym3 KV cache layout from
// `crates/engine/src/cask.rs` matches this exactly:
//   K cache slice per (pos, head) = [4-byte cnorm | 96 bytes indices]
//
// The codebook lookup uses an 8-way "broadcast + select" pattern
// rather than a gather instruction. AIE-2P does have shuffle/gather
// for byte-level permutations, but for a tiny 8-entry codebook the
// broadcast-select approach is competitive and simpler to verify.
//
//===------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Codebook constants — matches crates/engine/src/cask.rs::TURBO_C3_256.
static const bfloat16 TURBO_C3[8] = {
    (bfloat16)-0.135f, (bfloat16)-0.083f, (bfloat16)-0.046f, (bfloat16)-0.015f,
    (bfloat16)+0.015f, (bfloat16)+0.046f, (bfloat16)+0.083f, (bfloat16)+0.135f,
};

// Unpack 16 consecutive 3-bit indices from 6 input bytes (48 bits).
// Returns a vector<int8, 16> where each lane holds one 3-bit value
// in 0..7. Bit ordering: indices stored little-endian within bytes,
// indices[k] occupies bits 3k..3k+2 of the packed bitstream.
static inline aie::vector<int8, 16>
unpack_16_indices(const uint8_t *p) {
  // Treat 6 bytes as a 48-bit little-endian integer for cleanest extract.
  uint64_t bits = 0;
  bits |= (uint64_t)p[0] << 0;
  bits |= (uint64_t)p[1] << 8;
  bits |= (uint64_t)p[2] << 16;
  bits |= (uint64_t)p[3] << 24;
  bits |= (uint64_t)p[4] << 32;
  bits |= (uint64_t)p[5] << 40;

  aie::vector<int8, 16> idx;
  // Compiler will optimize the literal shifts. AIE-2P scalar throughput
  // for these is ~1 cycle each; vectorizing the unpack itself is not
  // worth the extra logic for 16 values.
  for (int k = 0; k < 16; ++k) {
    int8 v = (int8)((bits >> (3 * k)) & 0x7);
    idx[k] = v;
  }
  return idx;
}

// Look up the codebook for 16 indices and broadcast-multiply by cnorm.
// Uses 8-way broadcast+select rather than gather: for each codebook
// entry e ∈ 0..7, build a mask `indices == e`, then select between
// the running result and a broadcast of CODEBOOK[e]. After 8
// passes every lane has its codebook value. Then a single fmul
// applies cnorm.
static inline aie::vector<bfloat16, 16>
lookup_and_scale_16(const aie::vector<int8, 16> &idx,
                    const aie::vector<bfloat16, 16> &cnorm_v) {
  aie::vector<bfloat16, 16> result =
      aie::broadcast<bfloat16, 16>((bfloat16)0.0f);
  for (int e = 0; e < 8; ++e) {
    auto mask = aie::eq(idx, aie::broadcast<int8, 16>((int8)e));
    auto code_e = aie::broadcast<bfloat16, 16>(TURBO_C3[e]);
    result = aie::select(code_e, result, mask);
  }
  return aie::mul(result, cnorm_v).template to_vector<bfloat16>();
}

extern "C" {

// Single-head dequant for head_dim=256. One worker tile per call.
//
// Inputs:
//   packed:    256 indices × 3 bits = 96 bytes (uint8 array of 96)
//   cnorm_ptr: 1-element f32 array (degenerate; passed via a
//              4-byte ObjectFifo from the runtime)
// Output:
//   out: 256 bf16 values (512 bytes)
void asym3_dequant_256(uint8_t *packed, float *cnorm_ptr, bfloat16 *out) {
  bfloat16 cnorm = (bfloat16)(*cnorm_ptr);
  aie::vector<bfloat16, 16> cnorm_v = aie::broadcast<bfloat16, 16>(cnorm);

  // 256 outputs ÷ 16 SIMD lanes = 16 chunks
  // Each chunk consumes 6 input bytes (16 × 3 bits = 48 bits)
  for (int c = 0; c < 16; ++c) {
    aie::vector<int8, 16> idx = unpack_16_indices(packed + c * 6);
    aie::vector<bfloat16, 16> v = lookup_and_scale_16(idx, cnorm_v);
    aie::store_v(out + c * 16, v);
  }
}

} // extern "C"
