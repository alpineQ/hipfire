//===- asym3_score_kernel.cc -------------------------------*- C++ -*-===//
//
// Stage 2.6 fused asym3 score kernel for AIE-2P (Strix Halo).
// Single (head, position) per call; multi-core fan-out and the
// runtime-sequence iteration come from aie.mlir at a higher level
// (mirroring the asym3_dequant_layer multi-core pattern).
//
// Per call:
//   - Dequant 256 packed 3-bit indices to f32 v[256] = cnorm * codebook[idx]
//   - Inverse Givens for each band pair to recover (k_re, k_im)
//   - Compute k_mag = sqrt(k_re^2 + k_im^2)
//   - Accumulate s_trig and s_norm using the trig-eliminated form
//     (cos_a[f], sin_a[f] precomputed on host, see plan doc):
//       s_trig += c_mag[f] * (cos_a[f] * k_re + sin_a[f] * k_im)
//       s_norm += (1 - min(c_mag[f] / c_abs[f], 1)) * c_abs[f] * k_mag
//     where c_mag = sqrt(c_re^2 + c_im^2) is precomputed in centers
//     so we don't need atan2 anywhere.
//
// Inputs (host-prepared per (head, pos) call):
//   packed:    96 bytes  (32 * 3 bytes packed indices)
//   cnorm:     1 f32
//   c_mag:     n_bands f32  (sqrt(c_re^2 + c_im^2) per band)
//   c_abs:     n_bands f32  (e_abs_q per band)
//   cos_a:     n_bands f32  (cos(omega[f]*p_q + c_phase[f]))
//   sin_a:     n_bands f32  (sin(omega[f]*p_q + c_phase[f]))
//   cos_theta: n_bands f32  (Givens cos)
//   sin_theta: n_bands f32  (Givens sin)
// where n_bands = head_dim / 2 = 128 for head_dim=256.
//
// Output: 1 f32 score = s_trig + s_norm
//
// This MVP prioritizes correctness over speed. SIMD optimization
// follows once the math + parity vs the iGPU reference is proved.

#include <aie_api/aie.hpp>
#include <stdint.h>

// Peano's aie2p math.h only declares fabs/fabsf/fabsl. libm.a
// has the rest of the symbols but no header. Forward-declare
// sqrtf so clang++ accepts the call; aiecc links libm.a in the
// final kernel ELF.
extern "C" float sqrtf(float);

// Codebook stays f32 in the score kernel. Unlike asym3_dequant_layer
// (which writes bf16 K and so must match the AIE-2P bf16 mul-and-store
// shape with a bf16 codebook), the score kernel produces an f32 output
// scalar and never stores bf16. Matching the iGPU triattn_score_asym3
// reference means f32 codebook + f32 cnorm + f32 mul throughout.
static const float TURBO_C3[8] = {
    -0.134860f, -0.083320f, -0.046469f, -0.015176f,
    +0.015176f, +0.046469f, +0.083320f, +0.134860f,
};

constexpr int HEAD_DIM = 256;
constexpr int N_BANDS  = HEAD_DIM / 2;   // 128
constexpr int N_TIDS   = 32;
constexpr int N_DIMS_PER_TID = HEAD_DIM / N_TIDS;  // 8

static inline void
unpack_indices_for_thread(const uint8_t *base, int8_t *out8) {
  // 3 bytes -> 8 packed 3-bit indices (24 bits total).
  uint32_t bits = (uint32_t)base[0]
                | ((uint32_t)base[1] << 8)
                | ((uint32_t)base[2] << 16);
  for (int k = 0; k < 8; ++k) {
    out8[k] = (int8_t)((bits >> (3 * k)) & 0x7);
  }
}

extern "C" {

// Single-(head, position) fused score; n_bands implicitly == 128.
void asym3_score_one(
    uint8_t *packed,        // 96 bytes
    float   *cnorm_ptr,     // 1 f32
    float   *c_mag,         // 128 f32
    float   *c_abs,         // 128 f32
    float   *cos_a,         // 128 f32
    float   *sin_a,         // 128 f32
    float   *cos_theta,     // 128 f32
    float   *sin_theta,     // 128 f32
    float   *score_out      // 1 f32 output
) {
  // cnorm and codebook stay f32 (no bf16 round-trip): the iGPU
  // reference runs in f32 throughout and the score output is f32,
  // so any bf16 truncation here is pure precision loss with zero
  // hardware-shape benefit.
  float cnorm_f = *cnorm_ptr;

  float s_trig = 0.0f;
  float s_norm = 0.0f;

  // Process all 32 thread-equivalents serially in this simple
  // single-core MVP. 32 thread-ids x 8 dims = 256 dims.
  for (int tid = 0; tid < N_TIDS; ++tid) {
    int8_t idx_buf[N_DIMS_PER_TID];
    unpack_indices_for_thread(packed + tid * 3, idx_buf);

    // Dequant 8 dims for this tid -> v[0..8] in f32, matching the
    // iGPU triattn_score_asym3 reference exactly.
    float v[N_DIMS_PER_TID];
    for (int i = 0; i < N_DIMS_PER_TID; ++i) {
      v[i] = cnorm_f * TURBO_C3[idx_buf[i]];
    }

    // 4 band pairs per tid (since 8 dims = 4 complex bands).
    int b0 = tid * 4;
    for (int j = 0; j < 4; ++j) {
      int f = b0 + j;
      float a_gv = v[j * 2 + 0];
      float b_gv = v[j * 2 + 1];

      // Inverse Givens.
      float cb = cos_theta[f];
      float sb = sin_theta[f];
      float k_re =  cb * a_gv + sb * b_gv;
      float k_im = -sb * a_gv + cb * b_gv;

      // Reformulated s_trig: no atan2 / cos needed per band.
      // s_trig += c_mag * k_mag * cos(angle)
      //        = c_mag * (cos_a * k_re + sin_a * k_im)
      s_trig += c_mag[f] * (cos_a[f] * k_re + sin_a[f] * k_im);

      // s_norm still needs k_mag.
      float k_mag2 = k_re * k_re + k_im * k_im;
      // Peano libm scalar sqrtf; SIMD aie::sqrt comes in the
      // optimized variant once correctness is proven. The
      // __builtin_sqrtf variant fails to legalize on aie2p
      // ("unable to legalize instruction: G_FSQRT").
      float k_mag = sqrtf(k_mag2);

      float r = (c_abs[f] > 1e-20f) ? (c_mag[f] / c_abs[f]) : 0.0f;
      if (r > 1.0f) r = 1.0f;
      s_norm += (1.0f - r) * c_abs[f] * k_mag;
    }
  }

  *score_out = s_trig + s_norm;
}

} // extern "C"
