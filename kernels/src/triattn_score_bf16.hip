// TriAttention scoring on a pre-dequanted bf16 K cache.
//
// Companion kernel to triattn_score_asym3.hip for the NPU dequant
// path (stage 1.4). Same grid/block shape, same Givens + RoPE +
// score logic; the dequant block (cnorm * codebook[idx]) is
// replaced with a direct bf16 K read since the values arrive
// pre-dequanted from NpuRuntime::asym3_dequant_layer.
//
// bf16 K layout (output of asym3_dequant_layer):
//   bytes_per_head = head_dim * 2     // 512 for hd=256
//   per-thread owns 8 consecutive dims (tid*8 .. tid*8+7) = 16 bytes
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>

// bf16 to float helper. Bit-shift the bf16 LE pair into the high 16
// bits of an f32 (same semantics as upcasting on read; bf16 has the
// same exponent format as f32).
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    union { unsigned int u; float f; } x;
    x.u = (unsigned int)b << 16;
    return x.f;
}

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_bf16(
    const unsigned short* __restrict__ k_cache_bf16,  // [seq_len * n_kv_heads * head_dim] bf16
    const float* __restrict__ centers,                // [n_heads * n_bands * 3]
    const float* __restrict__ cos_theta,              // [n_bands]
    const float* __restrict__ sin_theta,              // [n_bands]
    float* __restrict__ scores,                       // [n_heads * seq_len]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int n_rot,
    float rope_theta,
    float p_q,
    int seq_len
) {
    const int pos = blockIdx.x;
    const int h = blockIdx.y;
    if (pos >= seq_len || h >= n_heads) return;
    const int tid = threadIdx.x;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;

    const int n_bands = head_dim / 2;
    const int n_rot_bands = n_rot / 2;

    // bf16 K layout: per-position contiguous block of (n_kv_heads * head_dim)
    // bf16 values. No cnorm prefix since dequant was done upstream.
    const int k_per_head = head_dim;
    const int k_per_pos  = n_kv_heads * head_dim;
    const unsigned short* k_head = k_cache_bf16
        + (size_t)pos * k_per_pos + (size_t)kv_h * k_per_head;

    // This thread reads 8 contiguous bf16 dims (tid*8 .. tid*8+7).
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = bf16_to_f32(k_head[tid * 8 + i]);
    }

    // Inverse Givens per band pair. Same trig as triattn_score_asym3.
    const int b0 = tid * 4;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int f = b0 + j;
        const float cb = cos_theta[f];
        const float sb = sin_theta[f];

        const float a_gv = v[j * 2 + 0];
        const float b_gv = v[j * 2 + 1];
        const float k_re =  cb * a_gv + sb * b_gv;
        const float k_im = -sb * a_gv + cb * b_gv;
        const float k_mag = sqrtf(k_re * k_re + k_im * k_im);
        const float k_phase = atan2f(k_im, k_re);

        const float* c = c_head + f * 3;
        const float c_re  = c[0];
        const float c_im  = c[1];
        const float c_abs = c[2];
        const float c_mag   = sqrtf(c_re * c_re + c_im * c_im);
        const float c_phase = atan2f(c_im, c_re);

        float omega = 0.0f;
        if (f < n_rot_bands) {
            const float exponent = -2.0f * (float)f / (float)n_rot;
            omega = expf(exponent * logf(rope_theta));
        }

        const float angle = omega * p_q + c_phase - k_phase;
        s_trig += c_mag * k_mag * cosf(angle);

        float r = (c_abs > 1e-20f) ? (c_mag / c_abs) : 0.0f;
        if (r > 1.0f) r = 1.0f;
        s_norm += (1.0f - r) * c_abs * k_mag;
    }

    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }

    if (tid == 0) {
        scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
    }
}
