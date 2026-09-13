/*
 * shaders_ds41.metal — Metal kernels for the DeepSeek-V4.1-Flash weight formats.
 *
 * STATUS: written against the CPU reference in ds41_ops.h (unit-tested), but NOT
 * yet compiled or benchmarked on a Mac — this session ran on Linux. First thing to
 * do on the M3 Max: build with `xcrun -sdk macosx metal -c shaders_ds41.metal`,
 * then run the `--verify` path against ds41_mxfp4_matvec() / ds41_fp8blk_matvec().
 *
 * Formats (from the official checkpoint):
 *   MXFP4 experts : W [out][in/2] bytes, low nibble = even column, E2M1 values
 *                   {0,.5,1,1.5,2,3,4,6}±; one E8M0 scale (2^(e-127)) per 32 columns,
 *                   S [out][in/32] bytes.  (OCP MX spec; same as MLX mxfp4.)
 *   Block FP8 dense: W [out][in] E4M3FN bytes; one E8M0 scale per 32x32 block,
 *                   S [out/32][in/32] bytes.
 *
 * Kernel structure mirrors dequant_matvec_4bit_v5 in shaders.metal (the FMA
 * kernel that won for Qwen3.5): 256 threads = 8 simdgroups = 8 output rows per
 * threadgroup, the input vector staged in threadgroup memory, each lane striding
 * over packed 32-bit words, simd_sum reduction at the end.
 *
 * Why the scale is applied once per 32-element block and not per element:
 * E8M0 scales are exact powers of two, so acc += scale * (sum of 8 LUT*x)
 * costs one extra FMA per word and keeps the inner loop at LUT-load + FMA, the
 * same shape the Qwen kernel profiled best with. A 16-entry LUT in constant
 * memory is what llama.cpp's mxfp4 Metal kernel does as well.
 */

#include <metal_stdlib>
using namespace metal;

constant float DS41_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/* 2^(e-127) via the float exponent field; e in 1..254 exact, e==0 -> 0 (true value 2^-127). */
inline float ds41_e8m0(uint8_t e) {
    return as_type<float>(uint(e) << 23);
}

/* OCP FP8 E4M3FN -> float. exp==0: subnormal m*2^-9; else 2^(e-7)*(1+m/8). NaN (0x7F/0xFF) -> ±448-ish is
 * avoided by the exporter, so we do not special-case it in the hot loop. */
inline float ds41_e4m3(uint v) {
    uint s = (v >> 7) & 1u, e = (v >> 3) & 0xFu, m = v & 7u;
    float mag = (e == 0u) ? float(m) * 0.001953125f
                          : as_type<float>(((e + 120u) << 23) | (m << 20));
    return s ? -mag : mag;
}

#define DS41_ROWS_PER_TG 8
#define DS41_MAX_IN      5120   /* hidden dim; gate/up read 5120, down reads 2304 */

// ============================================================================
// MXFP4 dequant matvec: out[row] = sum_i e2m1(W[row][i]) * 2^(S[row][i/32]-127) * x[i]
// ============================================================================
kernel void dequant_matvec_mxfp4(
    device const uint32_t* W_packed   [[buffer(0)]],   // [out_dim][in_dim/8] words (8 nibbles each)
    device const uint8_t*  scales     [[buffer(1)]],   // [out_dim][in_dim/32] E8M0
    device const float*    x          [[buffer(2)]],   // [in_dim]
    device float*          out        [[buffer(3)]],   // [out_dim]
    constant uint&         out_dim    [[buffer(4)]],
    constant uint&         in_dim     [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * DS41_ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 8;      // 32-bit words per row
    uint num_groups  = in_dim / 32;     // scale blocks per row (4 words per block)

    threadgroup float x_shared[DS41_MAX_IN];
    for (uint i = lid; i < in_dim; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint8_t*  s_row = scales + row * num_groups;

    float acc = 0.0f;
    // Lane l handles words l, l+32, l+64, ...: consecutive lanes read consecutive words
    // (coalesced), and every word lies inside exactly one 32-element scale block.
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint32_t packed = w_row[col];
        float scale = ds41_e8m0(s_row[col >> 2]);
        uint xb = col * 8;
        float part;
        part  = DS41_E2M1_LUT[(packed >>  0) & 0xF] * x_shared[xb + 0];
        part  = fma(DS41_E2M1_LUT[(packed >>  4) & 0xF], x_shared[xb + 1], part);
        part  = fma(DS41_E2M1_LUT[(packed >>  8) & 0xF], x_shared[xb + 2], part);
        part  = fma(DS41_E2M1_LUT[(packed >> 12) & 0xF], x_shared[xb + 3], part);
        part  = fma(DS41_E2M1_LUT[(packed >> 16) & 0xF], x_shared[xb + 4], part);
        part  = fma(DS41_E2M1_LUT[(packed >> 20) & 0xF], x_shared[xb + 5], part);
        part  = fma(DS41_E2M1_LUT[(packed >> 24) & 0xF], x_shared[xb + 6], part);
        part  = fma(DS41_E2M1_LUT[(packed >> 28) & 0xF], x_shared[xb + 7], part);
        acc = fma(scale, part, acc);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) out[row] = sum;
}

// ============================================================================
// Block-FP8 dequant matvec for the resident dense projections (wq_a/wq_b/wkv/wo_a/wo_b,
// shared expert, engram.wkv): out[row] = sum_i e4m3(W[row][i]) * 2^(S[row/32][i/32]-127) * x[i]
// ============================================================================
kernel void dequant_matvec_fp8_blk32(
    device const uint32_t* W_bytes    [[buffer(0)]],   // [out_dim][in_dim/4] words (4 fp8 each)
    device const uint8_t*  scales     [[buffer(1)]],   // [out_dim/32][in_dim/32] E8M0
    device const float*    x          [[buffer(2)]],   // [in_dim]
    device float*          out        [[buffer(3)]],   // [out_dim]
    constant uint&         out_dim    [[buffer(4)]],
    constant uint&         in_dim     [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * DS41_ROWS_PER_TG + simd_group;
    uint words_per_row = in_dim / 4;
    uint in_blocks = (in_dim + 31) / 32;

    // in_dim can be 6144 (engram.wkv) or 8192 (wo_b): stage in chunks of DS41_MAX_IN if larger.
    threadgroup float x_shared[DS41_MAX_IN];
    bool staged = in_dim <= DS41_MAX_IN;
    if (staged) {
        for (uint i = lid; i < in_dim; i += 256) x_shared[i] = x[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_bytes + row * words_per_row;
    device const uint8_t*  s_row = scales + (row >> 5) * in_blocks;

    float acc = 0.0f;
    for (uint col = simd_lane; col < words_per_row; col += 32) {
        uint32_t w4 = w_row[col];
        float scale = ds41_e8m0(s_row[col >> 3]);    // 8 words = 32 columns per block
        uint xb = col * 4;
        float x0, x1, x2, x3;
        if (staged) { x0 = x_shared[xb]; x1 = x_shared[xb + 1]; x2 = x_shared[xb + 2]; x3 = x_shared[xb + 3]; }
        else        { x0 = x[xb];        x1 = x[xb + 1];        x2 = x[xb + 2];        x3 = x[xb + 3]; }
        float part = ds41_e4m3(w4 & 0xFFu) * x0;
        part = fma(ds41_e4m3((w4 >>  8) & 0xFFu), x1, part);
        part = fma(ds41_e4m3((w4 >> 16) & 0xFFu), x2, part);
        part = fma(ds41_e4m3((w4 >> 24) & 0xFFu), x3, part);
        acc = fma(scale, part, acc);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) out[row] = sum;
}

// ============================================================================
// Clamped SwiGLU (Expert.forward): up in [-limit, limit], gate <= limit, out = silu(gate) * up.
// Drop-in replacement for swiglu_fused with the extra `limit` argument.
// ============================================================================
kernel void swiglu_clamped(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant float&     limit  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float g = min(gate[tid], limit);
    float u = clamp(up[tid], -limit, limit);
    out[tid] = (g / (1.0f + exp(-g))) * u;
}

// ============================================================================
// Hyper-connection helpers (Block.hc_pre / hc_post). The residual stream is hc=4
// copies of dim=5120 floats. These replace residual_add / moe_combine_residual.
// ============================================================================
kernel void hc_pre_collapse(
    device const float* x      [[buffer(0)]],   // [hc][dim]
    device const float* pre    [[buffer(1)]],   // [hc]
    device float*       y      [[buffer(2)]],   // [dim]
    constant uint&      dim    [[buffer(3)]],
    constant uint&      hc     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float acc = 0.0f;
    for (uint c = 0; c < hc; c++) acc = fma(pre[c], x[c * dim + tid], acc);
    y[tid] = acc;
}

// out[c][d] = post[c] * y[d] + sum_k comb[c][k] * residual[k][d]
// `y` is the sublayer output (attention or MoE combine); one thread per (c, d).
kernel void hc_post_expand(
    device const float* y        [[buffer(0)]],   // [dim]
    device const float* residual [[buffer(1)]],   // [hc][dim]
    device const float* post     [[buffer(2)]],   // [hc]
    device const float* comb     [[buffer(3)]],   // [hc][hc]
    device float*       out      [[buffer(4)]],   // [hc][dim]
    constant uint&      dim      [[buffer(5)]],
    constant uint&      hc       [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim * hc) return;
    uint c = tid / dim, d = tid - c * dim;
    float acc = post[c] * y[d];
    for (uint k = 0; k < hc; k++) acc = fma(comb[c * hc + k], residual[k * dim + d], acc);
    out[tid] = acc;
}

// ============================================================================
// MoE combine for K<=8 routed experts + shared expert (no sigmoid gate in V4.1: the
// shared expert is added with weight 1, the routed weights already include
// route_scale). Output is the FFN sublayer output `y` fed to hc_post_expand.
// ============================================================================
kernel void ds41_moe_combine(
    device const float* shared_out  [[buffer(0)]],   // [dim]
    device float*       y           [[buffer(1)]],   // [dim] output
    device const float* expert_out0 [[buffer(2)]],
    device const float* expert_out1 [[buffer(3)]],
    device const float* expert_out2 [[buffer(4)]],
    device const float* expert_out3 [[buffer(5)]],
    device const float* expert_out4 [[buffer(6)]],
    device const float* expert_out5 [[buffer(7)]],
    device const float* expert_out6 [[buffer(8)]],
    device const float* expert_out7 [[buffer(9)]],
    device const float* weights     [[buffer(10)]],  // [8] routed weights (route_scale applied)
    constant uint&      dim         [[buffer(11)]],
    constant uint&      K           [[buffer(12)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float moe = shared_out[tid];
    if (K > 0) moe = fma(weights[0], expert_out0[tid], moe);
    if (K > 1) moe = fma(weights[1], expert_out1[tid], moe);
    if (K > 2) moe = fma(weights[2], expert_out2[tid], moe);
    if (K > 3) moe = fma(weights[3], expert_out3[tid], moe);
    if (K > 4) moe = fma(weights[4], expert_out4[tid], moe);
    if (K > 5) moe = fma(weights[5], expert_out5[tid], moe);
    if (K > 6) moe = fma(weights[6], expert_out6[tid], moe);
    if (K > 7) moe = fma(weights[7], expert_out7[tid], moe);
    y[tid] = moe;
}
