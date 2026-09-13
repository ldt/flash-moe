/*
 * ds41_ops.h — CPU reference operators for DeepSeek-V4.1-Flash (single header, C11).
 *
 * These are the *numerics* the Metal port (shaders_ds41.metal) must reproduce,
 * written plainly so they can be unit-tested on any machine (make test-ds41)
 * and used as the CPU fallback / verification path inside infer.m, exactly as
 * cpu_dequant_matvec() is used for the Qwen3.5 4-bit path today.
 *
 * Every formula here is transcribed from the official reference implementation
 * (inference/model.py, kernel.py, engram.py, convert.py in the HF repo); the
 * comment above each function names its source.
 *
 * Usage:
 *   #define DS41_OPS_IMPL      // in exactly one .c/.m file
 *   #include "ds41_ops.h"
 */
#ifndef DS41_OPS_H
#define DS41_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- model constants (inference/config.json) ------------------------------ */
#define DS41_DIM              5120
#define DS41_MOE_INTER        2304
#define DS41_N_LAYERS         40
#define DS41_N_ROUTED         384
#define DS41_N_ACTIVE         6
#define DS41_ROUTE_SCALE      1.5f
#define DS41_SWIGLU_LIMIT     10.0f
#define DS41_NORM_EPS         1e-20f
#define DS41_HC_MULT          4
#define DS41_HC_SINKHORN_IT   20
#define DS41_HC_EPS           1e-6f
#define DS41_FP4_BLOCK        32     /* one E8M0 scale per 32 columns (MXFP4)            */
#define DS41_FP8_BLOCK        32     /* one E8M0 scale per 32x32 block of FP8 weights    */
#define DS41_ENGRAM_HEAD_DIM  256
#define DS41_ENGRAM_MAX_NGRAM 4
#define DS41_ENGRAM_N_HEADS   8
#define DS41_ENGRAM_N_LAYERS  2

/* Per-expert record as stored in the shards (see generate_ds41_expert_index.py). */
#define DS41_EXPERT_W13_BYTES   (DS41_MOE_INTER * DS41_DIM / 2)          /* 5,898,240  */
#define DS41_EXPERT_W2_BYTES    (DS41_DIM * DS41_MOE_INTER / 2)          /* 5,898,240  */
#define DS41_EXPERT_S13_BYTES   (DS41_MOE_INTER * (DS41_DIM / 32))       /*   368,640  */
#define DS41_EXPERT_S2_BYTES    (DS41_DIM * (DS41_MOE_INTER / 32))       /*   368,640  */
#define DS41_EXPERT_WEIGHTS_BYTES (2 * DS41_EXPERT_W13_BYTES + DS41_EXPERT_W2_BYTES)  /* 17,694,720 */
#define DS41_EXPERT_SCALES_BYTES  (2 * DS41_EXPERT_S13_BYTES + DS41_EXPERT_S2_BYTES)  /*  1,105,920 */
/* component offsets inside the two regions (order w1, w2, w3 = gate, down, up) */
#define DS41_W1_OFF  0
#define DS41_W2_OFF  DS41_EXPERT_W13_BYTES
#define DS41_W3_OFF  (DS41_EXPERT_W13_BYTES + DS41_EXPERT_W2_BYTES)
#define DS41_S1_OFF  0
#define DS41_S2_OFF  DS41_EXPERT_S13_BYTES
#define DS41_S3_OFF  (DS41_EXPERT_S13_BYTES + DS41_EXPERT_S2_BYTES)

/* ---- scalar decoders -------------------------------------------------------- */
float ds41_e8m0_to_f32(uint8_t e);          /* 2^(e-127); 0xFF is NaN in the spec, we return 0 */
float ds41_e4m3_to_f32(uint8_t v);          /* OCP FP8 E4M3FN (no inf, 0x7F/0xFF NaN -> 0)   */
float ds41_bf16_to_f32(uint16_t b);
extern const float DS41_E2M1_LUT[16];       /* FP4 E2M1: 0 .5 1 1.5 2 3 4 6 and negatives    */

/* ---- dequantized matvecs (out[o] = sum_i W[o][i] * x[i]) --------------------- */
/* MXFP4: W is [out_dim][in_dim/2] bytes, low nibble = even column; S is [out_dim][in_dim/32] E8M0. */
void ds41_mxfp4_dequant_row(const uint8_t *w_row, const uint8_t *s_row, float *out, int in_dim);
void ds41_mxfp4_matvec(const uint8_t *W, const uint8_t *S, const float *x, float *out, int out_dim, int in_dim);
/* Block FP8: W is [out_dim][in_dim] E4M3, S is [out_dim/32][in_dim/32] E8M0 (Linear in model.py). */
void ds41_fp8blk_matvec(const uint8_t *W, const uint8_t *S, const float *x, float *out, int out_dim, int in_dim);
/* BF16 dense rows (gate.weight, compressor, embed/head). */
void ds41_bf16_matvec(const uint16_t *W, const float *x, float *out, int out_dim, int in_dim);

/* ---- elementwise ------------------------------------------------------------ */
/* Expert.forward: up clamped to [-limit, limit], gate clamped from above only, then silu(gate)*up. */
void ds41_swiglu_clamped(const float *gate, const float *up, float *out, int n, float limit);
/* RMSNorm with bf16 weight: w * x * rsqrt(mean(x^2) + eps). */
void ds41_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int n, float eps);

/* ---- MoE routing (Gate.forward): sqrtsoftplus + noaux_tc bias + norm_topk_prob + route_scale --- */
/* logits: [n_experts] = gate.weight @ x (fp32). bias: gate.bias [n_experts] (gate.bias_vl for image tokens).
 * Returns K; idx/w receive the selected experts and their scaled weights (descending by biased score). */
int ds41_route(const float *logits, const float *bias, int n_experts, int K, float route_scale,
               int *idx, float *w);

/* ---- Hyper-connections (Block.hc_mixes / kernel.hc_split_sinkhorn) --------------------------- */
/* x: [hc*dim] flattened residual copies; hc_fn: [(2+hc)*hc][hc*dim] fp32 -> mixes[(2+hc)*hc]. */
void ds41_hc_mixes(const float *x, const float *hc_fn, int hc, int dim, float norm_eps, float *mixes);
/* mixes -> pre[hc], post[hc], comb[hc][hc] (doubly-stochastic after Sinkhorn). */
void ds41_hc_split_sinkhorn(const float *mixes, const float *hc_scale3, const float *hc_base,
                            int hc, int iters, float eps, float *pre, float *post, float *comb);
/* y[dim] = sum_c pre[c] * x[c][dim]  (Block.hc_pre) */
void ds41_hc_pre(const float *x, const float *pre, int hc, int dim, float *y);
/* out[c][d] = post[c] * y[d] + sum_k comb[c][k] * residual[k][d]  (Block.hc_post) */
void ds41_hc_post(const float *y, const float *residual, const float *post, const float *comb,
                  int hc, int dim, float *out);

/* ---- RoPE frequencies with YaRN (precompute_freqs_cis) -------------------------------------- */
/* inv_freq[dim/2]; original_seq_len <= 0 disables YaRN (pure sliding-window layers). */
void ds41_rope_inv_freq(int dim, float base, float factor, int beta_fast, int beta_slow,
                        int original_seq_len, float *inv_freq);

/* ---- Engram addressing (engram.py NgramHashState) ------------------------------------------- */
typedef struct {
    uint32_t vocab_size, compressed_vocab_size, n_layers, max_ngram, n_heads, pad_compressed_id;
    int32_t  layer_ids[DS41_ENGRAM_N_LAYERS];
    int64_t  multipliers[DS41_ENGRAM_N_LAYERS][DS41_ENGRAM_MAX_NGRAM];
    int64_t  primes[DS41_ENGRAM_N_LAYERS][DS41_ENGRAM_MAX_NGRAM - 1][DS41_ENGRAM_N_HEADS];
    int64_t  offsets[DS41_ENGRAM_N_LAYERS][(DS41_ENGRAM_MAX_NGRAM - 1) * DS41_ENGRAM_N_HEADS];
    int32_t *token_map;   /* [vocab_size] token id -> compressed id (malloc'd) */
} ds41_engram_meta;

/* Load ds41_engram_meta.bin written by export_ds41_engram_meta.py. Returns 0 on success. */
int  ds41_engram_meta_load(const char *path, ds41_engram_meta *m);
void ds41_engram_meta_free(ds41_engram_meta *m);
/* Row indices for one position. history: compressed ids, history[0] = current token,
 * history[s] = s tokens back, or -1 when before the start of the sequence (-> pad).
 * rows: [(max_ngram-1)*n_heads] table row indices (n-gram order major, head minor). */
void ds41_engram_hash_compressed(const ds41_engram_meta *m, int layer_idx, const int32_t *history, int64_t *rows);
/* Same, from raw token ids (applies token_map). */
void ds41_engram_hash(const ds41_engram_meta *m, int layer_idx, const int32_t *token_history, int64_t *rows);
/* Dequantize one fetched table row: 256 E4M3 + 8 E8M0 (one per 32 channels). */
void ds41_engram_row_dequant(const uint8_t *row_fp8, const uint8_t *row_scales, float *out);
/* Engram.forward gate + add, for one hc copy: h[dim] += gate(h, key) * value[dim].
 * qk_weight[dim] = q_weight[c] * k_weight[c] (bf16 -> f32 done by caller). */
void ds41_engram_apply(float *h, const float *key, const float *value, const float *qk_weight,
                       int dim, float norm_eps);

#ifdef __cplusplus
}
#endif
#endif /* DS41_OPS_H */

/* ============================================================================ */
#ifdef DS41_OPS_IMPL

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float DS41_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

float ds41_e8m0_to_f32(uint8_t e) {
    if (e == 0xFF) return 0.0f;                 /* NaN encoding; never produced by the exporter */
    return ldexpf(1.0f, (int)e - 127);
}

/* convert.py / torch.float8_e4m3fn: sign(1) exp(4, bias 7) mant(3); exp==0 -> subnormal m*2^-9. */
float ds41_e4m3_to_f32(uint8_t v) {
    uint32_t s = (v >> 7) & 1u, e = (v >> 3) & 0xFu, m = v & 7u;
    if (e == 15u && m == 7u) return 0.0f;      /* NaN */
    float mag;
    if (e == 0) {
        mag = (float)m * 0.001953125f;          /* 2^-9 */
    } else {
        union { uint32_t u; float f; } b;
        b.u = ((e + 120u) << 23) | (m << 20);   /* 2^(e-7) * (1 + m/8) */
        mag = b.f;
    }
    return s ? -mag : mag;
}

float ds41_bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } x;
    x.u = (uint32_t)b << 16;
    return x.f;
}

void ds41_mxfp4_dequant_row(const uint8_t *w_row, const uint8_t *s_row, float *out, int in_dim) {
    for (int g = 0; g < in_dim / DS41_FP4_BLOCK; g++) {
        float scale = ds41_e8m0_to_f32(s_row[g]);
        const uint8_t *wp = w_row + g * (DS41_FP4_BLOCK / 2);
        float *op = out + g * DS41_FP4_BLOCK;
        for (int j = 0; j < DS41_FP4_BLOCK / 2; j++) {
            uint8_t byte = wp[j];
            op[2 * j]     = DS41_E2M1_LUT[byte & 0xF] * scale;      /* low nibble  = even column */
            op[2 * j + 1] = DS41_E2M1_LUT[byte >> 4]  * scale;      /* high nibble = odd column  */
        }
    }
}

void ds41_mxfp4_matvec(const uint8_t *W, const uint8_t *S, const float *x, float *out, int out_dim, int in_dim) {
    int packed_cols = in_dim / 2, num_groups = in_dim / DS41_FP4_BLOCK;
    for (int o = 0; o < out_dim; o++) {
        const uint8_t *w_row = W + (size_t)o * packed_cols;
        const uint8_t *s_row = S + (size_t)o * num_groups;
        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            float scale = ds41_e8m0_to_f32(s_row[g]);
            float part = 0.0f;
            const uint8_t *wp = w_row + g * (DS41_FP4_BLOCK / 2);
            const float *xp = x + g * DS41_FP4_BLOCK;
            for (int j = 0; j < DS41_FP4_BLOCK / 2; j++) {
                uint8_t byte = wp[j];
                part += DS41_E2M1_LUT[byte & 0xF] * xp[2 * j];
                part += DS41_E2M1_LUT[byte >> 4]  * xp[2 * j + 1];
            }
            acc += scale * part;   /* scale is a power of two: exact, applied once per block */
        }
        out[o] = acc;
    }
}

void ds41_fp8blk_matvec(const uint8_t *W, const uint8_t *S, const float *x, float *out, int out_dim, int in_dim) {
    int in_blocks = (in_dim + DS41_FP8_BLOCK - 1) / DS41_FP8_BLOCK;
    for (int o = 0; o < out_dim; o++) {
        const uint8_t *w_row = W + (size_t)o * in_dim;
        const uint8_t *s_row = S + (size_t)(o / DS41_FP8_BLOCK) * in_blocks;
        float acc = 0.0f;
        for (int b = 0; b < in_blocks; b++) {
            float scale = ds41_e8m0_to_f32(s_row[b]);
            float part = 0.0f;
            int i0 = b * DS41_FP8_BLOCK, i1 = i0 + DS41_FP8_BLOCK < in_dim ? i0 + DS41_FP8_BLOCK : in_dim;
            for (int i = i0; i < i1; i++) part += ds41_e4m3_to_f32(w_row[i]) * x[i];
            acc += scale * part;
        }
        out[o] = acc;
    }
}

void ds41_bf16_matvec(const uint16_t *W, const float *x, float *out, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W + (size_t)o * in_dim;
        float acc = 0.0f;
        for (int i = 0; i < in_dim; i++) acc += ds41_bf16_to_f32(w_row[i]) * x[i];
        out[o] = acc;
    }
}

void ds41_swiglu_clamped(const float *gate, const float *up, float *out, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float g = gate[i], u = up[i];
        if (limit > 0.0f) {
            if (u >  limit) u =  limit;
            if (u < -limit) u = -limit;
            if (g >  limit) g =  limit;   /* gate clamped from above only */
        }
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * u;
    }
}

void ds41_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = ds41_bf16_to_f32(w_bf16[i]) * (x[i] * r);
}

/* F.softplus with PyTorch's default threshold 20. */
static inline float ds41_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

int ds41_route(const float *logits, const float *bias, int n_experts, int K, float route_scale,
               int *idx, float *w) {
    float *scores = (float *)malloc(sizeof(float) * n_experts);
    float *biased = (float *)malloc(sizeof(float) * n_experts);
    for (int e = 0; e < n_experts; e++) {
        scores[e] = sqrtf(ds41_softplus(logits[e]));       /* score_func == "sqrtsoftplus" */
        biased[e] = scores[e] + (bias ? bias[e] : 0.0f);   /* noaux_tc: bias only picks experts */
    }
    /* top-K by biased score (selection sort: K is 6 of 384) */
    for (int k = 0; k < K; k++) {
        int best = -1;
        for (int e = 0; e < n_experts; e++) {
            int taken = 0;
            for (int j = 0; j < k; j++) if (idx[j] == e) { taken = 1; break; }
            if (!taken && (best < 0 || biased[e] > biased[best])) best = e;
        }
        idx[k] = best;
        w[k] = scores[best];                               /* weights from the UNbiased scores */
    }
    if (K > 1) {                                           /* norm_topk_prob */
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += w[k];
        for (int k = 0; k < K; k++) w[k] /= (sum + 1e-20f);
    }
    for (int k = 0; k < K; k++) w[k] *= route_scale;
    free(scores);
    free(biased);
    return K;
}

void ds41_hc_mixes(const float *x, const float *hc_fn, int hc, int dim, float norm_eps, float *mixes) {
    int n = hc * dim, mix_hc = (2 + hc) * hc;
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + norm_eps);
    for (int m = 0; m < mix_hc; m++) {
        const float *row = hc_fn + (size_t)m * n;
        double acc = 0.0;
        for (int i = 0; i < n; i++) acc += (double)row[i] * x[i];
        mixes[m] = (float)acc * r;
    }
}

static inline float ds41_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void ds41_hc_split_sinkhorn(const float *mixes, const float *hc_scale3, const float *hc_base,
                            int hc, int iters, float eps, float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++) pre[j]  = ds41_sigmoid(mixes[j] * hc_scale3[0] + hc_base[j]) + eps;
    for (int j = 0; j < hc; j++) post[j] = 2.0f * ds41_sigmoid(mixes[hc + j] * hc_scale3[1] + hc_base[hc + j]);
    for (int j = 0; j < hc; j++)
        for (int k = 0; k < hc; k++) {
            int m = 2 * hc + j * hc + k;
            comb[j * hc + k] = mixes[m] * hc_scale3[2] + hc_base[m];
        }
    /* comb = softmax(comb, -1) + eps */
    for (int j = 0; j < hc; j++) {
        float mx = comb[j * hc];
        for (int k = 1; k < hc; k++) if (comb[j * hc + k] > mx) mx = comb[j * hc + k];
        float sum = 0.0f;
        for (int k = 0; k < hc; k++) { comb[j * hc + k] = expf(comb[j * hc + k] - mx); sum += comb[j * hc + k]; }
        for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] / sum + eps;
    }
    /* comb /= (col_sum + eps); then (iters-1) x { row normalize; col normalize } */
    for (int it = 0; it < iters; it++) {
        if (it > 0) {
            for (int j = 0; j < hc; j++) {
                float s = 0.0f;
                for (int k = 0; k < hc; k++) s += comb[j * hc + k];
                for (int k = 0; k < hc; k++) comb[j * hc + k] /= (s + eps);
            }
        }
        for (int k = 0; k < hc; k++) {
            float s = 0.0f;
            for (int j = 0; j < hc; j++) s += comb[j * hc + k];
            for (int j = 0; j < hc; j++) comb[j * hc + k] /= (s + eps);
        }
    }
}

void ds41_hc_pre(const float *x, const float *pre, int hc, int dim, float *y) {
    for (int d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (int c = 0; c < hc; c++) acc += pre[c] * x[(size_t)c * dim + d];
        y[d] = acc;
    }
}

void ds41_hc_post(const float *y, const float *residual, const float *post, const float *comb,
                  int hc, int dim, float *out) {
    for (int c = 0; c < hc; c++)
        for (int d = 0; d < dim; d++) {
            float acc = post[c] * y[d];
            for (int k = 0; k < hc; k++) acc += comb[c * hc + k] * residual[(size_t)k * dim + d];
            out[(size_t)c * dim + d] = acc;
        }
}

void ds41_rope_inv_freq(int dim, float base, float factor, int beta_fast, int beta_slow,
                        int original_seq_len, float *inv_freq) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) inv_freq[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
    if (original_seq_len > 0) {
        /* corrected_dim(rot) = dim * ln(L / (rot * 2pi)) / (2 ln base) */
        double lo = dim * log((double)original_seq_len / (beta_fast * 2.0 * M_PI)) / (2.0 * log((double)base));
        double hi = dim * log((double)original_seq_len / (beta_slow * 2.0 * M_PI)) / (2.0 * log((double)base));
        int low  = (int)floor(lo); if (low < 0) low = 0;
        int high = (int)ceil(hi);  if (high > dim - 1) high = dim - 1;
        double den = (high - low) > 1e-3 ? (double)(high - low) : 1e-3;
        for (int i = 0; i < half; i++) {
            double ramp = ((double)i - low) / den;
            if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
            double smooth = 1.0 - ramp;
            inv_freq[i] = (float)(inv_freq[i] / factor * (1.0 - smooth) + inv_freq[i] * smooth);
        }
    }
}

/* ---- Engram ------------------------------------------------------------------ */
static int ds41_read_exact(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n ? 0 : -1; }

int ds41_engram_meta_load(const char *path, ds41_engram_meta *m) {
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t hdr[8];
    if (ds41_read_exact(f, magic, 8) || memcmp(magic, "DS41ENGR", 8) != 0) { fclose(f); return -2; }
    if (ds41_read_exact(f, hdr, sizeof hdr) || hdr[0] != 1) { fclose(f); return -3; }
    m->vocab_size = hdr[1]; m->compressed_vocab_size = hdr[2]; m->n_layers = hdr[3];
    m->max_ngram = hdr[4]; m->n_heads = hdr[5]; m->pad_compressed_id = hdr[6];
    if (m->n_layers != DS41_ENGRAM_N_LAYERS || m->max_ngram != DS41_ENGRAM_MAX_NGRAM ||
        m->n_heads != DS41_ENGRAM_N_HEADS) { fclose(f); return -4; }
    int rc = 0;
    rc |= ds41_read_exact(f, m->layer_ids, sizeof(int32_t) * m->n_layers);
    rc |= ds41_read_exact(f, m->multipliers, sizeof(int64_t) * m->n_layers * m->max_ngram);
    rc |= ds41_read_exact(f, m->primes, sizeof(int64_t) * m->n_layers * (m->max_ngram - 1) * m->n_heads);
    rc |= ds41_read_exact(f, m->offsets, sizeof(int64_t) * m->n_layers * (m->max_ngram - 1) * m->n_heads);
    m->token_map = (int32_t *)malloc(sizeof(int32_t) * m->vocab_size);
    rc |= ds41_read_exact(f, m->token_map, sizeof(int32_t) * m->vocab_size);
    fclose(f);
    if (rc) { ds41_engram_meta_free(m); return -5; }
    return 0;
}

void ds41_engram_meta_free(ds41_engram_meta *m) {
    free(m->token_map);
    m->token_map = NULL;
}

void ds41_engram_hash_compressed(const ds41_engram_meta *m, int layer_idx, const int32_t *history, int64_t *rows) {
    /* products can't overflow: id < compressed_vocab, multiplier < 2^63 / compressed_vocab. */
    int64_t rolling = 0;
    for (uint32_t s = 0; s < m->max_ngram; s++) {
        int64_t tok = history[s] < 0 ? (int64_t)m->pad_compressed_id : (int64_t)history[s];
        int64_t prod = tok * m->multipliers[layer_idx][s];
        if (s == 0) rolling = prod; else rolling ^= prod;
        if (s == 0) continue;
        /* after XOR-ing in the s-th lookback, `rolling` hashes the (s+1)-gram */
        for (uint32_t h = 0; h < m->n_heads; h++) {
            int64_t p = m->primes[layer_idx][s - 1][h];
            int64_t r = rolling % p;                 /* rolling >= 0, so C `%` == torch `%` here */
            rows[(s - 1) * m->n_heads + h] = r + m->offsets[layer_idx][(s - 1) * m->n_heads + h];
        }
    }
}

void ds41_engram_hash(const ds41_engram_meta *m, int layer_idx, const int32_t *token_history, int64_t *rows) {
    int32_t comp[DS41_ENGRAM_MAX_NGRAM];
    for (uint32_t s = 0; s < m->max_ngram; s++)
        comp[s] = token_history[s] < 0 ? -1 : m->token_map[token_history[s]];
    ds41_engram_hash_compressed(m, layer_idx, comp, rows);
}

void ds41_engram_row_dequant(const uint8_t *row_fp8, const uint8_t *row_scales, float *out) {
    for (int b = 0; b < DS41_ENGRAM_HEAD_DIM / DS41_FP8_BLOCK; b++) {
        float scale = ds41_e8m0_to_f32(row_scales[b]);
        for (int i = 0; i < DS41_FP8_BLOCK; i++)
            out[b * DS41_FP8_BLOCK + i] = ds41_e4m3_to_f32(row_fp8[b * DS41_FP8_BLOCK + i]) * scale;
    }
}

void ds41_engram_apply(float *h, const float *key, const float *value, const float *qk_weight,
                       int dim, float norm_eps) {
    /* Engram.forward: rstd = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps);
       dot = sum(h * qk_weight * key) * rstd / sqrt(dim); gate = sigmoid(copysign(sqrt(max(|dot|,1e-6)), dot)) */
    double hh = 0.0, kk = 0.0, dot = 0.0;
    for (int i = 0; i < dim; i++) {
        hh += (double)h[i] * h[i];
        kk += (double)key[i] * key[i];
        dot += (double)h[i] * qk_weight[i] * key[i];
    }
    double rstd = 1.0 / sqrt(hh / dim + norm_eps) / sqrt(kk / dim + norm_eps);
    double d = dot * rstd / sqrt((double)dim);
    double mag = fabs(d) < 1e-6 ? 1e-6 : fabs(d);
    float gate = ds41_sigmoid((float)copysign(sqrt(mag), d));
    for (int i = 0; i < dim; i++) h[i] += gate * value[i];
}

#endif /* DS41_OPS_IMPL */
