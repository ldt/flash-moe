// arch_runtime.h — small architecture-dependent kernels that are cheap
// enough to keep on the CPU: rope frequency tables, attention output gating
// and the router activation.
//
// Everything here compiles down to the previous behaviour when the selected
// profile is Qwen3.5 (no rope scaling, sigmoid per-channel gate, softmax
// router), so the default build is unchanged.

#ifndef FLASH_MOE_ARCH_RUNTIME_H
#define FLASH_MOE_ARCH_RUNTIME_H

#include <math.h>

// Defined later in infer.m; the softmax router path calls it.
static void cpu_softmax(float *x, int dim);

// ---------------------------------------------------------------------------
// RoPE frequency tables
// ---------------------------------------------------------------------------
//
// Index 0 = unscaled frequencies (used by sliding-window layers, and by every
// layer when the model has no rope scaling). Index 1 = long-context scaled
// frequencies (YaRN) used by the global-attention layers.

// ROTARY_DIM is an integer literal in every profile, so this is a valid
// constant expression for the table sizes below.
// Defaults for models with a single rope configuration.
#ifndef ARCH_ROPE_THETA_GLOBAL
  #define ARCH_ROPE_THETA_GLOBAL ROPE_THETA
#endif
#ifndef ARCH_GLOBAL_ROTARY_DIM
  #define ARCH_GLOBAL_ROTARY_DIM ROTARY_DIM
#endif

// ROTARY_DIM is the maximum over layers, so it sizes both tables.
#define ARCH_ROPE_HALF (ROTARY_DIM / 2)

// Per-table geometry: table 0 is the sliding/default configuration,
// table 1 the global one (which is the only one that may carry scaling).
#define ARCH_ROPE_TABLE_THETA(t) ((t) ? ARCH_ROPE_THETA_GLOBAL : ROPE_THETA)
#define ARCH_ROPE_TABLE_DIM(t)   ((t) ? ARCH_GLOBAL_ROTARY_DIM : ROTARY_DIM)

static float g_rope_freq[2][ARCH_ROPE_HALF];
static float g_rope_mscale[2] = { 1.0f, 1.0f };
static int   g_rope_tables_ready = 0;

#if ARCH_ROPE_SCALING == ARCH_ROPE_YARN
// Dimension at which a rotation of `num_rot` cycles happens over the original
// context window — the YaRN correction range, same formula as the reference
// implementation. Uses the global table's geometry, the only one scaled.
static float arch_yarn_correction_dim(float num_rot) {
    return ((float)ARCH_GLOBAL_ROTARY_DIM *
            logf((float)ARCH_ROPE_ORIG_CTX / (num_rot * 2.0f * 3.14159265358979323846f))) /
           (2.0f * logf(ARCH_ROPE_THETA_GLOBAL));
}
#endif

static void arch_rope_init(void) {
    if (g_rope_tables_ready) return;

    for (int t = 0; t < 2; t++) {
        float theta = ARCH_ROPE_TABLE_THETA(t);
        int dim = ARCH_ROPE_TABLE_DIM(t);
        int half = dim / 2;
        for (int i = 0; i < half; i++) {
            g_rope_freq[t][i] = 1.0f / powf(theta, (float)(2 * i) / (float)dim);
        }
        for (int i = half; i < ARCH_ROPE_HALF; i++) g_rope_freq[t][i] = 0.0f;
        g_rope_mscale[t] = 1.0f;
    }

#if ARCH_ROPE_SCALING == ARCH_ROPE_YARN
    {
        int half = ARCH_GLOBAL_ROTARY_DIM / 2;
        float low  = floorf(arch_yarn_correction_dim(ARCH_ROPE_BETA_FAST));
        float high = ceilf(arch_yarn_correction_dim(ARCH_ROPE_BETA_SLOW));
        if (low < 0.0f) low = 0.0f;
        if (high > (float)(ARCH_GLOBAL_ROTARY_DIM - 1)) high = (float)(ARCH_GLOBAL_ROTARY_DIM - 1);
        // Guard against a degenerate range (identical bounds would divide by zero).
        float span = high - low;
        if (span < 1e-3f) span = 1e-3f;

        for (int i = 0; i < half; i++) {
            float extrapolation = g_rope_freq[1][i];
            float interpolation = extrapolation / ARCH_ROPE_YARN_FACTOR;

            float ramp = ((float)i - low) / span;
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            // ramp 0 near the high-frequency dims -> keep extrapolation there.
            float extrapolation_factor = 1.0f - ramp;

            g_rope_freq[1][i] = interpolation * (1.0f - extrapolation_factor) +
                                extrapolation * extrapolation_factor;
        }
        g_rope_mscale[1] = ARCH_ROPE_YARN_ATTN_FACTOR;
    }
#endif

    g_rope_tables_ready = 1;
}

// Which rope table a given layer uses.
#if ARCH_HAS_LINEAR_ATTN || ARCH_ROPE_SCALING == ARCH_ROPE_NONE
  #define ARCH_ROPE_TABLE(layer) 0
#else
  // Only the global layers see positions beyond the original window; the
  // sliding-window layers never look back more than ARCH_SWA_WINDOW tokens
  // and stay on unscaled frequencies.
  #define ARCH_ROPE_TABLE(layer) (ARCH_LAYER_IS_GLOBAL(layer) ? 1 : 0)
#endif

// ---------------------------------------------------------------------------
// Attention output gating
// ---------------------------------------------------------------------------

static inline float arch_softplus(float x) {
    // log1p(exp(x)) with the standard overflow-safe branch.
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static inline float arch_sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Apply the per-head output gate in place.
//   ARCH_GATE_FUSED_*  : gate_src holds NUM_ATTN_HEADS * HEAD_DIM values
//                        (one gate per channel), produced by the second half
//                        of q_proj.
//   ARCH_GATE_HEAD_*   : gate_src holds NUM_ATTN_HEADS values, one scalar per
//                        head, produced by a separate projection.
static inline void arch_apply_attn_gate(float *attn_out, const float *gate_src, int nheads) {
#if ARCH_ATTN_GATE == ARCH_GATE_NONE
    (void)attn_out; (void)gate_src; (void)nheads;
#elif ARCH_ATTN_GATE == ARCH_GATE_FUSED_SIGMOID
    int n = nheads * HEAD_DIM;
    for (int i = 0; i < n; i++) attn_out[i] *= arch_sigmoid_f(gate_src[i]);
#elif ARCH_ATTN_GATE == ARCH_GATE_FUSED_SOFTPLUS
    int n = nheads * HEAD_DIM;
    for (int i = 0; i < n; i++) attn_out[i] *= arch_softplus(gate_src[i]);
#elif ARCH_ATTN_GATE == ARCH_GATE_HEAD_SOFTPLUS
    for (int h = 0; h < nheads; h++) {
        float g = arch_softplus(gate_src[h]);
        float *oh = attn_out + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) oh[d] *= g;
    }
#else
    #error "unknown ARCH_ATTN_GATE"
#endif
}

// Does q_proj carry the gate in its second half?
#if ARCH_ATTN_GATE == ARCH_GATE_FUSED_SIGMOID || ARCH_ATTN_GATE == ARCH_GATE_FUSED_SOFTPLUS
  #define ARCH_GATE_FUSED_IN_QPROJ 1
  #define ARCH_GATE_DIM            (NUM_ATTN_HEADS * HEAD_DIM)
#elif ARCH_ATTN_GATE == ARCH_GATE_HEAD_SOFTPLUS
  #define ARCH_GATE_FUSED_IN_QPROJ 0
  #define ARCH_GATE_DIM            NUM_ATTN_HEADS
#else
  #define ARCH_GATE_FUSED_IN_QPROJ 0
  #define ARCH_GATE_DIM            0
#endif

#define ARCH_Q_PROJ_DIM (ARCH_GATE_FUSED_IN_QPROJ \
                         ? (NUM_ATTN_HEADS * HEAD_DIM * 2) \
                         : (NUM_ATTN_HEADS * HEAD_DIM))

// ---------------------------------------------------------------------------
// Router activation
// ---------------------------------------------------------------------------
//
// Qwen3.5 softmaxes the router logits before top-k. Laguna S gates each
// expert independently with a sigmoid, takes the top-k, then renormalizes the
// selected weights (which the caller does either way).

static inline void arch_router_activation(float *scores, int n) {
#if ARCH_ROUTER_SIGMOID
    for (int i = 0; i < n; i++) scores[i] = arch_sigmoid_f(scores[i]);
#else
    cpu_softmax(scores, n);
#endif
}

// ---------------------------------------------------------------------------
// Sliding-window helpers
// ---------------------------------------------------------------------------
//
// A sliding-window layer only attends to the last ARCH_LAYER_WINDOW(layer)
// positions. The KV cache stays a flat array: entries are appended until it
// holds 2*W of them, at which point the last W are moved to the front. That
// costs one memmove of W entries every W tokens and keeps the attention
// kernels reading a contiguous range, so nothing else has to change.

// First cache slot this layer may attend to, given the number of entries held.
static inline int arch_attn_start(int window, int cache_len) {
    if (window <= 0) return 0;
    return (cache_len > window) ? (cache_len - window) : 0;
}

#endif  // FLASH_MOE_ARCH_RUNTIME_H
