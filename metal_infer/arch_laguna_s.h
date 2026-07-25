// arch_laguna_s.h — poolside Laguna S 2.1 (118B total / ~8.5B active).
//
// The numbers below are now the REAL ones, read from poolside's config.json
// (archived at docs/laguna_s_2.1_config.json) and cross-checked against
// modeling_laguna.py. They are no longer guesses.
//
// !! THE ENGINE CANNOT RUN THIS MODEL CORRECTLY YET. !!
//
// Reading the real config surfaced four structural features the port does
// not implement. Each would silently produce wrong output rather than fail:
//
//   1. Per-layer head counts. Global layers have 48 query heads, sliding
//      layers have 72 (config: num_attention_heads_per_layer). NUM_ATTN_HEADS
//      is a single compile-time constant, so q_proj/o_proj dims and
//      heads_per_kv are wrong on one of the two layer kinds.
//
//   2. Per-layer rope. Global: theta 500000, YaRN factor 128,
//      partial_rotary 0.5 (rotary_dim 64). Sliding: theta 10000, no scaling,
//      partial_rotary 1.0 (rotary_dim 128). arch_runtime.h has two frequency
//      tables but they share one ROPE_THETA and one ROTARY_DIM.
//
//   3. Layer 0 is a dense MLP (intermediate_size 12288), not MoE
//      (config: mlp_only_layers=[0]). The engine assumes every layer routes
//      to experts. Only 47 of 48 layers are sparse.
//
//   4. Global layers sit at i % 4 == 0 — layer 0 is global — not at
//      (i+1) % 4 == 0 as ARCH_LAYER_IS_GLOBAL assumes.
//
// Smaller deltas, same category:
//   - routed expert output is scaled by moe_routed_scaling_factor (2.5)
//     before the shared expert is added;
//   - the shared expert is added ungated (Qwen applies a sigmoid gate);
//   - the router adds e_score_correction_bias to the SELECTION scores only,
//     keeping the returned weights unbiased (zero in this checkpoint, but
//     part of the architecture).
//
// What the port did get right: sliding window 512, per-head softplus output
// gate via self_attn.g_proj, sigmoid router with renormalized top-10 of 256,
// YaRN on the global layers, q/k RMSNorm per head, non-interleaved rope
// pairing, and the 5,308,416-byte packed expert layout.

#ifndef FLASH_MOE_ARCH_LAGUNA_S_H
#define FLASH_MOE_ARCH_LAGUNA_S_H

#define ARCH_NAME           "Laguna-S-2.1"

#define HIDDEN_DIM          3072
#define NUM_LAYERS          48
#define NUM_ATTN_HEADS      72   // MAX over layers: sliding layers use 72
#define ARCH_GLOBAL_NUM_HEADS 48 // global layers use 48
#define ARCH_LAYER_NUM_HEADS(i) (ARCH_LAYER_IS_GLOBAL(i) ? ARCH_GLOBAL_NUM_HEADS : NUM_ATTN_HEADS)
#define NUM_KV_HEADS        8
#define HEAD_DIM            128
#define VOCAB_SIZE          100352
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 10
#define MOE_INTERMEDIATE    1024
#define SHARED_INTERMEDIATE 1024
#define FULL_ATTN_INTERVAL  4     // one global layer per group of four
#define ARCH_GLOBAL_PHASE   0     // ...and it is the FIRST: layer 0 is global
#define GROUP_SIZE          64
#define BITS                4

// Every layer is softmax attention — there is no GatedDeltaNet path here.
// 12 global layers, 36 sliding-window layers with a 512-token window.
#define ARCH_HAS_LINEAR_ATTN  0
#define ARCH_SWA_WINDOW       512
#define NUM_FULL_ATTN_LAYERS  48
#define NUM_LINEAR_LAYERS     1   // unused; kept at 1 so the arrays stay valid

// Linear attention constants are dead code for this profile. They are kept
// at Qwen's values so the scratch buffers they size stay comfortably large
// rather than accidentally undersized; nothing reads them.
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128
#define LINEAR_VALUE_DIM    128
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM)
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE)
#define CONV_KERNEL_SIZE    4

// RoPE. Global layers use YaRN (factor 128 over an 8192-token base window,
// which is exactly the 1M advertised context). Sliding-window layers only
// ever look back 512 tokens, so they use unscaled rope.
// Sliding layers: theta 10000, unscaled, partial_rotary 1.0 -> 128 dims.
// Global layers:  theta 500000, YaRN x128, partial_rotary 0.5 -> 64 dims.
#define ROPE_THETA            10000.0f    // sliding layers (rope table 0)
#define ARCH_ROPE_THETA_GLOBAL 500000.0f  // global layers  (rope table 1)
#define PARTIAL_ROTARY        1.0f
#define ROTARY_DIM            128   // MAX over layers (sliding); global use 64
#define ARCH_GLOBAL_ROTARY_DIM 64
#define ARCH_LAYER_ROTARY_DIM(i) (ARCH_LAYER_IS_GLOBAL(i) ? ARCH_GLOBAL_ROTARY_DIM : ROTARY_DIM)
#define ARCH_ROPE_SCALING     ARCH_ROPE_YARN
#define ARCH_ROPE_YARN_FACTOR 128.0f
#define ARCH_ROPE_YARN_ATTN_FACTOR 1.4852030263919618f
#define ARCH_ROPE_ORIG_CTX    8192
#define ARCH_ROPE_BETA_FAST   32.0f
#define ARCH_ROPE_BETA_SLOW   1.0f

// Per-head softplus output gating.
#define ARCH_ATTN_GATE      ARCH_GATE_HEAD_SOFTPLUS

// Router: sigmoid per expert (not softmax), then top-k, then renormalize.
#define ARCH_ROUTER_SIGMOID 1

// Layer 0 is a dense MLP (config: mlp_only_layers=[0], intermediate_size).
// Only layers 1..47 route to experts.
#define ARCH_LAYER_IS_MOE(i)    ((i) != 0)
#define ARCH_DENSE_INTERMEDIATE 12288

// Routed expert output is scaled before the shared expert is added, and the
// shared expert is added ungated (Qwen applies a sigmoid gate to it).
#define ARCH_ROUTED_SCALING     2.5f
#define ARCH_SHARED_EXPERT_GATED 0

// Packed expert layout, 4-bit:
//   gate/up: [768, 4096] -> 768*4096/2      = 1572864 B each
//            scales/biases [768, 64] bf16   =   98304 B each
//   down:    [4096, 768] -> 4096*768/2      = 1572864 B
//            scales/biases [4096, 12] bf16  =   98304 B each
#define EXPERT_SIZE         5308416  // 3072x1024 at 4-bit, group 64

// 2-bit variant (repack_experts_2bit.py). Kept for experimentation; 2-bit
// broke JSON/tool calling on Qwen and this is a coding model, so 4-bit is
// the configuration to use.
#define EXPERT_SIZE_2BIT    2949120
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  786432
#define GATE_B_OFF_2  884736
#define UP_W_OFF_2    983040
#define UP_S_OFF_2    1769472
#define UP_B_OFF_2    1867776
#define DOWN_W_OFF_2  1966080
#define DOWN_S_OFF_2  2752512
#define DOWN_B_OFF_2  2850816

// KV cache. Only the 12 global layers scale with context; the 36 SWA layers
// are pinned at 2*512 slots each (~150 MB total). MAX_SEQ_LEN is the CPU-side
// cap for global layers: 12 layers * 8 KV heads * 128 dim * 2 * 4 B = 96 KB
// per token, so 65536 tokens ~ 6.3 GB. Raise it if you have the headroom and
// lower it on a 36 GB machine if you want more page cache for experts.
#define MAX_SEQ_LEN 65536
#define GPU_KV_SEQ  8192

// Special tokens — PLACEHOLDERS, regenerate before use.
#define EOS_TOKEN_1         2
#define EOS_TOKEN_2         24
#define THINK_START_TOKEN   -1
#define THINK_END_TOKEN     -1

#define ARCH_DEFAULT_MODEL_DIR "packed_experts_laguna"

#ifndef ARCH_GENERATED
#warning "Laguna S support is INCOMPLETE: per-layer head counts, per-layer rope, the dense layer 0 and the global-layer phase are not implemented. See the comment at the top of arch_laguna_s.h. Output will be wrong."
#endif

#endif  // FLASH_MOE_ARCH_LAGUNA_S_H
