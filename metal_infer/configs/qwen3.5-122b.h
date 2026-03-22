/*
 * Model config: Qwen3.5-122B-A10B
 *
 * 122B total parameters, 10B active per token.
 * 48 layers (36 GatedDeltaNet + 12 full attention), hidden_dim=3072, 256 experts.
 *
 * Compared to the 397B model:
 *   - 48 layers (vs 60), 3072 hidden dim (vs 4096), 256 experts (vs 512)
 *   - 8 routed experts/token (vs 10), same 1024 expert intermediate
 *   - Same attention head config, same vocab, same context length
 *   - ~62GB on disk at 4-bit (vs 209GB) — fits better in 48GB page cache
 *
 * To use: make MODEL=qwen3.5-122b
 * Or:     cp configs/qwen3.5-122b.h model_config.h && make
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define MODEL_NAME "Qwen3.5-122B-A10B"

// Core architecture
#define HIDDEN_DIM          3072
#define NUM_LAYERS          48
#define NUM_ATTN_HEADS      32      // Q heads for full attention
#define NUM_KV_HEADS        2       // KV heads for full attention (GQA 16:1)
#define HEAD_DIM            256
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f

// MoE configuration
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 8       // top-K routed (runtime --k overrides active count)
#define MOE_INTERMEDIATE    1024    // expert FFN intermediate dimension
#define SHARED_INTERMEDIATE 1024    // shared expert intermediate dimension

// Layer structure: every FULL_ATTN_INTERVAL-th layer is full attention
#define FULL_ATTN_INTERVAL  4       // layers 3,7,11,...,47 are full attention
#define NUM_FULL_ATTN_LAYERS 12     // 48 / 4
#define NUM_LINEAR_LAYERS    36     // 48 - 12

// Linear attention (GatedDeltaNet) head configuration
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128     // head_k_dim
#define LINEAR_VALUE_DIM    128     // head_v_dim
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 8192
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 12288
#define CONV_KERNEL_SIZE    4

// Full attention constants
#define ROPE_THETA          10000000.0f
#define PARTIAL_ROTARY      0.25f
#define ROTARY_DIM          (int)(HEAD_DIM * PARTIAL_ROTARY)  // 64

// Quantization
#define GROUP_SIZE          64
#define BITS                4

// Expert packed binary layout — 4-bit
// gate/up: [1024, 3072/8=384] u32 + [1024, 3072/64=48] bf16 x2
// down:    [3072, 1024/8=128] u32 + [3072, 1024/64=16] bf16 x2
// Per component: weight + scales + biases
//   gate: 1572864 + 98304 + 98304 = 1769472
//   up:   1572864 + 98304 + 98304 = 1769472
//   down: 1572864 + 98304 + 98304 = 1769472
#define EXPERT_SIZE         5308416

// Expert packed binary layout — 2-bit
// Weight arrays halve: 16 vals per u32 instead of 8
//   gate: 786432 + 98304 + 98304 = 983040
//   up:   786432 + 98304 + 98304 = 983040
//   down: 786432 + 98304 + 98304 = 983040
#define EXPERT_SIZE_2BIT    2949120
// 2-bit offsets within expert blob
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  786432
#define GATE_B_OFF_2  884736
#define UP_W_OFF_2    983040
#define UP_S_OFF_2    1769472
#define UP_B_OFF_2    1867776
#define DOWN_W_OFF_2  1966080
#define DOWN_S_OFF_2  2752512
#define DOWN_B_OFF_2  2850816

// KV cache
#define MAX_SEQ_LEN 1048576  // 1M context
#define GPU_KV_SEQ  8192     // GPU KV buffer pre-allocation

// Special tokens (shared across Qwen3.5 family)
#define EOS_TOKEN_1         248046
#define EOS_TOKEN_2         248044
#define THINK_START_TOKEN   248068  // <think>
#define THINK_END_TOKEN     248069  // </think>

#endif // MODEL_CONFIG_H
