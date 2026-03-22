/*
 * Model config: Qwen3.5-397B-A17B
 *
 * 397B total parameters, 17B active per token.
 * 60 layers (45 GatedDeltaNet + 15 full attention), hidden_dim=4096, 512 experts.
 *
 * To use: make MODEL=qwen3.5-397b
 * Or:     cp configs/qwen3.5-397b.h model_config.h && make
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define MODEL_NAME "Qwen3.5-397B-A17B"

// Core architecture
#define HIDDEN_DIM          4096
#define NUM_LAYERS          60
#define NUM_ATTN_HEADS      32      // Q heads for full attention
#define NUM_KV_HEADS        2       // KV heads for full attention (GQA 16:1)
#define HEAD_DIM            256
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f

// MoE configuration
#define NUM_EXPERTS         512
#define NUM_EXPERTS_PER_TOK 10      // top-K routed (runtime --k overrides active count)
#define MOE_INTERMEDIATE    1024    // expert FFN intermediate dimension
#define SHARED_INTERMEDIATE 1024    // shared expert intermediate dimension

// Layer structure: every FULL_ATTN_INTERVAL-th layer is full attention
#define FULL_ATTN_INTERVAL  4       // layers 3,7,11,...,59 are full attention
#define NUM_FULL_ATTN_LAYERS 15     // 60 / 4
#define NUM_LINEAR_LAYERS    45     // 60 - 15

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

// Expert packed binary layout — 4-bit (from repack_experts.py)
// gate/up: [MOE_INTERMEDIATE, HIDDEN_DIM/8] u32 + [MOE_INTERMEDIATE, HIDDEN_DIM/64] bf16 x2
// down:    [HIDDEN_DIM, MOE_INTERMEDIATE/8] u32 + [HIDDEN_DIM, MOE_INTERMEDIATE/64] bf16 x2
#define EXPERT_SIZE         7077888

// Expert packed binary layout — 2-bit (from repack_experts_2bit.py)
// Weight arrays halve: 16 vals per u32 instead of 8
#define EXPERT_SIZE_2BIT    3932160
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  1048576
#define GATE_B_OFF_2  1179648
#define UP_W_OFF_2    1310720
#define UP_S_OFF_2    2359296
#define UP_B_OFF_2    2490368
#define DOWN_W_OFF_2  2621440
#define DOWN_S_OFF_2  3670016
#define DOWN_B_OFF_2  3801088

// KV cache
#define MAX_SEQ_LEN 1048576  // 1M context
#define GPU_KV_SEQ  8192     // GPU KV buffer pre-allocation

// Special tokens (shared across Qwen3.5 family)
#define EOS_TOKEN_1         248046
#define EOS_TOKEN_2         248044
#define THINK_START_TOKEN   248068  // <think>
#define THINK_END_TOKEN     248069  // </think>

#endif // MODEL_CONFIG_H
