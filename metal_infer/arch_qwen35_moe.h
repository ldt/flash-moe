// arch_qwen35_moe.h — Qwen3.5-397B-A17B architecture profile.
//
// These are the constants the engine shipped with; they are unchanged, only
// moved out of infer.m so a second model can be built from the same source.

#ifndef FLASH_MOE_ARCH_QWEN35_MOE_H
#define FLASH_MOE_ARCH_QWEN35_MOE_H

#define ARCH_NAME           "Qwen3.5-397B-A17B"

#define HIDDEN_DIM          4096
#define NUM_LAYERS          60
#define NUM_ATTN_HEADS      32
#define NUM_KV_HEADS        2
#define HEAD_DIM            256
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         512
#define NUM_EXPERTS_PER_TOK 10
#define MOE_INTERMEDIATE    1024
#define SHARED_INTERMEDIATE 1024
#define FULL_ATTN_INTERVAL  4
#define GROUP_SIZE          64
#define BITS                4

// 45 GatedDeltaNet layers interleaved with 15 global-attention layers.
#define ARCH_HAS_LINEAR_ATTN  1
#define ARCH_SWA_WINDOW       0
#define NUM_FULL_ATTN_LAYERS  15
#define NUM_LINEAR_LAYERS     45

// Linear attention (GatedDeltaNet) constants
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128   // head_k_dim
#define LINEAR_VALUE_DIM    128   // head_v_dim
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 8192
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 12288
#define CONV_KERNEL_SIZE    4

// Full attention constants
#define ROPE_THETA          10000000.0f
#define PARTIAL_ROTARY      0.25f
#define ROTARY_DIM          64   // HEAD_DIM * PARTIAL_ROTARY, as an integer literal
#define ARCH_ROPE_SCALING   ARCH_ROPE_NONE

// q_proj emits [q | gate] per head; the gate is per-channel, sigmoid.
#define ARCH_ATTN_GATE      ARCH_GATE_FUSED_SIGMOID

// Router: softmax over all experts, then top-k, then renormalize.
#define ARCH_ROUTER_SIGMOID 0

// Expert packed binary layout
#define EXPERT_SIZE         7077888

// 2-bit expert layout (from repack_experts_2bit.py)
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

// KV cache maximum context length
#define MAX_SEQ_LEN 1048576  // 1M context — only 15 full-attn layers need KV cache, ~15GB at max
#define GPU_KV_SEQ  8192     // GPU KV buffer pre-allocation (grows if exceeded, falls back to CPU attn)

// Special tokens
#define EOS_TOKEN_1         248046
#define EOS_TOKEN_2         248044
#define THINK_START_TOKEN   248068  // <think>
#define THINK_END_TOKEN     248069  // </think>

#define ARCH_DEFAULT_MODEL_DIR "packed_experts"

#endif  // FLASH_MOE_ARCH_QWEN35_MOE_H
