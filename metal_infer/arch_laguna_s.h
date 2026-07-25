// arch_laguna_s.h — poolside Laguna S 2.1 (118B total / ~8.5B active) profile.
//
// !! THIS FILE IS A TEMPLATE. REGENERATE IT BEFORE YOU TRUST IT. !!
//
//     uv run metal_infer/gen_arch_header.py
//         --model <path to the downloaded Laguna S weights>
//         --output metal_infer/arch_laguna_s.h
//
// gen_arch_header.py reads the real config.json, tokenizer_config.json and
// safetensors shapes and rewrites every number below, then defines
// ARCH_GENERATED so the build stops warning.
//
// Provenance of the values as committed:
//
//   Confirmed from poolside's release material and the model card summaries:
//     48 layers, 1 global : 3 sliding-window (window 512), 8 KV heads,
//     head_dim 128, 256 routed experts + 1 shared, top-10 routing with
//     sigmoid gating and renormalized weights, per-head softplus output
//     gating, vocab 100352 BPE, YaRN rope factor 128 with
//     attention_factor 1.4852030263919618 on the global layers.
//
//   Derived, NOT confirmed (gen_arch_header.py will correct them):
//     HIDDEN_DIM 4096 and MOE_INTERMEDIATE 768 — the pair that reproduces
//     both the 118B total parameter count and the ~64 GB size of the 4-bit
//     MLX conversion. NUM_ATTN_HEADS 32. ROPE_THETA. All token ids.

#ifndef FLASH_MOE_ARCH_LAGUNA_S_H
#define FLASH_MOE_ARCH_LAGUNA_S_H

#define ARCH_NAME           "Laguna-S-2.1"

#define HIDDEN_DIM          4096
#define NUM_LAYERS          48
#define NUM_ATTN_HEADS      32
#define NUM_KV_HEADS        8
#define HEAD_DIM            128
#define VOCAB_SIZE          100352
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 10
#define MOE_INTERMEDIATE    768
#define SHARED_INTERMEDIATE 768
#define FULL_ATTN_INTERVAL  4     // every 4th layer is global, the rest are SWA
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
#define ROPE_THETA            1000000.0f   // PLACEHOLDER — regenerate
#define PARTIAL_ROTARY        1.0f
#define ROTARY_DIM            128  // HEAD_DIM * PARTIAL_ROTARY, as an integer literal
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

// Packed expert layout, 4-bit:
//   gate/up: [768, 4096] -> 768*4096/2      = 1572864 B each
//            scales/biases [768, 64] bf16   =   98304 B each
//   down:    [4096, 768] -> 4096*768/2      = 1572864 B
//            scales/biases [4096, 12] bf16  =   98304 B each
#define EXPERT_SIZE         5308416

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
#define EOS_TOKEN_1         100257
#define EOS_TOKEN_2         100257
#define THINK_START_TOKEN   -1
#define THINK_END_TOKEN     -1

#define ARCH_DEFAULT_MODEL_DIR "packed_experts_laguna"

#ifndef ARCH_GENERATED
#warning "arch_laguna_s.h holds placeholder values (see header comment). Run gen_arch_header.py against your downloaded weights before trusting the output."
#endif

#endif  // FLASH_MOE_ARCH_LAGUNA_S_H
