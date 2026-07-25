// arch.h — model architecture profile selection for the flash-moe engine.
//
// The engine used to hardcode Qwen3.5-397B-A17B's constants at the top of
// infer.m. They now live in one header per model family, selected at build
// time:
//
//     make                      # Qwen3.5-397B-A17B (default, unchanged)
//     make MODEL=laguna_s       # poolside Laguna S 2.1 (118B-A8B)
//
// A profile header must define the raw model constants (HIDDEN_DIM, layer
// counts, expert geometry, rope, special tokens). Everything derivable —
// packed expert byte layout, per-layer attention kind, KV sizing — is
// computed here so the two profiles cannot drift apart.
//
// Adding a model: copy arch_qwen35_moe.h, fill in the constants, add a
// branch below. For Laguna the header is generated from the real
// config.json by gen_arch_header.py — do not hand-edit the numbers.

#ifndef FLASH_MOE_ARCH_H
#define FLASH_MOE_ARCH_H

// ---------------------------------------------------------------------------
// Enumerations a profile picks from (must precede the profile include)
// ---------------------------------------------------------------------------

// How attention output is gated, per head.
#define ARCH_GATE_NONE            0  // no gating
#define ARCH_GATE_FUSED_SIGMOID   1  // q_proj emits [q | gate]; gate = sigmoid, per channel
#define ARCH_GATE_FUSED_SOFTPLUS  2  // q_proj emits [q | gate]; gate = softplus, per channel
#define ARCH_GATE_HEAD_SOFTPLUS   3  // separate [n_heads] gate projection; gate = softplus, per head

// How RoPE frequencies are scaled for long context.
#define ARCH_ROPE_NONE  0
#define ARCH_ROPE_YARN  1

#if defined(MODEL_ARCH_LAGUNA_S)
  #include "arch_laguna_s.h"
#else
  #include "arch_qwen35_moe.h"
#endif

// ---------------------------------------------------------------------------
// Attention layer taxonomy
// ---------------------------------------------------------------------------
//
// Two things vary per layer: whether the layer is softmax attention (as
// opposed to a linear-attention/GatedDeltaNet layer) and, if it is, how far
// back it may look.
//
//   ARCH_LAYER_IS_ATTN(i)  1 -> softmax attention layer with a KV cache
//                          0 -> linear attention (GatedDeltaNet) layer
//   ARCH_ATTN_IDX(i)       dense index of layer i among attention layers
//   ARCH_LINEAR_IDX(i)     dense index of layer i among linear layers
//   ARCH_LAYER_WINDOW(i)   sliding-window size, 0 = unbounded (global)
//
// Qwen3.5 interleaves 45 GatedDeltaNet layers with 15 global-attention
// layers (every 4th). Laguna S has no linear attention at all: every layer
// is softmax attention, and 3 out of 4 are sliding-window.

#if ARCH_HAS_LINEAR_ATTN
  #define ARCH_LAYER_IS_ATTN(i)  (((i) + 1) % FULL_ATTN_INTERVAL == 0)
  #define ARCH_ATTN_IDX(i)       (((i) + 1) / FULL_ATTN_INTERVAL - 1)
  #define ARCH_LINEAR_IDX(i)     ((i) - ((i) + 1) / FULL_ATTN_INTERVAL)
  #define ARCH_LAYER_WINDOW(i)   (0)
#else
  #define ARCH_LAYER_IS_ATTN(i)  (1)
  #define ARCH_ATTN_IDX(i)       (i)
  #define ARCH_LINEAR_IDX(i)     (-1)
  // Global layers are the ones the profile marks as full attention; every
  // other layer is sliding-window.
  #define ARCH_LAYER_WINDOW(i)   (ARCH_LAYER_IS_GLOBAL(i) ? 0 : ARCH_SWA_WINDOW)
#endif

// Inverse of ARCH_ATTN_IDX: which model layer owns attention slot s.
#if ARCH_HAS_LINEAR_ATTN
  #define ARCH_ATTN_SLOT_LAYER(s) ((s) * FULL_ATTN_INTERVAL + FULL_ATTN_INTERVAL - 1)
#else
  #define ARCH_ATTN_SLOT_LAYER(s) (s)
#endif

// Which layers are global attention. Models differ in the phase of the
// pattern: Qwen3.5 puts the global layer last in each group of four
// ((i+1) % 4 == 0), Laguna S puts it first (i % 4 == 0, so layer 0 is
// global). ARCH_GLOBAL_PHASE is that offset.
#ifndef ARCH_GLOBAL_PHASE
  #define ARCH_GLOBAL_PHASE (FULL_ATTN_INTERVAL - 1)
#endif
#ifndef ARCH_LAYER_IS_GLOBAL
  #define ARCH_LAYER_IS_GLOBAL(i) ((i) % FULL_ATTN_INTERVAL == ARCH_GLOBAL_PHASE)
#endif

// Query-head count can vary per layer (Laguna S: 48 on global layers, 72 on
// sliding ones). NUM_ATTN_HEADS is the MAXIMUM over layers — every buffer is
// sized from it — while ARCH_LAYER_NUM_HEADS(i) is what a given layer uses.
#ifndef ARCH_LAYER_NUM_HEADS
  #define ARCH_LAYER_NUM_HEADS(i) NUM_ATTN_HEADS
#endif

// Rotary width can also vary per layer (partial_rotary_factor differs between
// the global and sliding rope configurations). ROTARY_DIM is the maximum.
#ifndef ARCH_LAYER_ROTARY_DIM
  #define ARCH_LAYER_ROTARY_DIM(i) ROTARY_DIM
#endif

// Most models route every layer to experts; Laguna S makes layer 0 a plain
// dense MLP (config: mlp_only_layers). ARCH_DENSE_INTERMEDIATE is that
// layer's FFN width.
#ifndef ARCH_LAYER_IS_MOE
  #define ARCH_LAYER_IS_MOE(i) (1)
#endif
#ifndef ARCH_DENSE_INTERMEDIATE
  #define ARCH_DENSE_INTERMEDIATE SHARED_INTERMEDIATE
#endif

// Routed-expert output scaling, applied before the shared expert is added.
#ifndef ARCH_ROUTED_SCALING
  #define ARCH_ROUTED_SCALING 1.0f
#endif

// Whether the shared expert is gated by its own sigmoid projection (Qwen) or
// simply added (Laguna S).
#ifndef ARCH_SHARED_EXPERT_GATED
  #define ARCH_SHARED_EXPERT_GATED 1
#endif

// Number of KV caches to allocate, and how deep each one has to be. A
// sliding-window layer never needs more than 2*W slots: the cache is
// compacted back to the last W entries when it fills, which costs one
// memmove of W entries every W tokens.
#define ARCH_SWA_CACHE_SLOTS (2 * ARCH_SWA_WINDOW)

// ---------------------------------------------------------------------------
// Packed expert layout (4-bit, MLX-style {weight, scales, biases})
// ---------------------------------------------------------------------------
//
// Per expert, nine blobs in a fixed order. Weights are 4-bit nibbles packed
// 8-per-uint32; scales and biases are bf16, one pair per GROUP_SIZE inputs.
// repack_experts.py writes exactly this layout, so both sides derive the
// offsets from the same formulas.

#define ARCH_Q4_W_BYTES(out_f, in_f)  ((size_t)(out_f) * (size_t)(in_f) / 2)
#define ARCH_Q4_S_BYTES(out_f, in_f)  ((size_t)(out_f) * ((size_t)(in_f) / GROUP_SIZE) * 2)

#define GATE_W_BYTES  ARCH_Q4_W_BYTES(MOE_INTERMEDIATE, HIDDEN_DIM)
#define GATE_S_BYTES  ARCH_Q4_S_BYTES(MOE_INTERMEDIATE, HIDDEN_DIM)
#define DOWN_W_BYTES  ARCH_Q4_W_BYTES(HIDDEN_DIM, MOE_INTERMEDIATE)
#define DOWN_S_BYTES  ARCH_Q4_S_BYTES(HIDDEN_DIM, MOE_INTERMEDIATE)

#define GATE_W_OFF  ((size_t)0)
#define GATE_S_OFF  (GATE_W_OFF + GATE_W_BYTES)
#define GATE_B_OFF  (GATE_S_OFF + GATE_S_BYTES)
#define UP_W_OFF    (GATE_B_OFF + GATE_S_BYTES)
#define UP_S_OFF    (UP_W_OFF + GATE_W_BYTES)
#define UP_B_OFF    (UP_S_OFF + GATE_S_BYTES)
#define DOWN_W_OFF  (UP_B_OFF + GATE_S_BYTES)
#define DOWN_S_OFF  (DOWN_W_OFF + DOWN_W_BYTES)
#define DOWN_B_OFF  (DOWN_S_OFF + DOWN_S_BYTES)

#define ARCH_EXPERT_SIZE  (DOWN_B_OFF + DOWN_S_BYTES)

// 2-bit variant (repack_experts_2bit.py): same nine blobs, weights at 2 bits.
#define ARCH_Q2_W_BYTES(out_f, in_f)  ((size_t)(out_f) * (size_t)(in_f) / 4)
#define ARCH_EXPERT_SIZE_2BIT \
    (3 * (ARCH_Q2_W_BYTES(MOE_INTERMEDIATE, HIDDEN_DIM)) + \
     4 * GATE_S_BYTES + 2 * DOWN_S_BYTES)

// The profiles still spell EXPERT_SIZE out as a literal (it is load-bearing
// for the on-disk files); check the two agree at compile time.
_Static_assert(EXPERT_SIZE == ARCH_EXPERT_SIZE,
               "EXPERT_SIZE in the arch profile disagrees with the derived "
               "packed layout — expert geometry and packed files are out of sync");

// ---------------------------------------------------------------------------
// Sanity checks that catch a mis-generated profile before it corrupts output
// ---------------------------------------------------------------------------

_Static_assert(HIDDEN_DIM % GROUP_SIZE == 0, "HIDDEN_DIM must be a multiple of GROUP_SIZE");
_Static_assert(MOE_INTERMEDIATE % GROUP_SIZE == 0, "MOE_INTERMEDIATE must be a multiple of GROUP_SIZE");
_Static_assert(NUM_ATTN_HEADS % NUM_KV_HEADS == 0, "NUM_ATTN_HEADS must be a multiple of NUM_KV_HEADS");
_Static_assert(NUM_EXPERTS_PER_TOK <= NUM_EXPERTS, "top-k exceeds expert count");

#endif  // FLASH_MOE_ARCH_H
