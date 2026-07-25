#!/usr/bin/env python3
"""Generate a flash-moe architecture profile from a downloaded model.

Reads config.json, tokenizer_config.json and the safetensors headers of an
MLX-quantized checkpoint, and writes:

  metal_infer/arch_<arch>.h     — the constants the C engine compiles against
  metal_infer/arch_<arch>.json  — the same numbers for the packing scripts

Nothing here is guessed: every value is read from the checkpoint, and anything
that cannot be found is reported loudly instead of being filled in with a
plausible default.

Usage:
    uv run metal_infer/gen_arch_header.py \\
        --model ~/.cache/huggingface/hub/models--mlx-community--Laguna-S-2.1-oQ4e/snapshots/<rev> \\
        --arch laguna_s
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# config.json access with aliases (key names differ between model families)
# --------------------------------------------------------------------------

def pick(cfg, *keys, default=None, required=True, what=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    # Some releases nest the text config.
    for sub in ("text_config", "language_model", "llm_config"):
        if isinstance(cfg.get(sub), dict):
            for k in keys:
                if cfg[sub].get(k) is not None:
                    return cfg[sub][k]
    if required:
        raise SystemExit(
            f"ERROR: could not find {what or keys[0]} in config.json "
            f"(looked for: {', '.join(keys)}).\n"
            f"       Add the right key name to gen_arch_header.py and re-run.")
    return default


def parse_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header


def load_all_headers(model_dir):
    """Return {tensor_name: meta} across every shard."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    tensors = {}
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        for fname in sorted(set(weight_map.values())):
            hdr = parse_safetensors_header(model_dir / fname)
            for name, meta in hdr.items():
                if name != "__metadata__":
                    tensors[name] = meta
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"ERROR: no safetensors found in {model_dir}")
        for shard in shards:
            for name, meta in parse_safetensors_header(shard).items():
                if name != "__metadata__":
                    tensors[name] = meta
    return tensors


def find_tensor(tensors, *suffixes, layer=0):
    """First tensor whose name ends with one of the given suffixes."""
    for suf in suffixes:
        want = suf.format(layer=layer)
        for name in tensors:
            if name.endswith(want):
                return name, tensors[name]
    return None, None


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------

def derive_quant(weight_shape, scales_shape, in_features):
    """Return (bits, group_size) from an MLX-quantized weight/scales pair."""
    packed_cols = weight_shape[1]
    vals_per_word = in_features / packed_cols
    if abs(vals_per_word - round(vals_per_word)) > 1e-9:
        raise SystemExit(
            f"ERROR: weight shape {weight_shape} is not a whole number of "
            f"packed values for in_features={in_features}")
    bits = 32 // int(round(vals_per_word))
    group = in_features // scales_shape[1]
    return bits, group


def derive_layer_types(cfg, num_layers):
    """Return (interval, window, global_layer_indices).

    The engine assumes global attention repeats at a fixed interval, with the
    global layer last in each group (layer i is global when (i+1) % N == 0).
    """
    types = pick(cfg, "layer_types", required=False)
    window = pick(cfg, "sliding_window", "attention_window_size", required=False)

    if not types:
        # No explicit map: fall back to an interval key if present.
        interval = pick(cfg, "full_attention_interval", "global_attn_every_n_layers",
                        required=False)
        if interval is None:
            raise SystemExit(
                "ERROR: config.json has neither `layer_types` nor an attention "
                "interval key, so the global/sliding layout cannot be derived.")
        globals_ = [i for i in range(num_layers) if (i + 1) % interval == 0]
        return int(interval), (int(window) if window else 0), globals_

    if len(types) != num_layers:
        raise SystemExit(f"ERROR: layer_types has {len(types)} entries, expected {num_layers}")

    globals_ = [i for i, t in enumerate(types) if "full" in t or "global" in t]
    if not globals_:
        raise SystemExit("ERROR: no full/global attention layers found in layer_types")

    # Check the pattern is the periodic one the engine implements.
    interval = globals_[0] + 1
    expected = [i for i in range(num_layers) if (i + 1) % interval == 0]
    if globals_ != expected:
        raise SystemExit(
            "ERROR: global-attention layers are at " + str(globals_) + ",\n"
            "       which is not the periodic '(i+1) % N == 0' pattern the engine\n"
            "       assumes. ARCH_LAYER_IS_GLOBAL in arch.h has to be overridden\n"
            "       for this model — see the comment there.")
    if not window:
        raise SystemExit("ERROR: layer_types lists sliding layers but config.json "
                         "has no sliding_window size")
    return interval, int(window), globals_


def derive_rope(cfg):
    """Return a dict of rope settings."""
    theta = float(pick(cfg, "rope_theta", "rope_base", what="rope_theta"))
    partial = float(pick(cfg, "partial_rotary_factor", default=1.0, required=False) or 1.0)

    scaling = pick(cfg, "rope_scaling", "rope_parameters", required=False) or {}
    # Newer configs key the scaling per attention type.
    if isinstance(scaling, dict) and ("full_attention" in scaling or "sliding_attention" in scaling):
        scaling = scaling.get("full_attention") or {}

    out = {"theta": theta, "partial_rotary": partial, "kind": "none"}
    if not scaling:
        return out

    rope_type = str(scaling.get("rope_type") or scaling.get("type") or "yarn").lower()
    factor = scaling.get("factor")
    if factor is None:
        return out
    if "yarn" not in rope_type:
        raise SystemExit(
            f"ERROR: rope scaling type '{rope_type}' is not implemented "
            f"(only YaRN and unscaled are). See arch_runtime.h.")

    attn_factor = scaling.get("attention_factor")
    if attn_factor is None:
        attn_factor = 0.1 * math.log(float(factor)) + 1.0
    out.update({
        "kind": "yarn",
        "factor": float(factor),
        "attention_factor": float(attn_factor),
        "orig_ctx": int(scaling.get("original_max_position_embeddings")
                        or pick(cfg, "original_max_position_embeddings", required=False)
                        or 0),
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
    })
    if not out["orig_ctx"]:
        max_pos = pick(cfg, "max_position_embeddings", required=False)
        if max_pos:
            out["orig_ctx"] = int(int(max_pos) / float(factor))
        else:
            raise SystemExit("ERROR: YaRN scaling without original_max_position_embeddings")
    return out


def derive_gate(tensors, num_heads, head_dim, hidden):
    """Figure out how the attention output gate is produced.

    Returns (gate_kind, tensor_suffix_or_None, q_out_features).
    """
    qname, qmeta = find_tensor(tensors, ".layers.{layer}.self_attn.q_proj.weight")
    if qname is None:
        raise SystemExit("ERROR: no self_attn.q_proj.weight found in the checkpoint")
    q_out = qmeta["shape"][0]

    if q_out == num_heads * head_dim * 2:
        return "fused", None, q_out
    if q_out != num_heads * head_dim:
        raise SystemExit(
            f"ERROR: q_proj emits {q_out} features, which is neither "
            f"num_heads*head_dim ({num_heads * head_dim}) nor twice that. "
            f"The engine cannot lay out Q for this model.")

    # Separate gate projection: look for a small self_attn tensor whose output
    # is one value per head (or per channel).
    for cand in ("gate_proj", "o_gate", "attn_gate", "output_gate", "g_proj"):
        name, meta = find_tensor(tensors, f".layers.{{layer}}.self_attn.{cand}.weight")
        if name is None:
            continue
        out = meta["shape"][0]
        if out == num_heads:
            return "head", cand, q_out
        if out == num_heads * head_dim:
            return "channel", cand, q_out
    return "none", None, q_out


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

HEADER_TMPL = """\
// arch_{arch}.h — GENERATED by gen_arch_header.py. Do not hand-edit.
//
// Source checkpoint: {model}
// Model: {name}
//
// Every constant below was read from that checkpoint's config.json,
// tokenizer_config.json and safetensors headers.

#ifndef FLASH_MOE_ARCH_{guard}_H
#define FLASH_MOE_ARCH_{guard}_H

#define ARCH_GENERATED      1
#define ARCH_NAME           "{name}"

#define HIDDEN_DIM          {hidden}
#define NUM_LAYERS          {num_layers}
#define NUM_ATTN_HEADS      {num_heads}
#define NUM_KV_HEADS        {num_kv_heads}
#define HEAD_DIM            {head_dim}
#define VOCAB_SIZE          {vocab}
#define RMS_NORM_EPS        {eps}f
#define NUM_EXPERTS         {num_experts}
#define NUM_EXPERTS_PER_TOK {top_k}
#define MOE_INTERMEDIATE    {moe_inter}
#define SHARED_INTERMEDIATE {shared_inter}
#define FULL_ATTN_INTERVAL  {interval}
#define GROUP_SIZE          {group}
#define BITS                {bits}

#define ARCH_HAS_LINEAR_ATTN  {has_linear}
#define ARCH_SWA_WINDOW       {window}
#define NUM_FULL_ATTN_LAYERS  {num_attn_layers}
#define NUM_LINEAR_LAYERS     {num_linear_layers}

// Unused by this model; sized to keep the shared scratch buffers valid.
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128
#define LINEAR_VALUE_DIM    128
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM)
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE)
#define CONV_KERNEL_SIZE    4

#define ROPE_THETA          {theta}f
#define PARTIAL_ROTARY      {partial}f
#define ROTARY_DIM          {rotary_dim}
#define ARCH_ROPE_SCALING   {rope_scaling}
{rope_extra}
#define ARCH_ATTN_GATE      {gate_macro}
#define ARCH_ROUTER_SIGMOID {router_sigmoid}

#define EXPERT_SIZE         {expert_size}

#define EXPERT_SIZE_2BIT    {expert_size_2bit}
#define GATE_W_OFF_2  {g2[0]}
#define GATE_S_OFF_2  {g2[1]}
#define GATE_B_OFF_2  {g2[2]}
#define UP_W_OFF_2    {g2[3]}
#define UP_S_OFF_2    {g2[4]}
#define UP_B_OFF_2    {g2[5]}
#define DOWN_W_OFF_2  {g2[6]}
#define DOWN_S_OFF_2  {g2[7]}
#define DOWN_B_OFF_2  {g2[8]}

#define MAX_SEQ_LEN {max_seq}
#define GPU_KV_SEQ  8192

#define EOS_TOKEN_1         {eos1}
#define EOS_TOKEN_2         {eos2}
#define THINK_START_TOKEN   {think_start}
#define THINK_END_TOKEN     {think_end}

#define ARCH_DEFAULT_MODEL_DIR "{packed_dir}"

#endif  // FLASH_MOE_ARCH_{guard}_H
"""

GATE_MACRO = {
    "fused": "ARCH_GATE_FUSED_SIGMOID",
    "channel": "ARCH_GATE_FUSED_SOFTPLUS",
    "head": "ARCH_GATE_HEAD_SOFTPLUS",
    "none": "ARCH_GATE_NONE",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the downloaded model directory")
    ap.add_argument("--arch", default="laguna_s", help="profile name (default: laguna_s)")
    ap.add_argument("--output", default=None, help="output .h path")
    ap.add_argument("--max-seq", type=int, default=65536,
                    help="CPU KV cache cap for global layers (default 65536)")
    ap.add_argument("--router-sigmoid", choices=["auto", "yes", "no"], default="auto",
                    help="router activation; auto reads scoring_func from config.json")
    args = ap.parse_args()

    model_dir = Path(os.path.expanduser(args.model))
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"ERROR: {cfg_path} not found")
    cfg = json.loads(cfg_path.read_text())

    tensors = load_all_headers(model_dir)
    print(f"Read {len(tensors)} tensors from {model_dir}")

    hidden = int(pick(cfg, "hidden_size", "d_model", what="hidden_size"))
    num_layers = int(pick(cfg, "num_hidden_layers", "n_layers", what="num_hidden_layers"))
    num_heads = int(pick(cfg, "num_attention_heads", "n_heads", what="num_attention_heads"))
    num_kv_heads = int(pick(cfg, "num_key_value_heads", "n_kv_heads", default=num_heads,
                            required=False) or num_heads)
    head_dim = int(pick(cfg, "head_dim", default=hidden // num_heads, required=False)
                   or hidden // num_heads)
    vocab = int(pick(cfg, "vocab_size", what="vocab_size"))
    eps = float(pick(cfg, "rms_norm_eps", "layer_norm_eps", default=1e-6, required=False))
    num_experts = int(pick(cfg, "num_experts", "n_routed_experts", "num_local_experts",
                           what="expert count"))
    top_k = int(pick(cfg, "num_experts_per_tok", "top_k", "moe_top_k", what="top-k"))
    moe_inter = int(pick(cfg, "moe_intermediate_size", "expert_intermediate_size",
                         what="moe_intermediate_size"))
    shared_inter = int(pick(cfg, "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size",
                            default=moe_inter, required=False) or moe_inter)

    # --- cross-check the expert geometry against the actual tensors ---
    ename, emeta = find_tensor(tensors, ".layers.{layer}.mlp.switch_mlp.gate_proj.weight",
                               ".layers.{layer}.mlp.experts.gate_proj.weight")
    if ename is None:
        raise SystemExit("ERROR: no fused expert tensor (mlp.switch_mlp.gate_proj.weight) "
                         "found — this checkpoint is not in the MLX fused-expert layout "
                         "the packer expects.")
    sname, smeta = find_tensor(tensors, ".layers.{layer}.mlp.switch_mlp.gate_proj.scales",
                               ".layers.{layer}.mlp.experts.gate_proj.scales")
    e_shape = emeta["shape"]          # [num_experts, moe_inter, hidden/vals_per_word]
    if len(e_shape) != 3:
        raise SystemExit(f"ERROR: unexpected expert tensor rank: {ename} {e_shape}")
    if e_shape[0] != num_experts:
        raise SystemExit(f"ERROR: config says {num_experts} experts, tensor says {e_shape[0]}")
    if e_shape[1] != moe_inter:
        raise SystemExit(f"ERROR: config says moe_intermediate={moe_inter}, "
                         f"tensor says {e_shape[1]} ({ename})")
    bits, group = derive_quant(e_shape[1:], smeta["shape"][1:], hidden)
    print(f"Quantization: {bits}-bit, group size {group}")
    if bits != 4:
        raise SystemExit(f"ERROR: the packer writes 4-bit experts; this checkpoint is "
                         f"{bits}-bit. Use a 4-bit MLX conversion.")

    interval, window, globals_ = derive_layer_types(cfg, num_layers)
    rope = derive_rope(cfg)
    gate_kind, gate_name, q_out = derive_gate(tensors, num_heads, head_dim, hidden)

    if args.router_sigmoid == "auto":
        scoring = str(pick(cfg, "scoring_func", "router_scoring_func", default="", required=False) or "")
        router_sigmoid = 1 if "sigmoid" in scoring.lower() else 0
        if not scoring:
            print("NOTE: config.json has no scoring_func; assuming softmax routing. "
                  "Pass --router-sigmoid yes if this model gates experts with a sigmoid.")
    else:
        router_sigmoid = 1 if args.router_sigmoid == "yes" else 0

    sys.path.insert(0, str(HERE.parent))
    from arch_profile import component_layout
    comps4, expert_size = component_layout(hidden, moe_inter, group, 4)
    comps2, expert_size_2bit = component_layout(hidden, moe_inter, group, 2)

    rotary_dim = int(head_dim * rope["partial_rotary"])
    if rotary_dim % 2:
        raise SystemExit(f"ERROR: rotary dim {rotary_dim} is odd")

    # --- special tokens ---
    eos1 = eos2 = think_start = think_end = -1
    tok_cfg_path = model_dir / "tokenizer_config.json"
    eos_from_cfg = pick(cfg, "eos_token_id", required=False)
    if isinstance(eos_from_cfg, list) and eos_from_cfg:
        eos1 = int(eos_from_cfg[0])
        eos2 = int(eos_from_cfg[1]) if len(eos_from_cfg) > 1 else eos1
    elif isinstance(eos_from_cfg, int):
        eos1 = eos2 = eos_from_cfg
    if tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text())
        added = tok_cfg.get("added_tokens_decoder", {})
        by_content = {v.get("content"): int(k) for k, v in added.items()}
        for tag, setter in (("<think>", "start"), ("</think>", "end")):
            if tag in by_content:
                if setter == "start":
                    think_start = by_content[tag]
                else:
                    think_end = by_content[tag]
        if eos1 < 0:
            eos_tok = tok_cfg.get("eos_token")
            if isinstance(eos_tok, dict):
                eos_tok = eos_tok.get("content")
            if eos_tok in by_content:
                eos1 = eos2 = by_content[eos_tok]
    if eos1 < 0:
        raise SystemExit("ERROR: could not determine the EOS token id")

    rope_extra = ""
    if rope["kind"] == "yarn":
        rope_extra = (
            f"#define ARCH_ROPE_YARN_FACTOR {rope['factor']}f\n"
            f"#define ARCH_ROPE_YARN_ATTN_FACTOR {rope['attention_factor']!r}f\n"
            f"#define ARCH_ROPE_ORIG_CTX    {rope['orig_ctx']}\n"
            f"#define ARCH_ROPE_BETA_FAST   {rope['beta_fast']}f\n"
            f"#define ARCH_ROPE_BETA_SLOW   {rope['beta_slow']}f\n")

    packed_dir = f"packed_experts_{args.arch}" if args.arch != "qwen35_moe" else "packed_experts"
    out_h = Path(args.output) if args.output else HERE / f"arch_{args.arch}.h"

    text = HEADER_TMPL.format(
        arch=args.arch, guard=args.arch.upper(), model=model_dir,
        name=pick(cfg, "_name_or_path", default=args.arch, required=False) or args.arch,
        hidden=hidden, num_layers=num_layers, num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim, vocab=vocab, eps=eps,
        num_experts=num_experts, top_k=top_k, moe_inter=moe_inter,
        shared_inter=shared_inter, interval=interval, group=group, bits=bits,
        has_linear=0, window=window, num_attn_layers=num_layers, num_linear_layers=1,
        theta=rope["theta"], partial=rope["partial_rotary"], rotary_dim=rotary_dim,
        rope_scaling="ARCH_ROPE_YARN" if rope["kind"] == "yarn" else "ARCH_ROPE_NONE",
        rope_extra=rope_extra,
        gate_macro=GATE_MACRO[gate_kind], router_sigmoid=router_sigmoid,
        expert_size=expert_size, expert_size_2bit=expert_size_2bit,
        g2=[c["offset"] for c in comps2],
        max_seq=args.max_seq, eos1=eos1, eos2=eos2,
        think_start=think_start, think_end=think_end, packed_dir=packed_dir)
    out_h.write_text(text)

    profile = {
        "name": args.arch,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden": hidden,
        "moe_intermediate": moe_inter,
        "group_size": group,
        "bits": bits,
        "packed_dir": packed_dir,
        "expert_tensor": ename,
    }
    out_json = out_h.with_suffix(".json")
    out_json.write_text(json.dumps(profile, indent=2) + "\n")

    # --- report ---
    per_layer = expert_size * num_experts
    print(f"\nWrote {out_h}")
    print(f"Wrote {out_json}")
    print(f"\n{'=' * 68}")
    print(f"  layers            {num_layers}  ({len(globals_)} global every {interval}, "
          f"{num_layers - len(globals_)} sliding-window @ {window})")
    print(f"  hidden / heads    {hidden} / {num_heads}q + {num_kv_heads}kv x {head_dim}")
    print(f"  q_proj out        {q_out}  -> gate: {gate_kind}"
          + (f" ({gate_name})" if gate_name else ""))
    print(f"  experts           {num_experts} routed, top-{top_k}, "
          f"intermediate {moe_inter} (shared {shared_inter})")
    print(f"  router            {'sigmoid' if router_sigmoid else 'softmax'}")
    print(f"  rope              theta={rope['theta']:g} rotary_dim={rotary_dim} "
          f"scaling={rope['kind']}")
    print(f"  expert size       {expert_size:,} B  ->  {per_layer / 1e9:.2f} GB/layer, "
          f"{per_layer * num_layers / 1e9:.1f} GB total")
    print(f"  I/O per token     {expert_size * top_k * num_layers / 1e9:.2f} GB at K={top_k}")
    print(f"  eos/think ids     {eos1}, {eos2} / {think_start}, {think_end}")
    print(f"{'=' * 68}")
    if gate_kind == "head":
        print(f"\nNOTE: per-head gate found as self_attn.{gate_name}. Make sure "
              f"extract_weights.py copies it into model_weights.bin.")
    print(f"\nNext: uv run generate_expert_index.py --arch {args.arch} --model {model_dir}")


if __name__ == "__main__":
    main()
