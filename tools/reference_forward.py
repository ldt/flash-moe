#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx", "numpy"]
# ///
"""Reference forward pass for Laguna S, in MLX, for diffing against the engine.

The engine's own numbers come from FLASH_MOE_DEBUG_LAYERS=1, which prints the
RMS of the hidden state at a fixed point in each layer: after that layer's
attention and residual, before its FFN/MoE. This script reproduces exactly
that quantity so the two can be compared layer by layer.

It only evaluates position 0 of a single token, which makes attention trivial
(softmax over one key, so attn_out == v) and removes rope, the KV cache and
the sliding window from the comparison. That is deliberate: it isolates the
projections, gating, norms and the MoE. Once those agree, extend to position
1+ to bring rope and the window into scope.

Usage:
    uv run tools/reference_forward.py --model ~/models/laguna-s-mlx-4bit --layers 3
"""

import argparse
import json
import os

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--token", type=int, default=1172, help="token id at position 0")
    ap.add_argument("--layers", type=int, default=3, help="how many layers to walk")
    args = ap.parse_args()

    md = os.path.expanduser(args.model)
    weight_map = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(md, "config.json")))
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    n_kv = cfg["num_key_value_heads"]
    heads_per_layer = cfg["num_attention_heads_per_layer"]
    dense_layers = set(cfg.get("mlp_only_layers", []))
    top_k = cfg["num_experts_per_tok"]
    scaling = float(cfg.get("moe_routed_scaling_factor", 1.0))

    cache = {}

    def T(name):
        if name not in cache:
            cache[name] = mx.load(os.path.join(md, weight_map[name]))[name]
        return cache[name]

    def deq(base, bits=4):
        return mx.dequantize(T(base + ".weight"), T(base + ".scales"), T(base + ".biases"),
                             group_size=64, bits=bits).astype(mx.float32)

    def rn(x, w, eps=None):
        eps = cfg["rms_norm_eps"] if eps is None else eps
        return (x * mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)) * w.astype(mx.float32)

    def silu(z):
        return z * mx.sigmoid(z)

    def rms(z):
        return float(mx.sqrt(mx.mean(z ** 2)))

    h = deq("model.embed_tokens")[args.token]
    print(f"embedding                       rms={rms(h):.5f}")

    for layer in range(args.layers):
        p = f"model.layers.{layer}"
        n_heads = heads_per_layer[layer]

        # --- attention (position 0: attn_out == v, gated per head) ---
        xn = rn(h, T(f"{p}.input_layernorm.weight"))
        v = xn @ deq(f"{p}.self_attn.v_proj").T
        g = xn @ deq(f"{p}.self_attn.g_proj").T
        attn = mx.repeat(v.reshape(n_kv, head_dim), n_heads // n_kv, axis=0)
        attn = attn * mx.log1p(mx.exp(g))[:, None]          # per-head softplus gate
        h = h + (attn.reshape(-1) @ deq(f"{p}.self_attn.o_proj").T)
        kind = "global " if layer % 4 == 0 else "sliding"
        print(f"layer {layer:2d} {kind} heads={n_heads:2d}  after attn  rms={rms(h):.5f}"
              f"   <- compare with the engine's [dbg] line")

        # --- FFN ---
        hn = rn(h, T(f"{p}.post_attention_layernorm.weight"))
        if layer in dense_layers:
            ff = silu(hn @ deq(f"{p}.mlp.gate_proj").T) * (hn @ deq(f"{p}.mlp.up_proj").T)
            h = h + (ff @ deq(f"{p}.mlp.down_proj").T)
            print(f"           dense MLP              rms={rms(h):.5f}")
            continue

        router = mx.dequantize(T(f"{p}.mlp.gate.proj.weight"), T(f"{p}.mlp.gate.proj.scales"),
                               T(f"{p}.mlp.gate.proj.biases"), group_size=64, bits=8).astype(mx.float32)
        scores = mx.sigmoid(hn @ router.T)
        idx = mx.argsort(-scores)[:top_k]
        w = scores[idx]
        w = w / w.sum()

        parts = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            parts[proj] = [T(f"{p}.mlp.switch_mlp.{proj}.{s}") for s in ("weight", "scales", "biases")]
        moe = mx.zeros((hidden,))
        for j, e in enumerate([int(i) for i in idx]):
            def ex(proj):
                W, S, B = parts[proj]
                return mx.dequantize(W[e], S[e], B[e], group_size=64, bits=4).astype(mx.float32)
            moe = moe + float(w[j]) * ((silu(hn @ ex("gate_proj").T) * (hn @ ex("up_proj").T)) @ ex("down_proj").T)

        shared = (silu(hn @ deq(f"{p}.mlp.shared_expert.gate_proj").T)
                  * (hn @ deq(f"{p}.mlp.shared_expert.up_proj").T)) @ deq(f"{p}.mlp.shared_expert.down_proj").T

        print(f"           experts {sorted(int(i) for i in idx)}")
        print(f"           moe rms={rms(moe):.5f} (x{scaling}) shared rms={rms(shared):.5f} (ungated)")
        h = h + scaling * moe + shared
        print(f"           after MoE              rms={rms(h):.5f}")


if __name__ == "__main__":
    main()
