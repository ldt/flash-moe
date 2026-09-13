#!/usr/bin/env python3
"""
generate_ds41_expert_index.py — Build ds41_expert_index.json for DeepSeek-V4.1-Flash.

Unlike the Qwen3.5 pipeline (generate_expert_index.py + repack_experts.py, which
needed a 218 GB repack because MLX stacks all experts of a layer per component),
the official DeepSeek checkpoint stores one tensor per expert, and the exporter
wrote them sorted by dtype then name. So for every expert the three MXFP4 weight
tensors (w1, w2, w3 — logical gate, down, up) are contiguous in the shard, and so
are the three E8M0 scale tensors. Two pread() calls per expert, straight from
the downloaded shards, no repack:

    weights: 3 x 5,898,240 B = 17,694,720 B   (w1, w2, w3 in that order)
    scales:  3 x   368,640 B =  1,105,920 B   (w1, w2, w3 in that order)

The index records, per layer, the shard file and per expert the absolute byte
offsets of both regions. Layer L lives in shard L+3 (model-00003 .. model-00042).

Works from headers only (--headers, see fetch_ds41_headers.py) or from a local
download (--model).

Usage:
    uv run deepseek_v41/generate_ds41_expert_index.py --headers ds41_headers/ --model ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/<rev>
    uv run deepseek_v41/generate_ds41_expert_index.py --model <dir>          # headers read from the shards
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds41_common import default_hf_dir, expert_records, fmt_bytes, load_config, load_headers  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="local snapshot dir (recorded in the index; shards read if --headers absent)")
    ap.add_argument("--headers", default=None, help="dir of *.header.json from fetch_ds41_headers.py")
    ap.add_argument("--config", default=None, help="config.json (default: deepseek_v41/config.json)")
    ap.add_argument("--output", default="ds41_expert_index.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = args.model or default_hf_dir()
    if args.headers:
        headers = load_headers(headers_dir=args.headers)
    elif model_dir:
        headers = load_headers(model_dir=model_dir)
    else:
        ap.error("need --headers or --model")

    recs = expert_records(headers, cfg)

    # Every expert must have identical sizes and component order — the C engine
    # relies on fixed offsets inside the two regions.
    sizes = {(r["weights_size"], r["scales_size"], tuple(r["weights_order"]), tuple(r["scales_order"]))
             for layer in recs.values() for r in layer["experts"].values()}
    if len(sizes) != 1:
        raise RuntimeError(f"inconsistent expert layouts: {sizes}")
    (wsize, ssize, worder, sorder), = sizes

    dim, inter = cfg["dim"], cfg["moe_inter_dim"]
    w13 = inter * dim // 2          # fp4 pairs
    w2 = dim * inter // 2
    s13 = inter * (dim // 32)
    s2 = dim * (inter // 32)
    assert wsize == w13 * 2 + w2 and ssize == s13 * 2 + s2, (wsize, ssize)

    index = {
        "model": "DeepSeek-V4.1-Flash",
        "model_path": model_dir,
        "format": "mxfp4 e2m1 nibbles (low nibble = even column), 32-column blocks, e8m0 scales",
        "n_layers": cfg["n_layers"],
        "n_routed_experts": cfg["n_routed_experts"],
        "expert_layout": {
            "weights_size": wsize,
            "weights_order": list(worder),
            "weights_component_offsets": {"w1": 0, "w2": w13, "w3": w13 + w2},
            "weights_component_shapes": {"w1": [inter, dim // 2], "w2": [dim, inter // 2], "w3": [inter, dim // 2]},
            "scales_size": ssize,
            "scales_order": list(sorder),
            "scales_component_offsets": {"w1": 0, "w2": s13, "w3": s13 + s2},
            "scales_component_shapes": {"w1": [inter, dim // 32], "w2": [dim, inter // 32], "w3": [inter, dim // 32]},
            "note": "w1=gate_proj, w3=up_proj, w2=down_proj (SwiGLU: silu(min(w1x,10)) * clamp(w3x,-10,10) -> w2)",
        },
        "layers": {
            str(layer): {
                "shard": info["shard"],
                "experts": [
                    [info["experts"][e]["weights_offset"], info["experts"][e]["scales_offset"]]
                    for e in range(cfg["n_routed_experts"])
                ],
            }
            for layer, info in recs.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(index, f, indent=1)

    n_exp = cfg["n_layers"] * cfg["n_routed_experts"]
    print(f"[ds41-index] {cfg['n_layers']} layers x {cfg['n_routed_experts']} experts = {n_exp} experts")
    print(f"[ds41-index] per expert: weights {fmt_bytes(wsize)} + scales {fmt_bytes(ssize)} = {fmt_bytes(wsize + ssize)}")
    print(f"[ds41-index] total routed experts on disk: {fmt_bytes(n_exp * (wsize + ssize))}")
    print(f"[ds41-index] wrote {args.output}")


if __name__ == "__main__":
    main()
