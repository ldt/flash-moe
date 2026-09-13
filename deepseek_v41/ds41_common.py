#!/usr/bin/env python3
"""
ds41_common.py — Shared helpers for the DeepSeek-V4.1-Flash tooling.

Everything here works from safetensors *headers* only (the JSON block at the
start of each shard), so it runs before the 510 GB download: fetch the headers
with fetch_ds41_headers.py, or point at a local model directory.

Tensor naming in the official checkpoint (deepseek-ai/DeepSeek-V4.1-Flash):

  embed.weight, head.weight, norm.weight                    BF16 (resident)
  layers.{L}.attn.*, layers.{L}.ffn.gate.*, hc_*, *_norm    FP8 blk32 / BF16 / F32 (resident)
  layers.{L}.ffn.shared_experts.w{1,2,3}.{weight,scale}     FP8 blk32 (resident)
  layers.{L}.ffn.experts.{E}.w{1,2,3}.{weight,scale}        MXFP4 (streamed from SSD)
  layers.{1,14}.engram.embed.{weight,scale}                 FP8 rows (streamed from SSD)
  layers.{1,14}.engram.{wkv,q_weight,k_weight}              resident
  mtp.{0,1,2}.*                                             DSpark draft blocks (optional)
  vision.*, aligner.*, image_*                              vision (not needed for text)

Shard layout (48 shards): shard 1 = vision, 2 = embed, 3..42 = one backbone
layer each (layer L in shard L+3), 43 = head+norm, 44..46 = MTP stages,
47/48 = Engram tables for layers 1 and 14.
"""

import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

HF_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
NUM_SHARDS = 48

# Bytes per element for safetensors dtypes used by this checkpoint.
DTYPE_BYTES = {
    "F32": 4, "BF16": 2, "F16": 2,
    "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
}

EXPERT_RE = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
ENGRAM_TABLE_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
LAYER_RE = re.compile(r"^layers\.(\d+)\.")
MTP_RE = re.compile(r"^mtp\.(\d+)\.")
VISION_RE = re.compile(r"^(vision\.|aligner\.|image_)")

# Fixed component order inside one packed expert record (see docs/deepseek-v41-flash-plan.md).
EXPERT_COMPONENTS = ["w1.weight", "w3.weight", "w2.weight", "w1.scale", "w3.scale", "w2.scale"]


def load_config(path=None):
    """Load the official config.json (the inference/ flavour with flat keys)."""
    if path is None:
        path = Path(__file__).with_name("config.json")
    with open(path) as f:
        return json.load(f)


def shard_name(i):
    return f"model-{i:05d}-of-{NUM_SHARDS:05d}.safetensors"


def parse_safetensors_header(filepath):
    """Read the header of a local safetensors shard.
    Returns (header_dict, data_start) where data_start = 8 + header_len."""
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def load_headers(model_dir=None, headers_dir=None):
    """Return {shard_filename: (header_dict, data_start)} for all shards.

    model_dir:   directory with the real .safetensors shards (local download)
    headers_dir: directory with <shard>.header.json files written by
                 fetch_ds41_headers.py (header-only mode, no weights needed)
    """
    out = {}
    if headers_dir is not None:
        hdir = Path(headers_dir)
        for i in range(1, NUM_SHARDS + 1):
            name = shard_name(i)
            p = hdir / f"{name}.header.json"
            if not p.exists():
                continue
            raw = p.read_bytes()
            header = json.loads(raw)
            # data_start must match what the real file has: 8 + len(header bytes).
            # fetch_ds41_headers.py stores the header bytes verbatim, so len(raw) is exact.
            out[name] = (header, 8 + len(raw))
    elif model_dir is not None:
        mdir = Path(model_dir)
        for i in range(1, NUM_SHARDS + 1):
            name = shard_name(i)
            p = mdir / name
            if p.exists():
                out[name] = parse_safetensors_header(p)
    else:
        raise ValueError("need model_dir or headers_dir")
    if not out:
        raise FileNotFoundError("no shards / headers found")
    return out


def iter_tensors(headers):
    """Yield (name, shard, info, data_start) for every tensor in all headers."""
    for shard, (header, data_start) in headers.items():
        for name, info in header.items():
            if name == "__metadata__":
                continue
            yield name, shard, info, data_start


def tensor_nbytes(info):
    a, b = info["data_offsets"]
    return b - a


def classify(name):
    """Bucket a tensor name into a streaming/residency class."""
    if EXPERT_RE.match(name):
        return "mtp_experts" if name.startswith("mtp.") else "experts"
    if ENGRAM_TABLE_RE.match(name):
        return "engram_tables"
    if VISION_RE.match(name):
        return "vision"
    if MTP_RE.match(name):
        return "mtp_dense"
    return "dense"


def summarize_bytes(headers):
    """Total bytes per class and per layer for the dense part."""
    totals = defaultdict(int)
    per_layer_dense = defaultdict(int)
    for name, _shard, info, _ds in iter_tensors(headers):
        cls = classify(name)
        n = tensor_nbytes(info)
        totals[cls] += n
        if cls == "dense":
            m = LAYER_RE.match(name)
            per_layer_dense[int(m.group(1)) if m else -1] += n
    return dict(totals), dict(per_layer_dense)


def expert_records(headers, cfg):
    """Build the per-(layer, expert) record list.

    For each expert we locate the six tensors, check that the three weights are
    contiguous and the three scales are contiguous (true for the official
    shards: the exporter sorted tensors by dtype then name), and return absolute
    file offsets usable directly with pread().

    Returns dict: layer -> {"shard": str, "experts": {E: {...}}, ...}
    """
    n_layers = cfg["n_layers"]
    per_layer = defaultdict(lambda: defaultdict(dict))
    shard_of = {}
    for name, shard, info, data_start in iter_tensors(headers):
        m = EXPERT_RE.match(name)
        if not m or m.group(1) != "layers":
            continue
        layer, expert, w, kind = int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        shard_of.setdefault(layer, shard)
        if shard_of[layer] != shard:
            raise RuntimeError(f"layer {layer} experts span two shards ({shard_of[layer]}, {shard})")
        a, b = info["data_offsets"]
        per_layer[layer][expert][f"{w}.{kind}"] = {
            "abs_offset": data_start + a,
            "size": b - a,
            "shape": info["shape"],
            "dtype": info["dtype"],
        }

    result = {}
    for layer in range(n_layers):
        experts = per_layer.get(layer)
        if not experts:
            raise RuntimeError(f"layer {layer}: no experts found")
        if len(experts) != cfg["n_routed_experts"]:
            raise RuntimeError(f"layer {layer}: {len(experts)} experts, expected {cfg['n_routed_experts']}")
        recs = {}
        for e in range(cfg["n_routed_experts"]):
            comps = experts[e]
            if set(comps) != {"w1.weight", "w2.weight", "w3.weight", "w1.scale", "w2.scale", "w3.scale"}:
                raise RuntimeError(f"layer {layer} expert {e}: missing components {sorted(comps)}")
            w = sorted((comps[k]["abs_offset"], comps[k]["size"], k) for k in ("w1.weight", "w2.weight", "w3.weight"))
            s = sorted((comps[k]["abs_offset"], comps[k]["size"], k) for k in ("w1.scale", "w2.scale", "w3.scale"))
            for grp in (w, s):
                for (o0, s0, _), (o1, _, _) in zip(grp, grp[1:]):
                    if o0 + s0 != o1:
                        raise RuntimeError(f"layer {layer} expert {e}: non-contiguous {grp}")
            recs[e] = {
                "weights_offset": w[0][0],
                "weights_size": sum(x[1] for x in w),
                "weights_order": [x[2] for x in w],
                "scales_offset": s[0][0],
                "scales_size": sum(x[1] for x in s),
                "scales_order": [x[2] for x in s],
            }
        result[layer] = {"shard": shard_of[layer], "experts": recs}
    return result


def fmt_bytes(n):
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def default_hf_dir():
    """Best-effort guess of the local HF snapshot directory."""
    base = Path(os.path.expanduser("~/.cache/huggingface/hub")) / f"models--{HF_REPO.replace('/', '--')}" / "snapshots"
    if base.exists():
        snaps = sorted(base.iterdir())
        if snaps:
            return str(snaps[-1])
    return None
