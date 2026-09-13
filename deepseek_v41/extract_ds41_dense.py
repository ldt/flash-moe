#!/usr/bin/env python3
"""
extract_ds41_dense.py — Pack every *resident* tensor of DeepSeek-V4.1-Flash into
one mmap-able blob (ds41_dense.bin) plus a JSON manifest (ds41_dense.json), in
the same spirit as metal_infer/extract_weights.py for Qwen3.5.

Resident = everything the engine needs in unified memory for text decode:
  - embed.weight, head.weight, norm.weight                 (BF16)
  - layers.*.attn.* (wq_a, wq_b, wkv, wo_a, wo_b + scales, norms, attn_sink,
    compressor.*, indexer.*)                              (FP8 blk32 / BF16 / F32)
  - layers.*.ffn.gate.{weight,bias,bias_vl}, ffn.shared_experts.*  (BF16 / F32 / FP8)
  - layers.*.hc_* and *_norm.weight                       (F32 / BF16)
  - layers.{1,14}.engram.{wkv.weight, wkv.scale, q_weight, k_weight}

Excluded (streamed from the shards or unused): routed experts, the two Engram
embedding tables, mtp.* (DSpark), vision.*/aligner.*/image_*.

Tensors are stored in their native safetensors dtype, 64-byte aligned, and the
manifest uses the same {offset, size, shape, dtype} schema as model_weights.json
so the C manifest loader (load_manifest / find_tensor) can be reused as is.

Usage:
    uv run deepseek_v41/extract_ds41_dense.py --model <snapshot dir> --output metal_infer/
    uv run deepseek_v41/extract_ds41_dense.py --headers ds41_headers/ --dry-run   # plan only, no weights needed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds41_common import classify, default_hf_dir, fmt_bytes, iter_tensors, load_headers, tensor_nbytes  # noqa: E402

ALIGN = 64


def plan(headers):
    """Return (entries, total_size). entries: list of dicts sorted by name."""
    entries = []
    for name, shard, info, data_start in iter_tensors(headers):
        if classify(name) != "dense":
            continue
        entries.append({
            "name": name, "shard": shard,
            "src_offset": data_start + info["data_offsets"][0],
            "size": tensor_nbytes(info),
            "shape": info["shape"], "dtype": info["dtype"],
        })
    entries.sort(key=lambda e: e["name"])
    off = 0
    for e in entries:
        e["offset"] = off
        off = (off + e["size"] + ALIGN - 1) // ALIGN * ALIGN
    return entries, off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="local snapshot dir with the 48 shards")
    ap.add_argument("--headers", default=None, help="header dir (fetch_ds41_headers.py); implies --dry-run unless --model given")
    ap.add_argument("--output", default=".", help="output dir for ds41_dense.bin / ds41_dense.json")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and write only the manifest")
    args = ap.parse_args()

    model_dir = args.model or default_hf_dir()
    if args.headers:
        headers = load_headers(headers_dir=args.headers)
        if not args.model:
            args.dry_run = True
    elif model_dir:
        headers = load_headers(model_dir=model_dir)
    else:
        ap.error("need --model or --headers")

    entries, total = plan(headers)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_dtype = {}
    for e in entries:
        by_dtype[e["dtype"]] = by_dtype.get(e["dtype"], 0) + e["size"]
    print(f"[ds41-dense] {len(entries)} resident tensors, {fmt_bytes(total)} total")
    for dt, n in sorted(by_dtype.items(), key=lambda x: -x[1]):
        print(f"    {dt:8s} {fmt_bytes(n)}")

    manifest = {
        "model": "DeepSeek-V4.1-Flash",
        "source": model_dir,
        "total_size": total,
        "alignment": ALIGN,
        "dtypes": "F8_E4M3 weights carry a sibling .scale tensor of F8_E8M0 with one scale per 32x32 block "
                  "(shape [out/32, in/32]); BF16/F32 are stored raw little-endian",
        "tensors": {e["name"]: {"offset": e["offset"], "size": e["size"], "shape": e["shape"], "dtype": e["dtype"]}
                    for e in entries},
    }
    json_path = out_dir / "ds41_dense.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[ds41-dense] wrote manifest {json_path}")

    if args.dry_run:
        print("[ds41-dense] dry run: ds41_dense.bin not written")
        return

    bin_path = out_dir / "ds41_dense.bin"
    t0 = time.monotonic()
    fds = {}
    written = 0
    with open(bin_path, "wb") as out:
        out.truncate(total)
        # Read shard by shard, in source-offset order, for sequential I/O.
        for e in sorted(entries, key=lambda e: (e["shard"], e["src_offset"])):
            fd = fds.get(e["shard"])
            if fd is None:
                fd = fds[e["shard"]] = os.open(os.path.join(model_dir, e["shard"]), os.O_RDONLY)
            remaining, src, dst = e["size"], e["src_offset"], e["offset"]
            out.seek(dst)
            while remaining:
                chunk = os.pread(fd, min(remaining, 64 << 20), src)
                if not chunk:
                    raise IOError(f"short read on {e['shard']} for {e['name']}")
                out.write(chunk)
                src += len(chunk)
                remaining -= len(chunk)
                written += len(chunk)
    for fd in fds.values():
        os.close(fd)
    dt = time.monotonic() - t0
    print(f"[ds41-dense] wrote {fmt_bytes(written)} to {bin_path} in {dt:.1f} s ({written / dt / 1e9:.2f} GB/s)")


if __name__ == "__main__":
    main()
