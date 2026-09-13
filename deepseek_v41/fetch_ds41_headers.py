#!/usr/bin/env python3
"""
fetch_ds41_headers.py — Download only the safetensors *headers* of
DeepSeek-V4.1-Flash (about 11 MB total instead of 510 GB).

Each shard starts with an 8-byte little-endian header length followed by a
JSON header describing every tensor (dtype, shape, byte offsets). Two HTTP
range requests per shard are enough to get it. The headers are all the
budget / index tooling needs, so you can check whether the model fits your
machine before committing to the download.

Usage:
    uv run deepseek_v41/fetch_ds41_headers.py --out ds41_headers/
    uv run deepseek_v41/ds41_budget.py --headers ds41_headers/ --ram 36

Requires `curl` on PATH (uses the same proxy settings as your shell).
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds41_common import HF_REPO, NUM_SHARDS, shard_name  # noqa: E402


def curl_range(url, first, last, timeout):
    cmd = ["curl", "-sSL", "--fail", "--max-time", str(timeout), "-r", f"{first}-{last}", url]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url} [{first}-{last}]: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def fetch_one(i, out_dir, revision, timeout):
    name = shard_name(i)
    dst = out_dir / f"{name}.header.json"
    if dst.exists() and dst.stat().st_size > 0:
        return name, dst.stat().st_size, "cached"
    url = f"https://huggingface.co/{HF_REPO}/resolve/{revision}/{name}"
    head = curl_range(url, 0, 7, timeout)
    if len(head) != 8:
        raise RuntimeError(f"{name}: expected 8 bytes, got {len(head)}")
    (hlen,) = struct.unpack("<Q", head)
    raw = curl_range(url, 8, 8 + hlen - 1, timeout)
    if len(raw) != hlen:
        raise RuntimeError(f"{name}: header truncated ({len(raw)}/{hlen})")
    json.loads(raw)  # validate before writing
    # Stored verbatim: the tooling recomputes data_start as 8 + len(file).
    dst.write_bytes(raw)
    return name, hlen, "fetched"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ds41_headers", help="output directory")
    ap.add_argument("--revision", default="main", help="HF revision (branch/tag/commit)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180, help="per-request timeout (s)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(fetch_one, i, out_dir, args.revision, args.timeout) for i in range(1, NUM_SHARDS + 1)]
        for f in futs:
            name, n, how = f.result()
            total += n
            print(f"  {name}  {n:>8,} bytes  ({how})")
    print(f"\n{NUM_SHARDS} headers, {total / 1e6:.1f} MB total -> {out_dir}/")
    print("Next: uv run deepseek_v41/ds41_budget.py --headers", os.fspath(out_dir))


if __name__ == "__main__":
    main()
