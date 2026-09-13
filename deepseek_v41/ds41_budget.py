#!/usr/bin/env python3
"""
ds41_budget.py — Disk / RAM / per-token I/O budget for running DeepSeek-V4.1-Flash
with the flash-moe SSD-streaming approach on an Apple Silicon Mac.

All byte counts come from the real safetensors headers (fetch_ds41_headers.py),
not from parameter-count folklore. Throughput numbers are *estimates* driven by
the --ssd-gbps and --hit-rate knobs; the defaults are calibrated on the Qwen3.5
runs documented in results.tsv (M3 Max, 17.5 GB/s sequential SSD, ~71 % page
cache hit rate with 35 GB of cache over 209 GB of experts).

Usage:
    uv run deepseek_v41/ds41_budget.py --headers ds41_headers/ --ram 36
    uv run deepseek_v41/ds41_budget.py --headers ds41_headers/ --ram 36 --k 4 --dense-bits 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds41_common import (  # noqa: E402
    ENGRAM_TABLE_RE, default_hf_dir, expert_records, fmt_bytes, iter_tensors,
    load_config, load_headers, summarize_bytes, tensor_nbytes,
)


def gb(n):
    return n / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headers", default=None, help="dir of *.header.json (fetch_ds41_headers.py)")
    ap.add_argument("--model", default=None, help="local snapshot dir (alternative to --headers)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ram", type=float, default=36, help="unified memory in GB (default 36)")
    ap.add_argument("--os-reserve", type=float, default=5.0, help="GB kept for macOS + apps (default 5)")
    ap.add_argument("--k", type=int, default=None, help="routed experts per token (default: config, 6)")
    ap.add_argument("--dense-bits", type=int, default=8, choices=(8, 4),
                    help="resident dense weights kept as stored (8) or requantized to 4-bit (4)")
    ap.add_argument("--expert-bits", type=int, default=4, choices=(4, 2),
                    help="experts as stored (MXFP4 = 4) or requantized to 2-bit (2, quality loss)")
    ap.add_argument("--ssd-gbps", type=float, default=8.0,
                    help="effective SSD throughput for ~18 MB random reads (GB/s, default 8)")
    ap.add_argument("--hit-rate", type=float, default=None,
                    help="page-cache hit rate; default: scaled from the Qwen run by cache/expert ratio")
    ap.add_argument("--gpu-ms-per-layer", type=float, default=2.2,
                    help="GPU+CPU time per layer excluding expert I/O (ms). Qwen: 1.8 ms; V4.1 has 4x wider residual + sparse attn")
    ap.add_argument("--prompt-tokens", type=int, default=2000, help="for the prefill estimate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.headers:
        headers = load_headers(headers_dir=args.headers)
    else:
        model_dir = args.model or default_hf_dir()
        if not model_dir:
            ap.error("need --headers or --model")
        headers = load_headers(model_dir=model_dir)

    totals, per_layer_dense = summarize_bytes(headers)
    recs = expert_records(headers, cfg)
    any_rec = next(iter(recs[0]["experts"].values()))
    expert_bytes = any_rec["weights_size"] + any_rec["scales_size"]
    if args.expert_bits == 2:
        # 2-bit nibbles halve the weight bytes; keep the same scale granularity.
        expert_bytes = any_rec["weights_size"] // 2 + any_rec["scales_size"]

    L = cfg["n_layers"]
    K = args.k or cfg["n_activated_experts"]
    E = cfg["n_routed_experts"]

    engram_rows = {}
    for name, _s, info, _d in iter_tensors(headers):
        m = ENGRAM_TABLE_RE.match(name)
        if m:
            engram_rows.setdefault(int(m.group(1)), {})[m.group(2)] = (info["shape"], tensor_nbytes(info))

    print("=" * 78)
    print("DeepSeek-V4.1-Flash on-disk footprint (from safetensors headers)")
    print("=" * 78)
    total_disk = sum(totals.values())
    for cls, label in (("experts", "routed experts, MXFP4 (streamed)"),
                       ("engram_tables", "Engram n-gram tables, FP8 (streamed, ~13 KB/token)"),
                       ("dense", "backbone dense: attention/shared/gate/hc/embed/head"),
                       ("mtp_experts", "DSpark draft experts (optional, speculative decoding)"),
                       ("mtp_dense", "DSpark draft dense (optional)"),
                       ("vision", "vision encoder + aligner (not needed for text)")):
        print(f"  {fmt_bytes(totals.get(cls, 0)):>10}  {label}")
    print(f"  {fmt_bytes(total_disk):>10}  TOTAL download (48 shards)")
    text_min = totals["experts"] + totals["engram_tables"] + totals["dense"]
    print(f"  {fmt_bytes(text_min):>10}  minimum kept for text inference (delete vision + mtp shards)")
    print(f"  {fmt_bytes(totals['experts'] + totals['dense']):>10}  ... without Engram (model would run degraded: Engram is trained in)")
    print()
    print(f"  per routed expert: {fmt_bytes(expert_bytes)}  "
          f"({L} layers x {E} experts, {K} active per token + 1 shared)")
    for layer, rows in sorted(engram_rows.items()):
        (n, d), wb = rows["weight"]
        print(f"  engram layer {layer:2d}: {n:,} rows x {d} fp8 + {rows['scale'][0][1]} e8m0 scales/row  "
              f"-> {fmt_bytes(wb + rows['scale'][1])}")

    print()
    print("=" * 78)
    print(f"Unified memory plan for {args.ram:g} GB (dense weights {args.dense_bits}-bit)")
    print("=" * 78)
    dense_resident = totals["dense"]
    per_layer = sum(v for k, v in per_layer_dense.items() if k >= 0) / L
    if args.dense_bits == 4:
        # BF16 embed/head (2.65 GB) -> 4-bit (x0.25); FP8 projections -> 4-bit (x0.5).
        # We keep norms / hc / gate as-is (small).
        bf16 = sum(tensor_nbytes(i) for n, _s, i, _d in iter_tensors(headers)
                   if n in ("embed.weight", "head.weight"))
        dense_resident = (totals["dense"] - bf16) * 0.5 + bf16 * 0.25
    scratch = 0.4e9  # Metal scratch, KV ring buffers, indexer caches for 128K ctx
    page_cache = args.ram * 1e9 - args.os_reserve * 1e9 - dense_resident - scratch
    print(f"  {gb(dense_resident):6.2f} GB  resident dense weights (mmap, read-only)  [{gb(per_layer * 1e0):.3f} GB/layer as stored]")
    print(f"  {gb(scratch):6.2f} GB  Metal scratch + KV caches (890 B/token main KV, 2.6 MB SWA rings)")
    print(f"  {args.os_reserve:6.2f} GB  macOS + apps reserve")
    print(f"  {gb(page_cache):6.2f} GB  left for the OS page cache (expert working set)")
    cache_cov = page_cache / (L * E * expert_bytes)
    print(f"             = {100 * cache_cov:.1f} % of the routed-expert bytes  (Qwen3.5 run: 35 GB / 209 GB = 16.7 %)")
    if page_cache < 8e9:
        print("  WARNING: under ~8 GB of page cache the hit rate collapses; use --dense-bits 4 or a bigger machine")

    print()
    print("=" * 78)
    print(f"Decode cost per token (K={K})")
    print("=" * 78)
    per_token = L * K * expert_bytes
    print(f"  expert bytes per token: {L} layers x {K} x {fmt_bytes(expert_bytes)} = {fmt_bytes(per_token)}")
    print(f"  (Qwen3.5-397B at K=4 in this repo: 60 x 4 x 7.08 MB = 1.70 GB/token -> 4.4 tok/s)")
    # Hit-rate model: the Qwen run got 71 % hits with 16.7 % coverage. Reuse is
    # driven by routing skew; assume hits scale with sqrt(coverage ratio), capped.
    if args.hit_rate is None:
        hit = min(0.71 * (cache_cov / 0.167) ** 0.5, 0.85)
    else:
        hit = args.hit_rate
    ssd_bytes = per_token * (1 - hit)
    io_ms = ssd_bytes / (args.ssd_gbps * 1e9) * 1e3
    compute_ms = L * args.gpu_ms_per_layer
    engram_ms = 2 * 1.0  # 2 layers x ~48 parallel 4K random reads; sub-ms each in practice
    total_ms = io_ms + compute_ms + engram_ms
    print(f"  assumed page-cache hit rate: {100 * hit:.0f} %  -> {fmt_bytes(ssd_bytes)} from SSD per token")
    print(f"  SSD time     @ {args.ssd_gbps:g} GB/s : {io_ms:7.1f} ms")
    print(f"  GPU/CPU time @ {args.gpu_ms_per_layer:g} ms/layer: {compute_ms:7.1f} ms  (serial with SSD on unified memory)")
    print(f"  Engram lookups             : {engram_ms:7.1f} ms")
    print(f"  ---------------------------------------")
    print(f"  ESTIMATE: {total_ms:.0f} ms/token  ->  {1000 / total_ms:.1f} tok/s")
    cold = per_token / (args.ssd_gbps * 1e9) * 1e3 + compute_ms + engram_ms
    print(f"  cold cache (0 % hits):  {cold:.0f} ms/token -> {1000 / cold:.1f} tok/s")

    print()
    print("=" * 78)
    print(f"Prefill estimate for a {args.prompt_tokens}-token prompt")
    print("=" * 78)
    # Batched prefill loads each touched expert once per layer. With N tokens
    # and K of E experts per token, the expected number of distinct experts is
    # E * (1 - (1 - K/E)^N).
    n = args.prompt_tokens
    distinct = E * (1 - (1 - K / E) ** n)
    enc_layers = cfg["n_layers"] // 2 + 1        # layers 0..20 run over the whole prompt
    dec_layers = cfg["n_layers"] - enc_layers    # layers 21..39: only the last window (CED bounded replay)
    win = cfg["window_size"]
    dec_distinct = E * (1 - (1 - K / E) ** min(n, win))
    bytes_full = L * distinct * expert_bytes
    bytes_ced = enc_layers * distinct * expert_bytes + dec_layers * dec_distinct * expert_bytes
    seq_gbps = 15.0  # large sequential-ish reads approach the 17.5 GB/s device limit
    print(f"  distinct experts touched per layer: {distinct:.0f} / {E}")
    print(f"  all 40 layers over the prompt : {fmt_bytes(bytes_full)}  ~{bytes_full / (seq_gbps * 1e9):.0f} s of SSD")
    print(f"  CED (encoder 0..20 full, decoder 21..39 last {win} tokens): {fmt_bytes(bytes_ced)}  ~{bytes_ced / (seq_gbps * 1e9):.0f} s of SSD")
    print("  (plus GPU matvec time, which for batched prefill is small next to the I/O)")


if __name__ == "__main__":
    main()
