#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "tokenizers>=0.20"]
# ///
"""
export_ds41_engram_meta.py — Export everything the C engine needs to address the
Engram n-gram tables of DeepSeek-V4.1-Flash: the compressed token map, the
per-(layer, n-gram order, head) prime moduli, the bucket offsets, and the hash
multipliers. This is a faithful port of inference/engram.py from the official
repo; the numbers must match bit-for-bit or every lookup lands on the wrong row.

How Engram addressing works (per token, per Engram layer):
  1. token id -> compressed id via a table built by normalizing each token's
     text (NFKC, strip accents, lowercase, collapse whitespace). 129,280 ids
     collapse to 99,092 classes ("The", " the", "THE" share a class).
  2. take the compressed ids of the current token and the 3 before it
     (pad_id when the sequence is shorter), multiply each by an odd int64
     multiplier drawn from numpy PCG64 seeded with 10007 * layer_id.
  3. XOR the products cumulatively: after i steps the running value is the hash
     of the (i+1)-gram. For n-gram orders 2, 3, 4 and each of the 8 heads, the
     row index is (hash mod prime[layer][order][head]) + offset[layer][order][head].
  4. 24 rows of 256 FP8 (+8 E8M0 scales) are fetched from the table, dequantized,
     flattened to 6144 floats and projected by engram.wkv.

Output (little-endian, "DS41ENGR" magic):
  u32 version=1, u32 vocab_size, u32 compressed_vocab_size, u32 n_layers(2),
  u32 max_ngram(4), u32 n_heads(8), u32 pad_compressed_id, u32 reserved
  i32 layer_ids[n_layers]
  i64 multipliers[n_layers][max_ngram]
  i64 primes[n_layers][max_ngram-1][n_heads]
  i64 offsets[n_layers][(max_ngram-1)*n_heads]
  i32 token_map[vocab_size]

Usage:
    uv run deepseek_v41/export_ds41_engram_meta.py --tokenizer <snapshot dir or tokenizer.json> --output metal_infer/ds41_engram_meta.bin
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds41_common import load_config  # noqa: E402

MAGIC = b"DS41ENGR"


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for n < 3.3e24 (covers our ~16M range many times over)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


def build_layout_primes(layer_ids, max_ngram, n_heads, vocab_size):
    """primes[layer][order][head], drawn in order and never reused (mirrors EngramLayout.from_args)."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def compute_hash_multipliers(layer_ids, max_ngram, compressed_vocab_size):
    """Mirrors engram.compute_hash_multipliers: numpy PCG64(10007*layer) odd int64s."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        vals = gen.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64)
        rows.append([int(v) * 2 + 1 for v in vals])
    return rows


def build_compressed_token_map(tokenizer_json_path):
    """Mirrors engram.build_compressed_token_map using the raw `tokenizers` backend."""
    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(str(tokenizer_json_path))
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    key_to_new = {}
    lookup = [0] * vocab_size
    for token_id in range(vocab_size):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def hash_positions(token_ids, layer_idx, meta):
    """Pure-Python reference of NgramHashState.forward for one layer (no image masks).
    Returns list over positions of the 24 row indices. Used by the C unit test."""
    max_ngram, n_heads = meta["max_ngram"], meta["n_heads"]
    mult = meta["multipliers"][layer_idx]
    primes = meta["primes"][layer_idx]
    offsets = meta["offsets"][layer_idx]
    pad = meta["pad_compressed_id"]
    comp = [meta["token_map"][t] for t in token_ids]
    out = []
    for pos in range(len(comp)):
        toks = [comp[pos - s] if pos - s >= 0 else pad for s in range(max_ngram)]
        rolling = toks[0] * mult[0]
        rows = []
        for i in range(1, max_ngram):
            rolling ^= toks[i] * mult[i]
            for h in range(n_heads):
                rows.append(rolling % primes[i - 1][h] + offsets[(i - 1) * n_heads + h])
        out.append(rows)
    return out


def build_meta(cfg, tokenizer_json):
    layer_ids = list(cfg["engram_layer_ids"])
    max_ngram, n_heads = cfg["engram_max_ngram_size"], cfg["engram_n_heads"]
    token_map, comp_vocab = build_compressed_token_map(tokenizer_json)
    if comp_vocab != cfg["engram_compressed_vocab_size"]:
        raise RuntimeError(f"compressed vocab {comp_vocab} != config {cfg['engram_compressed_vocab_size']}; "
                           "a different tokenizer.json or tokenizers version would rehash every table")
    primes = build_layout_primes(layer_ids, max_ngram, n_heads, cfg["engram_vocab_size"])
    flat = [[p for per in layer for p in per] for layer in primes]
    offsets = [list(np.cumsum([0, *sizes[:-1]]).tolist()) for sizes in flat]
    for li, sizes in enumerate(flat):
        expected = cfg["engram_num_embeddings"][li]
        if sum(sizes) != expected:
            raise RuntimeError(f"layer {layer_ids[li]}: table rows {sum(sizes)} != checkpoint {expected}")
    mult = compute_hash_multipliers(layer_ids, max_ngram, comp_vocab)
    return {
        "vocab_size": len(token_map), "compressed_vocab_size": comp_vocab,
        "layer_ids": layer_ids, "max_ngram": max_ngram, "n_heads": n_heads,
        "pad_compressed_id": token_map[cfg["engram_pad_id"]],
        "multipliers": mult, "primes": primes, "offsets": offsets, "token_map": token_map,
    }


def write_meta(meta, path):
    n_layers, max_ngram, n_heads = len(meta["layer_ids"]), meta["max_ngram"], meta["n_heads"]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8I", 1, meta["vocab_size"], meta["compressed_vocab_size"], n_layers,
                            max_ngram, n_heads, meta["pad_compressed_id"], 0))
        f.write(struct.pack(f"<{n_layers}i", *meta["layer_ids"]))
        for row in meta["multipliers"]:
            f.write(struct.pack(f"<{max_ngram}q", *row))
        for layer in meta["primes"]:
            for per in layer:
                f.write(struct.pack(f"<{n_heads}q", *per))
        for offs in meta["offsets"]:
            f.write(struct.pack(f"<{len(offs)}q", *offs))
        f.write(struct.pack(f"<{meta['vocab_size']}i", *meta["token_map"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json or a directory containing it")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default="ds41_engram_meta.bin")
    ap.add_argument("--json", default=None, help="also dump the metadata (minus token_map) as JSON")
    ap.add_argument("--selftest", nargs="*", type=int, metavar="TOKEN_ID",
                    help="print reference row indices for these token ids (layer index 0), for the C test")
    args = ap.parse_args()

    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    cfg = load_config(args.config)
    meta = build_meta(cfg, tok_path)
    write_meta(meta, args.output)

    print(f"[engram] compressed vocab {meta['compressed_vocab_size']} (config OK), pad -> {meta['pad_compressed_id']}")
    for li, lid in enumerate(meta["layer_ids"]):
        print(f"[engram] layer {lid}: multipliers {meta['multipliers'][li]}")
        print(f"[engram] layer {lid}: {sum(sum(p) for p in meta['primes'][li]):,} rows, "
              f"primes {meta['primes'][li][0][0]:,} .. {meta['primes'][li][-1][-1]:,}")
    print(f"[engram] wrote {args.output}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: v for k, v in meta.items() if k != "token_map"}, f, indent=1)
    if args.selftest is not None and args.selftest:
        for li in range(len(meta["layer_ids"])):
            rows = hash_positions(args.selftest, li, meta)
            for pos, r in enumerate(rows):
                print(f"L{li} pos{pos}", " ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
