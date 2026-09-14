# Running DeepSeek-V4.1-Flash on a 36 GB M3 Max with Flash-MoE

Status: **groundwork on branch `claude/deepseek-v4-flash-m3-7bqkms`**. Everything in
`deepseek_v41/` and the `ds41_*` files in `metal_infer/` was validated on Linux against the
official checkpoint metadata and reference code. Nothing has run on a Mac yet; the Metal
kernels are drafted, not compiled. Sections 8 and 9 say exactly what to do first on the
laptop.

Sources used (all verbatim, none from memory):

- `deepseek-ai/DeepSeek-V4.1-Flash` on Hugging Face: `config.json`, `inference/model.py`,
  `inference/kernel.py`, `inference/engram.py`, `inference/convert.py`, `encoding/README.md`,
  `tokenizer.json`, `model.safetensors.index.json`, and the header of all 48 shards
  (fetched with HTTP range requests, ~11 MB total).
- `DeepSeek_V41_Tech_Report.pdf` (sections 2.1–2.4).
- The Latent Space "AINews" post the request pointed at, which is where the "SSD streaming"
  framing comes from.

## 1. Verdict in one paragraph

It is feasible with the Flash-MoE approach, but it is a **port of the whole engine, not a
config change**, and the laptop will be I/O-bound harder than with Qwen3.5. Per decode token
the model needs 40 layers × 6 experts × 18.8 MB = **4.5 GB of expert weights** (Qwen3.5 in
this repo at K=4: 1.7 GB). With ~21 GB of page cache on a 36 GB machine the honest projection
is **~1.8 tok/s at K=6, ~2.6 tok/s at K=4**, cold cache ~1.5–2 tok/s. A page-cache hit is not
free either: with today's `pread()` design it is a ~14 GB/s memcpy into the Metal buffer
(calibrated on the Qwen run), which is why more RAM alone barely helps until the design goes
zero-copy (section 4). Prefill of a 2 000-token
prompt costs ~18 s of SSD reads. Disk: 501 GB for text inference (289 GB experts + 203 GB
Engram tables + 10 GB dense), read directly from the downloaded shards, no repack. The
article's "optimized for SSD streaming" refers to DeepSeek's *datacenter* design (KV cache and
Engram tables offloaded to SSD/host RAM); the "8B active / 1–2 % sparsity" figures are real but
those 8–16 B active parameters still have to come off the SSD every token.

One caveat on the machine: `CLAUDE.md` describes a 48 GB M3 Max; the request says 36 GB.
Everything below is computed for 36 GB (`ds41_budget.py --ram 48` gives the other case:
about 33 GB of page cache and ~2 tok/s at K=6). Section 4 has the 64 GB / 128 GB M5 Max cases.

## 2. What DeepSeek-V4.1-Flash actually is (from `inference/model.py`)

| | Qwen3.5-397B-A17B (current engine) | DeepSeek-V4.1-Flash |
|---|---|---|
| Layers | 60 (45 GatedDeltaNet + 15 full attention) | 40, all the same block type; layers 0–19 "causal encoder", 20–39 "decoder" (CED) |
| Hidden / residual | 4096, one residual stream | 5120, **4 parallel residual copies** (Hyper-Connections, `hc_mult=4`) → 20 480 floats of state |
| Attention | GQA 32 heads × 256, RoPE, plus linear attention | MLA-style single 512-d KV latent (`num_kv_heads=1`), 64 query heads × 512, q LoRA rank 1280, grouped output LoRA (`o_groups=8`, rank 1024), RoPE on last 64 dims, **learned attention sink**, inverse RoPE on the output |
| Attention context | full causal | **sliding window 128** on every layer + **compressed sparse attention** (CSA2): top-512 compressed positions chosen by an indexer |
| KV sharing | none | layers 2–19 share the compressed KV of layers 2/8/14 (ratio 2 = one latent per 2 tokens); layers 20–39 share layer 20's (ratio 1). Indexers at 2, 8, 14, 20, 24, 28, 32, 36; layers in between reuse the previous top-k |
| Hierarchical indexer | – | layer 20 also picks 2 048 blocks × 8 positions as a candidate pool; indexers 24–36 only score inside it |
| MoE | 512 experts, top-10 (run at K=4), softmax, shared expert with sigmoid gate | **384 experts, top-6**, `sqrt(softplus)` scores, selection bias (`noaux_tc`), weights renormalised × 1.5, shared expert added with weight 1, **SwiGLU clamped at ±10** |
| Expert size | 3 × [1024×4096] 4-bit affine (scale+bias/64) = 7.08 MB | 3 × [2304×5120] **MXFP4** (E2M1 nibbles, E8M0 scale per 32) = 17.69 MB + 1.11 MB scales |
| Dense weights | 4-bit affine | **FP8 E4M3 with one E8M0 scale per 32×32 block**; BF16 for gate/compressor/embed/head; F32 for hc |
| Extra memory | – | **Engram**: at layers 1 and 14, 24 n-gram hash lookups per token into two 384 M-row × 256 FP8 tables (101 GB each), projected by a [25600×6144] FP8 matrix and gated into each residual copy |
| Speculative decoding | MTP (tested, break-even) | DSpark: 3 draft blocks (128 experts, top-3, 7.2 GB) drafting 5 tokens; optional |
| Vocab / tokenizer | 248 320, Qwen byte-level BPE | 129 280, DeepSeek byte-level BPE with a different pre-tokenizer (digits split 1–3, CJK runs, custom regex); chat format is not Jinja but `encoding/encoding.py` (DSML tool calls) |
| RoPE | theta 1e7, partial 0.25 | theta 1e4 (window) / 1.6e5 (compressed) with **YaRN** factor 16 over 64 K |

Two details that matter for the port and are easy to miss:

- **Single-Pass mHC.** Each sub-block computes `(pre, post, comb)` from the *current* residual
  with a [24 × 20480] projection followed by Sinkhorn normalisation of the 4×4 `comb`, but the
  `pre` it produces is used by the **next** sub-block (`Block.forward` returns `ffn_pre` for the
  next layer's attention). The first layer starts with the one-hot `pre = [1,0,0,0]`.
- **CED at prefill.** Layers 21–39 read their global KV from layer 20's cache, and their own
  sliding-window KV only needs the last 128 hidden states. So a prompt only has to run layers
  0–20 for all tokens and layers 21–39 for the last 128 ("Decoder SWA Bounded Replay", tech
  report §2.2). The reference `model.py` does not implement this shortcut; we should, because
  prefill is SSD-bound for us too.

## 3. Exact footprint (from the 48 shard headers)

`uv run deepseek_v41/ds41_budget.py --headers ds41_headers/ --ram 36`

```
 288.78 GB  routed experts, MXFP4 (streamed)              40 × 384 × 18.80 MB
 202.76 GB  Engram n-gram tables, FP8 (streamed)          2 × (384 M rows × 264 B)
   9.85 GB  backbone dense (attn/shared/gate/hc/embed/head)  0.180 GB/layer + 2.65 GB embed+head
   7.22 GB  DSpark draft experts (optional)
   0.71 GB  DSpark draft dense (optional)
   0.97 GB  vision encoder + aligner (not needed)
 510.29 GB  total download (48 shards)
 501.38 GB  minimum kept for text inference
```

Unified memory plan, 36 GB:

| | as stored (FP8/BF16) | dense requantised to 4-bit |
|---|---|---|
| resident dense weights (mmap) | 9.85 GB | 4.26 GB |
| Metal scratch + KV (890 B/token main KV, 2.6 MB SWA rings) | 0.4 GB | 0.4 GB |
| macOS + apps reserve | 5 GB | 5 GB |
| **left for the page cache** | **20.8 GB (7.2 % of experts)** | **26.3 GB (9.1 %)** |

Qwen3.5 today runs with 35 GB of cache over 209 GB (16.7 %) and sees ~71 % hits. Expect a
materially lower hit rate here; the budget script assumes hits scale with the square root of the
coverage ratio (47 % at 36 GB), which is a guess to be replaced by a measurement.

Disk on the 1 TB laptop: 501 GB fits only if the Qwen3.5 download **or** its `packed_experts/`
(209 GB + 218 GB today) goes. Since we read DeepSeek experts straight from the shards there is
no second copy to make room for; delete the 48 shards' vision/MTP files (8.9 GB) if needed.

## 4. Per-token cost model and the levers we have

```
decode, K=6:  40 × 6 × 18.80 MB = 4.51 GB expert bytes/token
              47 % hits → 2.41 GB from SSD  @  8 GB/s = 301 ms
                          2.10 GB from RAM  @ 14 GB/s = 150 ms   (pread copy of cached experts)
              + 40 layers × ~2.2 ms GPU/CPU (serial with SSD on unified memory) = 88 ms
              + Engram 2 × ~1 ms
              ≈ 541 ms/token → 1.8 tok/s     (cold: 654 ms → 1.5 tok/s; 100 % cached: 412 ms → 2.4)
decode, K=4:  3.01 GB/token → 391 ms → 2.6 tok/s (cold 2.1, fully cached 3.3)
```

The "100 % cached" line is the ceiling of the current design: even with every expert in RAM,
4.5 GB/token × (1/14 GB/s) = 322 ms goes into copying page-cache pages into Metal buffers. The
Qwen engine lives with that because its per-token volume is 2.7× smaller.

### More RAM, newer chip (`ds41_budget.py --ram N [--gpu-ms-per-layer 1.5] [--zero-copy]`)

M5 Max assumptions, from Apple's spec page and press measurements: 614 GB/s memory bandwidth
(vs ~400 on the M3 Max, so the GPU term shrinks to ~1.5 ms/layer), SSD ~13.6 GB/s sustained
read (no better than the 17.5 GB/s measured on this M3 Max, so the SSD term is unchanged at
~8 GB/s effective for 18 MB reads).

| RAM | page cache | est. hit rate | pread design, K=6 / K=4 | zero-copy resident experts, K=6 / K=4 |
|---|---|---|---|---|
| 36 GB (M3 Max) | 21 GB (7 %) | 47 % | **1.8 / 2.6 tok/s** | not worth it |
| 48 GB (M3 Max) | 33 GB (11 %) | 59 % | 2.0 / 2.7 | – |
| 64 GB (M5 Max) | 49 GB (17 %) | 71 % | 2.2 / 3.1 | **4.5 / 5.9** |
| 128 GB (M5 Max) | 113 GB (39 %) | 85 % | 2.4 / 3.3 | **6.8 / 8.4** (9.3 / 10.9 if hits reach 92 %) |

Reading the table: with the current `pread()`-into-a-scratch-buffer design, going from 36 GB to
128 GB buys ~0.5 tok/s, because the copy of cached experts replaces the SSD wait almost one for
one. The RAM only pays off with a **zero-copy resident-expert design**: cached experts stay in
GPU-visible memory (a page-aligned `mmap` of the shard wrapped with
`newBufferWithBytesNoCopy`, or a pinned LRU of Metal buffers) so a hit costs nothing but the
GPU's own ~10 ms/token read at 400–600 GB/s, and only misses touch the SSD. CLAUDE.md records
that custom caches lost to the page cache on 48 GB (GPU memory pressure, "Trust the OS"); at
128 GB the trade-off flips because 110 GB of resident experts is 38 % of the model and the
misses are rare enough that SSD DMA no longer fights the GPU every layer. This is the same
regime the Latent Space post's "128 GB M5 Max, SSD streaming unexpectedly fast" report was in.
Prefill is RAM-independent: ~18 s of SSD for a 2 000-token prompt whatever the machine.

Levers, in the order I would try them on the machine:

1. **K=4 instead of 6** (`--k 4`): −33 % I/O. Qwen3.5 already runs at K=4 instead of its native 10
   with "excellent" quality; V4.1's renormalised weights make this a fair experiment. Measure.
2. **Dense weights to 4-bit** (5.6 GB more page cache): affine-quantise wq_b/wo_a/wo_b/shared
   expert/embed/head at extraction time with the existing MLX-style layout so the *current*
   `dequant_matvec_4bit_v5` kernel serves them. Norms, hc, gate stay as is.
3. **2-bit experts** (`--expert-bits 2`, ~2.3 GB/token, ~5 tok/s): re-quantising E2M1 (already a
   7-level format) to 2 bits will hurt more than it did for Qwen's 4-bit affine. Last resort.
4. **DSpark** drafting 5 tokens: verification of 5 tokens touches ~5× the experts, so it is
   break-even on I/O like MTP was (CLAUDE.md) unless neighbouring tokens share experts; the
   draft blocks add 7.2 GB of streamed experts. Skip for v1.
5. **Zero-copy resident experts** (`--zero-copy`): the only lever that turns RAM into speed; see the
   table above. Needs 64 GB or more to matter and contradicts the "Trust the OS" result on 48 GB,
   so it must be re-measured, not assumed.
6. What does *not* help: Engram is cheap (48 random 4 KB reads per token, ~13 KB), and the
   attention/indexer compute is small (64 heads × 512 dims × 640 positions per layer).

## 5. Data pipeline (implemented, validated)

All scripts in `deepseek_v41/`, run with `uv run`. They work from headers alone, so the
budget/index steps can run **before** the 510 GB download.

| step | script | output | validated here |
|---|---|---|---|
| fetch headers (11 MB) | `fetch_ds41_headers.py --out ds41_headers/` | 48 × `*.header.json` | yes, all 48 |
| budget | `ds41_budget.py --headers ds41_headers/ --ram 36 [--k 4 --dense-bits 4]` | report above | yes |
| expert index | `generate_ds41_expert_index.py --headers ds41_headers/ --model <snapshot>` | `ds41_expert_index.json` | yes: all 15 360 experts have the same layout |
| dense blob | `extract_ds41_dense.py --model <snapshot> --output metal_infer/` | `ds41_dense.bin` (9.85 GB) + `ds41_dense.json` (same manifest schema as `model_weights.json`) | dry-run: 1 254 tensors, layout planned; the copy itself needs the shards |
| Engram metadata | `export_ds41_engram_meta.py --tokenizer <snapshot> --output metal_infer/ds41_engram_meta.bin` | 518 KB: token map, primes, offsets, multipliers | **yes**: compressed vocab = 99 092 and table sizes = 384 006 168 / 384 016 682 exactly as in the checkpoint |

Expert layout found in every layer (why no repack is needed): the exporter sorted tensors by
dtype then name, so for expert *E* the three FP4 weight tensors are contiguous
(`w1, w2, w3` = gate, down, up: 17 694 720 B) and the three scale tensors are contiguous
(1 105 920 B). Two `pread()` per expert instead of Qwen's one; still 6 MB+ reads, which the
NVMe handles at full bandwidth (results.tsv: "NVMe ignores scatter at 7 MB granularity").
Layer *L* is shard *L+3*; keep 40 fds open like today.

Engram addressing: `token → compressed id (99 092 classes)`, then for the current token and 3
before it, `rolling ^= id × multiplier[s]`; after each XOR the value hashes the (s+1)-gram and,
for each of 8 heads, `row = rolling mod prime[s-1][head] + offset`. 24 rows × 264 B per layer
per token, at deterministic addresses → `pread()` them in parallel with the layer's experts.

## 6. CPU reference ops and Metal kernels (implemented)

`metal_infer/ds41_ops.h` (C11, header-only, `make test-ds41`, all tests pass):

- E8M0 / E4M3 / E2M1 decoders (exhaustively tested against the arithmetic definitions).
- `ds41_mxfp4_matvec`, `ds41_fp8blk_matvec`, `ds41_bf16_matvec` (checked against a double
  reference, rel. err. < 1e-4).
- `ds41_swiglu_clamped`, `ds41_rms_norm`, `ds41_route` (sqrtsoftplus + bias + renorm × 1.5;
  note `norm_topk_prob` is skipped for K=1 like the reference).
- Hyper-connections: `ds41_hc_mixes`, `ds41_hc_split_sinkhorn` (doubly-stochastic checked),
  `ds41_hc_pre`, `ds41_hc_post`.
- YaRN RoPE frequencies.
- Engram: metadata loader, hashing (**bit-identical to the Python reference** for 16 positions ×
  24 rows with the real token map), row dequant, gate/add.

`metal_infer/shaders_ds41.metal` (drafted, **not compiled yet**): `dequant_matvec_mxfp4`,
`dequant_matvec_fp8_blk32` (both shaped like the winning `dequant_matvec_4bit_v5`: 8 rows per
threadgroup, x staged in threadgroup memory, LUT + FMA inner loop, power-of-two scale applied
once per 32-element block), `swiglu_clamped`, `hc_pre_collapse`, `hc_post_expand`,
`ds41_moe_combine`.

## 7. Engine work remaining (the real cost)

The Qwen engine is 7 000 lines of `infer.m` with the model hard-coded in `#define`s and the
pipeline specialised for GatedDeltaNet/full-attention layers. The clean path is a **second
engine file `ds41_infer.m`** that reuses the infrastructure (Metal context and buffer pools,
batched matvec dispatch, `IOThreadPool` / `parallel_pread_experts`, deferred CMD3, HTTP serve
loop, timing) and replaces the model code. Phases, each ending in a test on the Mac:

| phase | work | how to validate |
|---|---|---|
| **P1 kernels** | compile `shaders_ds41.metal`; add `--verify` comparing each kernel to `ds41_ops.h` on a real expert read from the shards | max rel. err. < 1e-3 vs CPU; time one expert (target ≈ 2.7× Qwen's 0.04 ms encode, bandwidth-bound) |
| **P2 CPU-correct forward** | `ds41_infer.m` decode loop entirely on CPU with `ds41_ops.h` (slow, minutes/token) : embed → hc expand → 40 × {hc_mixes, hc_pre, attn_norm, attention, hc_post, hc_mixes, hc_pre, ffn_norm, MoE + shared, hc_post} → hc_pre → norm → head; Engram at 1 and 14 | greedy tokens on a short prompt match the Python reference run on a rented GPU box (`inference/generate.py`, one prompt, `temperature 0`). Without this oracle we would be debugging blind |
| **P3 Metal pipeline** | port CMD1 (wq_a, q_norm, wq_b, wkv, kv_norm, RoPE), CMD2 (wo_a grouped, wo_b, hc_post, hc_mixes, ffn_norm, gate, shared expert), CMD3 (6 experts + combine + hc_post + hc_mixes + next attn_norm, deferred) | same tokens as P2; per-layer timing ≈ 2 ms + I/O |
| **P4 attention** | SWA ring buffer (128 × 512 FP8/token/layer), compressor for layers 2/8/14/20 (ratio 2 pools pairs with a softmax gate; ratio 1 is a plain projection), FP4 compressed-KV cache (E2M1 + E4M3 scale per 16), indexer (32 heads × 128, FP4 q/k, ReLU-weighted sum, top-512, position-sorted), candidate blocks at layer 20, shared runtime state across layers, sparse attention with sink and inverse RoPE | logits match P2 for prompts > 128 tokens (exercises compressed path) and > 4 096 (exercises top-k) |
| **P5 Engram** | 48 parallel `pread()` of 4 KB pages per token (two layers), row dequant, `wkv` FP8 matvec [25600×6144] on GPU, per-copy gate | compare against P2 CPU path |
| **P6 tokenizer + chat** | new pre-tokenizer in `tokenizer.h` (DeepSeek regex, digits 1–3, CJK), `export_tokenizer.py` for this `tokenizer.json`, chat template per `encoding/README.md` (BOS, `<｜User｜>`, `<｜Assistant｜>`, `</think>` for chat mode, DSML `<｜DSML｜ calls>` tool blocks), parse DSML tool calls in `chat.m` | round-trip encode/decode of `encoding/tests/*`; a bash tool call end to end |
| **P7 prefill** | batched prefill loading each touched expert once per layer; CED bounded replay (layers 21–39 only for the last 128 prompt tokens) | 2 000-token prompt in < 30 s |
| **P8 tuning** | K=4 vs 6, dense 4-bit, page-cache hit telemetry (reuse `--cache-telemetry`), fd choice per shard, `F_NOCACHE` for Engram pages | tok/s table in results.tsv |
| **P9 optional** | DSpark speculative decoding | only if P8 shows expert overlap between adjacent tokens |

Rough size: P1–P3 are the bulk of a working single-token decoder; P4 is the largest single
piece of new code (the whole CSA2 machinery); P6 is mechanical but fiddly. This is weeks of
work, not the 24 hours of the Qwen paper, mainly because of P4 and because the numerics must be
matched against a reference we cannot run on the laptop.

## 8. Risks and open questions

- **Performance ceiling.** The GPU side is fine: 4.5 GB of expert weights per token at
  418 GiB/s ≈ 10 ms. The ceilings are the SSD (4.5 GB at 8 GB/s = 560 ms cold) and, once the
  cache is warm, the pread copy of cached pages (4.5 GB at ~14 GB/s = 322 ms). Only K,
  quantisation, hit rate and a zero-copy design move those numbers.
- **36 GB vs 48 GB.** With 21 GB of cache the working set is 7 % of the experts. If the measured
  hit rate is far below 40 %, dense 4-bit (lever 2) is not optional.
- **Numerics we cannot check without the reference run.** Sinkhorn `comb`, attention sink,
  inverse RoPE on the output, the `o_groups` block-diagonal `wo_a`, and FP4 quantisation of the
  compressed KV cache (E2M1 with E4M3 scales per 16, applied *after* RoPE) are all easy to get
  subtly wrong. Budget a GPU rental for one reference run per phase.
- **Engram is not optional.** It is trained into the network at layers 1 and 14; skipping it
  saves 203 GB of disk but degrades the model in unknown ways. The lookups themselves are cheap.
- **Tokenizer normalisation drift.** The compressed token map must be byte-identical to
  training or every Engram lookup lands on the wrong row. The export script asserts the
  99 092 class count; keep `tokenizers` pinned.
- **Thermals / CPU.** Sinkhorn and routing are tiny, but the 4× residual stream means 4× the
  norm/mix traffic; keep it on the GPU as in the deployment "Mega-mHC" kernel (tech report §2.4.1).
- **Licensing.** MIT weights; no issue.

## 9. First steps on the M3 Max

```bash
# 0. build and run what exists
cd metal_infer && make test-ds41
xcrun -sdk macosx metal -c shaders_ds41.metal -o /tmp/ds41.air      # kernels compile?

# 1. budget check with the real headers (no download)
uv run deepseek_v41/fetch_ds41_headers.py --out ds41_headers/
uv run deepseek_v41/ds41_budget.py --headers ds41_headers/ --ram 36 --k 4 --dense-bits 4

# 2. make room, then download (resumable)
uvx --from huggingface-hub huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash \
  --exclude "model-00001-of-00048.safetensors" "model-0004[456]-of-00048.safetensors" \
  --local-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/main

# 3. derived files
SNAP=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/main
uv run deepseek_v41/generate_ds41_expert_index.py --headers ds41_headers/ --model $SNAP
uv run deepseek_v41/extract_ds41_dense.py --model $SNAP --output metal_infer/
uv run deepseek_v41/export_ds41_engram_meta.py --tokenizer $SNAP --output metal_infer/ds41_engram_meta.bin

# 4. P1: time one expert through dequant_matvec_mxfp4 and compare with ds41_mxfp4_matvec
```

Step 2 excludes the vision shard and the three DSpark shards (8.9 GB); add them back if P9 is
ever attempted.
