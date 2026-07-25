# Running Laguna S 2.1 on an M3 Max / 36 GB

poolside released **Laguna S 2.1** on 21 July 2026: 118B total parameters,
~8.5B activated per token, MoE, 1M context, open weights. It is a coding
model, so tool calling and JSON have to survive quantization — that rules out
the 2-bit tricks.

This document is the plan and the runbook for getting it onto a 36 GB
MacBook Pro through flash-moe's SSD expert streaming.

---

## 1. Why it doesn't just fit

| Format | On disk | Fits in 36 GB? |
|---|---|---|
| BF16 | ~236 GB | no |
| 4-bit MLX (`mlx-community/Laguna-S-2.1-oQ4e`) | ~64 GB | no |
| 3-bit GGUF (`UD-Q3_K_XL`) | ~50 GB | no |
| 2-bit GGUF (`UD-Q2_K_XL`) | ~40 GB | no — and 2-bit breaks JSON |
| Laguna **XS** 2.1, 4-bit | fits | yes, but it's a different model |

On a 36 GB machine the OS and the engine need ~10 GB, so roughly 24–26 GB is
available as page cache. Nothing at usable quality fits.

**But the model only activates ~8.5B parameters per token.** That is exactly
the case flash-moe was built for: keep the non-expert weights resident, stream
the K active experts per layer from SSD, and let the page cache keep the hot
ones. The 397B Qwen model already runs this way at 4.4 tok/s on 48 GB —
Laguna S is a third of the size, so the arithmetic is friendlier:

|  | Qwen3.5-397B-A17B | Laguna S 2.1 |
|---|---|---|
| Expert bytes on disk (4-bit) | 209 GB | **65 GB** |
| Resident fraction (page cache) | ~17% | **~35%** |
| Bytes read per token | 1.6 GB (K=4 of 10) | 2.55 GB (K=10) / 1.5 GB (K=6) |
| Active params | 17B | **8.5B** |

Half the GPU work, a third of the model, twice the cache residency. The
per-token I/O is the one number that gets worse at full K=10, and `--k 6`
trades a little routing fidelity to bring it back under Qwen's figure.

**Projected throughput: 3–6 tok/s.** That is a projection from the measured
Qwen pipeline, not a measurement — see §6.

---

## 2. What was built

The engine was hardcoded to Qwen3.5's shape. It is now built from an
architecture profile:

```bash
make                    # Qwen3.5-397B-A17B  -> ./infer      (unchanged)
make MODEL=laguna_s     # Laguna S 2.1       -> ./infer_laguna
```

| File | Role |
|---|---|
| `metal_infer/arch.h` | profile selection + derived expert layout, layer taxonomy |
| `metal_infer/arch_qwen35_moe.h` | the old constants, moved out of `infer.m` |
| `metal_infer/arch_laguna_s.h` | Laguna profile — **regenerate before use** |
| `metal_infer/arch_runtime.h` | rope tables, gating, router activation |
| `metal_infer/gen_arch_header.py` | writes the profile from the real checkpoint |
| `arch_profile.py` | same geometry for the Python packers |

`main.m` (the standalone single-expert micro-benchmark) still carries its
own Qwen constants and is not part of this path; `infer.m` is.

Four things differ from Qwen and are now implemented behind the profile:

1. **No linear attention.** Qwen interleaves 45 GatedDeltaNet layers with 15
   global-attention layers. Laguna has 48 softmax-attention layers: 12 global,
   36 sliding-window. The GatedDeltaNet path is simply not compiled in.
2. **Sliding window (512).** A windowed layer's KV cache is capped at 2×512
   entries and compacted back to 512 when it fills — one memmove every 512
   tokens. Attention reads a contiguous suffix, so the Metal kernels are
   unchanged: the KV buffers are just bound at an offset. This also collapses
   KV memory: 36 of 48 layers cost 512 slots instead of growing with context.
3. **Per-head softplus output gate** instead of Qwen's per-channel sigmoid
   fused into `q_proj`. Both layouts are supported; the generator detects
   which one the checkpoint uses.
4. **Sigmoid router.** Laguna scores each expert independently and renormalizes
   the top-k, where Qwen softmaxes first.

Plus **YaRN rope** (factor 128 over an 8192-token base window) on the global
layers only — the windowed layers never look past 512 tokens and stay on
unscaled frequencies.

---

## 3. Runbook

Everything below runs on the Mac. Python scripts use `uv`.

### 3.1 Get the weights (~64 GB)

flash-moe consumes the MLX 4-bit layout (`{weight, scales, biases}` per
tensor, group size 64):

```bash
uv run huggingface-cli download mlx-community/Laguna-S-2.1-oQ4e \
    --local-dir ~/models/laguna-s-4bit
```

Check the free space first — the packed copy needs another ~65 GB, so budget
~130 GB total. You can delete the safetensors once packing is verified.

### 3.2 Generate the architecture profile

**Do this before building.** The committed `arch_laguna_s.h` is a template
with placeholder values and it will emit a `#warning` until it is replaced:

```bash
uv run metal_infer/gen_arch_header.py \
    --model ~/models/laguna-s-4bit \
    --arch laguna_s
```

It reads `config.json`, `tokenizer_config.json` and the safetensors headers,
derives every constant, and refuses to guess: if a key it needs is missing or
a tensor shape contradicts the config, it stops and tells you which. It prints
a summary — check that layers, experts, top-k, window, rope and the detected
gate kind match what you expect, then it writes `arch_laguna_s.h` and
`arch_laguna_s.json`.

### 3.3 Pack the experts

```bash
uv run generate_expert_index.py --arch laguna_s \
    --model ~/models/laguna-s-4bit --output expert_index_laguna.json

uv run repack_experts.py --arch laguna_s --index expert_index_laguna.json --dry-run
uv run repack_experts.py --arch laguna_s --index expert_index_laguna.json
```

This writes `packed_experts_laguna/layer_XX.bin`, 1.36 GB per layer, 48
layers, ~65 GB total. Each expert is a 5,308,416-byte contiguous block, so a
routed expert is exactly one `pread`.

### 3.4 Non-expert weights and tokenizer

```bash
uv run metal_infer/extract_weights.py --model ~/models/laguna-s-4bit
uv run metal_infer/export_tokenizer.py --model ~/models/laguna-s-4bit
uv run metal_infer/export_vocab.py     --model ~/models/laguna-s-4bit
```

`extract_weights.py` keeps everything that is not a routed expert, including
the attention gate projection.

### 3.5 Build and run

```bash
cd metal_infer && make MODEL=laguna_s
./infer_laguna --prompt "Write a Python LRU cache with a TTL" --tokens 200 --k 10 --timing
```

Start with `--timing` on a short prompt: the per-layer breakdown tells you
immediately whether you are I/O bound (`expert_io`) or compute bound
(`cmd1_wait` / `cmd2_wait`), which is what §5 is for.

---

## 4. Memory budget on 36 GB

| | |
|---|---|
| Non-expert weights (mmap, 4-bit) | ~2.5 GB |
| KV cache, 12 global layers @ 8k context | ~800 MB |
| KV cache, 36 windowed layers (512 each) | ~150 MB |
| Metal scratch | ~200 MB |
| **Engine total** | **~3.7 GB** |
| Left for OS + page cache | ~32 GB |

That leaves roughly 24 GB of page cache against 65 GB of experts — about 35%
resident, twice Qwen's ratio. `MAX_SEQ_LEN` in the profile caps the global-layer
KV cache; lower it if you want more page cache and don't need long context.

---

## 5. Tuning knobs, in the order worth trying

1. **`--k 6` instead of `--k 10`.** Cuts per-token I/O by 40%. Laguna routes
   top-10 of 256; the tail weights after renormalization are small. Qwen ships
   at K=4 of a native 10 for exactly this reason. Check quality on tool calls
   before keeping it.
2. **`MAX_SEQ_LEN`.** Every 8k of context costs ~800 MB that the page cache
   would rather have.
3. **Stay at 4-bit.** 2-bit is implemented (`--2bit`, ~36 GB) and it will be
   faster, but on Qwen it turned `"name"` into `\name\` and broke tool calling.
   On a coding model that is not a trade worth making.
4. **Leave the page cache alone.** No custom cache beat the OS page cache on
   this engine — see the discard table in `CLAUDE.md`. With 35% residency the
   OS should do better here than it did on Qwen.

---

## 6. What is verified, and what is not

Verified here:

- Both architecture profiles compile, and their derived expert layouts agree
  with the byte offsets `repack_experts.py` has always used (checked against
  the Qwen numbers: 7,077,888 B/expert, offsets 0 / 2097152 / 2228224 / …).
- `gen_arch_header.py` end-to-end on a synthetic checkpoint with Laguna's
  published shapes: it derives 48 layers (12 global every 4, 36 sliding @ 512),
  256 experts top-10, 4-bit group 64, YaRN, the per-head gate, and emits a
  header that compiles clean.
- The Qwen build path is unchanged by construction: every new branch is behind
  a profile `#if`, and the Qwen profile reproduces the old constants exactly.

**Not verified — this could not be run here:**

- Nothing was compiled against Metal or executed. This environment is Linux
  with no GPU, no macOS toolchain and no network access to Hugging Face, so
  `infer_laguna` has never been built or run, and no token has been generated.
- The committed `arch_laguna_s.h` values for `HIDDEN_DIM` (4096),
  `MOE_INTERMEDIATE` (768), `NUM_ATTN_HEADS` (32) and `ROPE_THETA` are
  *inferred*, not read from poolside's config.json — they are the values that
  reproduce both the 118B parameter count and the ~64 GB 4-bit checkpoint
  size. Step 3.2 replaces them with the real ones. Do not skip it.
- Tensor naming for the gate projection is a search over five plausible names.
  If the checkpoint uses something else, `gen_arch_header.py` reports the
  q_proj shape and gate kind it found, the engine prints a loud warning at
  startup, and the name goes in the list in `build_layer_cache`.
- The 3–6 tok/s figure is extrapolated from the Qwen pipeline's measured
  per-layer costs, scaled for half the active parameters and a 30-core GPU
  (~300 GB/s vs the 40-core's ~400). Your SSD's sequential read matters as
  much as the GPU here — measure it before trusting the estimate.

## 7. Two things the port does not cover yet

Both surfaced from poolside's release material after the engine work, and
neither blocks generating tokens — they block *using* the model the way it
expects to be used.

**Tool calling uses poolside's own XML protocol** (`poolside_v1`), with
thinking blocks interleaved between tool calls and toggled per request via
`enable_thinking`. `chat.m` parses Qwen's Hermes-style
`<tool_call>{"name":…,"arguments":{…}}</tool_call>` — the XML wrapper happens
to look similar, but the body is poolside's format, not that JSON object. The
parser in `chat.m` (around the `<tool_call>` scan) needs a second variant
before agentic use works. Plain completion and chat are unaffected.

**The quantized checkpoints are configured for 256K context, not 1M.** Only
the BF16 checkpoint carries the full 1,048,576. That means the YaRN factor in
the 4-bit config is probably 32 (256K / 8192), not the 128 quoted for BF16 —
`gen_arch_header.py` reads whatever the checkpoint actually declares, so the
generated header will be right either way, but do not be surprised when the
printed rope summary disagrees with the launch blog. The quantized releases
also ship an FP8 KV cache; flash-moe keeps KV in fp32, which is why §4 budgets
~800 MB for 8k of context rather than a quarter of that.

---

## 8. If you want something running tonight

The port needs a build and a 64 GB download before it produces a token. Two
things work immediately:

- **Laguna XS 2.1** at 4-bit fits in 36 GB with room to spare, via `mlx_lm` or
  Ollama. Same family, same tokenizer, much smaller — a good way to validate
  prompts and the chat template while the S weights download.
- **llama.cpp b10087+** with `UD-Q2_K_XL` (~40 GB) will run S on this machine,
  but 40 GB on a 36 GB box means it pages continuously, and 2-bit is the
  quantization that breaks JSON. Treat it as a correctness baseline to diff
  flash-moe's output against, not as a daily driver.
