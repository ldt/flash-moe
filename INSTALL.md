# Installation Guide

Step-by-step setup for running Qwen3.5-397B-A17B on Apple Silicon.

## Requirements

- MacBook Pro with Apple Silicon (M3 Max or later recommended, 48 GB RAM minimum)
- ~430 GB free disk space (209 GB model + 218 GB repacked experts)
- Xcode Command Line Tools: `xcode-select --install`
- [uv](https://docs.astral.sh/uv/) for running Python scripts

## Steps

### 1. Download the model (~209 GB)

```bash
uvx --from huggingface-hub huggingface-cli download mlx-community/Qwen3.5-397B-A17B-4bit \
  --repo-type model \
  --local-dir ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/latest
```

Resumable — rerun the same command if interrupted.

### 2. Generate expert index

```bash
uv run generate_expert_index.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/latest \
  --output expert_index.json
```

Produces `expert_index.json` (mapping of expert tensors to safetensors offsets).

### 3. Extract non-expert weights (~2 min, produces 5.5 GB)

```bash
uv run metal_infer/extract_weights.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/latest \
  --output metal_infer/
```

Produces `metal_infer/model_weights.bin` and `metal_infer/model_weights.json`.

### 4. Export vocab

```bash
uv run metal_infer/export_vocab.py \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/latest/tokenizer.json \
  metal_infer/vocab.bin
```

Produces `metal_infer/vocab.bin` in the simple format expected by `infer.m`.

### 5. Export tokenizer

```bash
uv run metal_infer/export_tokenizer.py \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/latest/tokenizer.json \
  metal_infer/tokenizer.bin
```

Produces `metal_infer/tokenizer.bin` (BPE format for the C tokenizer).

### 6. Repack expert weights (~2-3 hours, produces 218 GB)

```bash
uv run repack_experts.py --index expert_index.json
```

Produces `packed_experts/layer_00.bin` … `layer_59.bin` (60 files × 3.63 GB each).
The model path is read from `expert_index.json` automatically.

### 7. Build

```bash
cd metal_infer
make        # builds infer and metal_infer
make chat   # builds interactive chat client
```

Metal shaders compile at runtime — no extra step needed.

### 8. Test

```bash
cd metal_infer
./infer --prompt "Hello" --tokens 10 --timing
```

Expected: ~4.4 tok/s at 4-bit quantization.

## Usage

```bash
# Single prompt
./infer --prompt "Explain quantum computing" --tokens 200

# Interactive chat with tool calling
./chat

# Server mode + chat client (two terminals)
./infer --serve 6601
./chat --port 6601

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing

# 2-bit mode (faster ~5.7 tok/s, but breaks JSON/tool calling)
./infer --prompt "Hello" --tokens 100 --2bit
```

## Cleaning

If need to free storage space, you can delete the original safetensors files.

The __46 original safetensors files__ are the raw download from HuggingFace. They were already transformed into `packed_experts/` (by `repack_experts.py`) and `model_weights.bin` (by `extract_weights.py`). __They are not read during inference.__

__Warning:__ After deleting the safetensors files, you cannot re-run `extract_weights.py` or `repack_experts.py` without re-downloading the model from HuggingFace. But for __running inference__, you don't need them.

```bash
# Safe to delete — saves 208 GB
rm ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/latest/model-*.safetensors
```

## Uninstall

Remove the remainingn weights:

```bash
rm ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/
```