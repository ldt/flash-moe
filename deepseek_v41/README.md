# DeepSeek-V4.1-Flash tooling

Groundwork for running `deepseek-ai/DeepSeek-V4.1-Flash` (552B backbone + 196B Engram,
16B active per decode token) with the Flash-MoE SSD-streaming engine. The design, budget
and phased plan are in [`docs/deepseek-v41-flash-plan.md`](../docs/deepseek-v41-flash-plan.md).

All scripts run with `uv run` and work from safetensors **headers** alone, so the "does it
fit?" questions can be answered before downloading 510 GB.

```bash
uv run deepseek_v41/fetch_ds41_headers.py --out ds41_headers/           # 11 MB, 48 headers
uv run deepseek_v41/ds41_budget.py --headers ds41_headers/ --ram 36     # disk / RAM / tok-s estimate
uv run deepseek_v41/generate_ds41_expert_index.py --headers ds41_headers/ --model <snapshot dir>
uv run deepseek_v41/extract_ds41_dense.py --model <snapshot dir> --output metal_infer/
uv run deepseek_v41/export_ds41_engram_meta.py --tokenizer <snapshot dir> --output metal_infer/ds41_engram_meta.bin
```

| file | purpose |
|---|---|
| `config.json` | official `inference/config.json` (flat keys) |
| `ds41_common.py` | header parsing, tensor classification, expert record layout |
| `fetch_ds41_headers.py` | HTTP range-fetch of the 48 shard headers |
| `ds41_budget.py` | exact on-disk footprint, unified-memory plan, per-token I/O and tok/s model |
| `generate_ds41_expert_index.py` | per-expert `pread()` offsets straight into the shards (no repack) |
| `extract_ds41_dense.py` | resident weights → `ds41_dense.bin` + manifest (`--dry-run` needs headers only) |
| `export_ds41_engram_meta.py` | Engram token map / primes / multipliers, bit-exact port of `engram.py` |

C side: `metal_infer/ds41_ops.h` (CPU reference ops, `make test-ds41`) and
`metal_infer/shaders_ds41.metal` (MXFP4 / block-FP8 kernels, not yet compiled on a Mac).
