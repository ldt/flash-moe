# DeepSeek-V4.1-Flash at 50+ tok/s locally: what actually does it today

Companion to [deepseek-v41-flash-plan.md](deepseek-v41-flash-plan.md). That document is about
the Flash-MoE SSD-streaming approach on a laptop (2–8 tok/s). This one answers a different
question: **which local setup, with any software stack, decodes DeepSeek-V4.1-Flash at 50 tok/s
or more, single stream, today (September 2026)?** Numbers below are the ones people have
published, not our model's estimates; each row links to its source.

## Reality check first

- The official vLLM recipe wants **614 GB of VRAM** (weights + Engram tables in GPU memory,
  ×1.2 headroom): one GB200 NVL4 tray, an 8×H200 node or 4×MI355X
  ([vLLM recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)). That is the
  datacenter path. `torch.compile` is unsupported, DSpark speculative decoding is on by default
  with 5 draft tokens.
- Every local setup below gets under that number the same way: **the two 101 GB Engram tables
  stay on NVMe** (24 random 264-byte rows per token per Engram layer, deterministic addresses),
  and only the 289 GB of MXFP4 experts + 10 GB dense need to be resident. That is exactly the
  split our `ds41_budget.py` assumes.
- Engine support is fresh: Docker images or patched forks of SGLang/vLLM, SM120-specific kernel
  work, no pip wheel. Expect to build things.

## Measured single-stream decode, by configuration

| Setup | Resident memory | Software | Single-stream decode | Aggregate | ≥ 50 tok/s? | Source |
|---|---|---|---|---|---|---|
| **4× RTX PRO 6000 Blackwell 96 GB**, 128 GB DDR5, NVMe, 275 W/GPU cap, PCIe (no NVLink) | 384 GB VRAM: native FP4 experts + FP8/BF16 dense; Engram in a 64 GiB DDR5 cache + NVMe | patched SGLang, DSpark block 5 (79–88 % acceptance) | **196–231 tok/s** | 599–747 tok/s at 8 streams; prefill 5.8–7.4 k tok/s; 500 k-token inputs OK | **yes, 4×** | [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) |
| same class, other users | 4× RTX PRO 6000 | patched vLLM, no vision, DSpark off (tool-call failures) | ~170 tok/s | – | yes | [HF discussion #28](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/28) |
| **4× DGX Spark** (128 GB unified each), ConnectX-7 RoCE | 4 × ~82 GiB weights (MXFP4 experts) + KV; Engram on disk | vLLM TP4 pinned commit, FlashInfer SM12.1 prebuilt, DSpark on | **code 52–74, math 47, reasoning 39, prose 23** tok/s | 214–260 tok/s at 6 streams; prefill 0.9–1.2 k tok/s | **borderline**: yes on code, no on prose | [NVIDIA forum](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-552b-moe-on-4x-dgx-spark-tp4-77-2-tok-s-c1-on-peak-52-code-72-tok-s-on-a-warm-code-run-47-math-39-reasoning-23-prose/382897), [tonyd2wild repo](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) |
| **Mac Studio M3 Ultra 256 GB** | 197 GB resident: 3-bit base, 27 of 40 MoE layers requantised to 2-bit; Engram on disk; no expert paging | oMLX 0.7.0.dev2 + custom Metal kernels, CED prefill, DSpark 3 drafts (75 % code / 59 % prose acceptance) | **code 29.9, prose 25.1** tok/s; prefill 551 tok/s | – | no | [drowzeys repo](https://github.com/drowzeys/keys-Mac-oMLX-0.7.0.dev2-DeepSeek-V4.1-Flash-oQ3e-2b27-CED-MTP) |
| Mac Studio M3 Ultra 512 GB | 4/8-bit build, ~427 GB resident | MLX / oMLX | no published V4.1 number; V4-Flash 4-bit did 29.5 tok/s on the same machine | – | no (today) | [modelfit.io](https://modelfit.io/blog/deepseek-v4-1-flash-mac-memory-requirements/) |
| Mac Studio M3 Ultra 256 GB, 2-bit MLX | 239 GB | mlx-lm | 9.5 tok/s | – | no | [modelfit.io](https://modelfit.io/blog/deepseek-v4-1-flash-mac-memory-requirements/) |
| 1× RTX 5090 + dual Xeon, experts on CPU | ≥ 200 GB DDR4/5 | KTransformers (V4-Flash only, no V4.1 yet) | 20–28 tok/s on **V4-Flash** | – | no | [ktransformers doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md), [gist](https://gist.github.com/RockmSockmJesus/30a195ccd9b62e981ec2676a99a57b7e) |
| 2× RTX PRO 6000 (192 GB) | fits **V4-Flash** (284B) in FP4+FP8 | vLLM, no speculation | 106 tok/s on **V4-Flash**, 96 at 1 M context | prefill 13 k tok/s | n/a for V4.1: 289 GB of experts do not fit in 192 GB | [Millstone AI](https://www.millstoneai.com/inference-benchmark/deepseek-v4-flash-fp4fp8-2x-rtx-pro-6000-blackwell) |
| Flash-MoE style SSD streaming, 36–128 GB Mac | 10 GB dense resident, experts streamed | this repo's plan | 2–8 tok/s (estimate) | – | no | [plan](deepseek-v41-flash-plan.md) |

## The answer

**The only configuration that clears 50 tok/s single-stream with margin, on the unpruned model
at native precision, is a 4× RTX PRO 6000 Blackwell workstation** (384 GB of VRAM, 128 GB or
more of DDR5, a fast NVMe for the Engram tables, patched SGLang, DSpark on). It does ~200 tok/s
single stream and 600–750 tok/s aggregate, handles 500 k-token inputs, and passes tool-call and
JSON round trips. Order of magnitude: 4 × 8–9 k€ for the cards plus a Threadripper/EPYC host,
so 40–45 k€, ~1.2–1.5 kW under load.

Cheaper options and why they fall short of the target:

- **4× DGX Spark (~16 k$)** is the cheapest rig that touches 50 tok/s, but only on code
  (52–74); prose sits at 23 tok/s and there is an unresolved "GB10 slow state" that makes decode
  steps vary 63–94 ms. Good for agentic coding, not a general 50 tok/s machine.
- **2× RTX PRO 6000 (192 GB)** would be the sweet spot but the experts alone are 289 GB. Making
  it fit means REAP pruning to 272 experts (~205 GB, still too big) or 2-bit experts (~153 GB,
  quality unproven). Nobody has published a V4.1 number on two cards.
- **Mac Studio**: the best published result is 30 tok/s on a 256 GB M3 Ultra after requantising
  most experts to 2-bit. A 512 GB machine has the memory for the full 4-bit model, but no engine
  is within 2× of 50 tok/s on Apple Silicon today. Our own bandwidth arithmetic (12–13 GB of
  weights per token at 819 GB/s ≈ 15 ms) says ~65 tok/s is the physical ceiling without
  speculation, so 50 is *possible* with a purpose-built engine plus DSpark, but it is a
  research project, not a purchase.
- **CPU expert offload (KTransformers)** tops out around 28 tok/s on the smaller V4-Flash and
  does not support V4.1 yet; DDR5 bandwidth (300–500 GB/s) caps it well under 50 for 4.5 GB of
  experts per token.

## Two levers that matter on any GPU rig

1. **DSpark speculative decoding** is what turns 40 tok/s into 200: 5 draft tokens, 75–88 %
   acceptance reported. It is on by default in vLLM; one report disabled it because of
   tool-calling benchmark failures, so validate tool calls with it on.
2. **Engram on NVMe with a RAM cache** costs almost nothing (0.4 % of runtime in the Mac build,
   "only slightly faster" pinned in RAM on the 4-GPU rig). Do not buy 256 GB of RAM to hold the
   tables; buy a fast NVMe.
