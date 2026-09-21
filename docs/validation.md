# Model validation and assumptions

The performance model lives in `site/js/core/` (pure functions, no DOM) and is exercised by `tests/` (`npm test`, or `node --test 'tests/*.test.mjs'`).
It follows PLAN.html §6: a per-layer op list → roofline per op → α–β collectives → dual-batch overlap → pipeline formula.

## Anchors (tests/anchors.test.mjs)

Acceptance criterion from the plan: within ±30% of published absolute numbers using only the **global** efficiency knobs (never per-anchor fudge factors).

| Anchor | Source | Measured | Model | Ratio |
|---|---|---|---|---|
| DeepSeek-V3 decode, GB200 NVL72, 48-GPU EP, 2K ISL, batch 1408, FP8 attention + NVFP4 MoE | LMSYS GB200 NVL72 Part II (Sept 2025) | 13,386 output tok/s/GPU | ≈ 16.6k | **1.24×** |
| DeepSeek-V3 prefill, 2 GPUs per instance, 2K ISL | same | 26,156 input tok/s/GPU | ≈ 30.9k | **1.18×** |
| Decode batch capacity, V3, 48-GPU EP, 2K ISL | same (1,408 requests "to fill the KV cache") | 1,408 | 1,897 max/rank (fits) | fits |
| Wide-EP trend: per-GPU throughput rises with EP | TRT-LLM wide-EP blog | up to 6× with MTP | +25% EP8→EP72 (no MTP) | direction only |
| DeepSeek-V4-Pro interactivity reach on GB200 NVL72 | NVIDIA marketing ("over 150 tok/s/user") | ≥ 150 | ≈ 300 with MTP1 | loose upper bound |

Both absolute anchors are optimistic by 18–24%, the usual sign for a roofline model (real kernels have more small inefficiencies than the knobs capture).
The three knobs that were fitted are `attnEff` (0.2), `gemmEff` (0.6) and `moeEff` (0.55); everything else is at its first-principles default.

Ranking checks (also tested): wide-EP + attention-DP beats TP for decode at high batch; PP does not reduce per-user latency; DBO helps at large batch and not at tiny batch;
speculative decoding helps at small batch and fades at large batch; CP shards KV reads; MLA replicates KV under TP.

## Hand-checked derivations (tests/model.test.mjs, tests/memory.test.mjs)

- Parameter counts: V3 671.0B / 36.6B active; Llama-3.1-405B 405.8B; Qwen3-235B 235.1B / 21.6B; Kimi-K2 1026B; V4-Pro 1572B / 48.2B (published 1.6T / 49B); V4-Flash 284.1B / 13.0B (published 284B / 13B).
- KV per token: V3 MLA 576 × 61 B = 35.1 KB (FP8); the transcript's 10T / GQA-8 example = 204,800 B; V4 hybrid CSA/HCA ≈ 5.4 KB at 1M context (paper: ≈ 10% of V3.2).
- V3 experts: 654.0B over 58 layers × 256 × 3·7168·2048.
- EP=72 with 256 experts → 288 slots (256 + 32 redundant); 384 experts (V4-Pro) → 432.

## Constants (all `assumed` until calibrated) — site/js/core/data.js

| Constant | Default | Meaning |
|---|---|---|
| `gemmEff` | 0.60 | dense GEMM ceiling as a fraction of dense peak (small-M penalty `M/(M+gemmHalfM)` on top) |
| `moeEff` | 0.55 | grouped / expert GEMM ceiling |
| `attnEff` | 0.20 | attention kernel fraction of peak (FP8 peak for FP8 KV); absorbs DP-attention stragglers and decode kernel gaps |
| `hbmEff` | 0.85 | achievable HBM bandwidth fraction |
| `overlap` | 0.80 | share of hideable communication actually hidden by dual-batch overlap |
| `alphaUs` | 10 | base collective latency; per collective: all-reduce one-shot 0.8α, ring step 0.4α, all-to-all 1.5α, p2p 0.5α |
| `linkEff` | 0.70 | NVLink efficiency for all-to-all / p2p |
| `floorUs` | 30 | per-layer kernel-launch floor with CUDA graphs (doubled with DBO) |
| `overheadMs` | 0.3 | per-step scheduler / sampling overhead |
| `memUtil` | 0.90 | gpu-memory-utilization cap; plus 3 GB for CUDA graphs and communicator buffers |
| hardware | see `HARDWARE.gb200` | 186 GB usable HBM, 8 TB/s, dense 2.5 / 5 / 10 PF (BF16 / FP8 / FP4), NVLink 900 GB/s per direction — NVIDIA GB200 NVL72 datasheet (sparse ÷ 2) |

## Modeling choices worth knowing

- **Mean-field steady state.** Average decode context = context length + 512 (output length is fixed at 1,024 tokens; no prefix cache); no queueing or tails.
- **Layer averaging.** Dense-first layers and MoE layers are costed separately; CSA/HCA layers are averaged per layer. Pipeline stages get `ceil(layers/PP)` layers of the average mix.
- **Prefill.** Chunk size is per attention rank. Throughput assumes a full pipeline; latency adds `(PP−1)` chunk times. Attention FLOPs use 2·(d_qk + d_v)·pairs; MLA prefill runs un-absorbed.
- **Decode MLA** runs absorbed (KV read = latent, FLOPs ∝ 576 + 512 per head per token).
- **MoE.** Touched experts `1 − (1 − k/E)^b`; max-rank imbalance `μ + √(2μ ln EP)` plus a skew term that EPLB (+12.5% redundant slots) mostly removes; dispatch sends one copy per destination rank.
- **Speculative decoding (MTP).** Verify step processes γ+1 tokens per request (raising expert touch), drafts run γ layer-steps, `E = (1 − α^(γ+1)) / (1 − α)`.
- **DeepSeek-V4.** Dimensions from the HF `DeepseekV4Config` and NVIDIA / vLLM model cards. Layer schedule (2 HCA bootstrap layers, then alternating CSA / HCA), the FP8/BF16 KV entry layout and Pro's output rank of 1024 are **assumed**. The four mHC residual streams only enter through the 4× wider pipeline hop; hash-routed layers are costed as ordinary MoE layers.

## Not implemented yet (PLAN milestones M6+)

Aggregated / chunked-prefill-with-decode mode (removed from the product for now), KV offload to Grace, attention-FFN disaggregation, DWDP, discrete-event tails, HF `config.json` import,
calibration mode (fit efficiency knobs to user measurements), sensitivity tornado, other hardware, launch-command export. Uneven pipeline stage assignment is approximated.
