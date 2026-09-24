# Model validation and assumptions

The performance model lives in `site/js/core/` (pure functions, no DOM) and is exercised by `tests/` (`npm test`, or `node --test 'tests/*.test.mjs'`).
It follows PLAN.html §6: a per-layer op list → roofline per op → α–β collectives → dual-batch overlap → pipeline formula.

## Anchors and calibration

The registry is `tests/anchors.data.mjs`. Each entry has its source URL, a verbatim quote, the setup (from the LMSYS reproduction recipes where they exist), what was assumed, and a status.
`tests/anchors.test.mjs` requires calibration and validation anchors to sit within ±30%. `npm run calibrate` (`scripts/calibrate.mjs`) fits `gemmEff`, `moeEff` and `attnEff`,
minimizing Σ (ln model/measured)² / 0.1² over the calibration anchors plus a log-normal prior on each knob taken from public kernel benchmarks. A test fails if the defaults in
`data.js` drift more than 5% from the fit, so re-run the calibration after changing the model or the anchors.

| Anchor | Status | Setup | Measured | Model | Leave-one-out |
|---|---|---|---|---|---|
| V3 decode, 2K ISL, FP8 attn + NVFP4 MoE | calibration | DP-attn 48 + EP48, batch 1408 | 13,386 out tok/s/GPU | 1.10× | 1.16× |
| V3 prefill, 2K ISL, FP8 attn + NVFP4 MoE | calibration | **4** GPUs ("4 ranks per instance"; the old test used 2), assumed DP-attn 4 + EP4, TBO, 16K chunks | 26,156 in tok/s/GPU | 1.12× | **1.38×** |
| R1 128K/8K end to end, no MTP | calibration | 3 × PP4 prefill + DEP8 decode, mem-fraction 0.72; **per GPU over all 20 GPUs** (`run_loader.py`: `tps / total_gpus`) | 147.9 out tok/s/GPU | 0.95× | 0.85× |
| V3 decode, 2K, BF16 attn + FP8 MoE | validation | as above, batch fills KV | 9,087 | 0.99× | — |
| 128K TTFT | validation | 1 × PP4 × TP1, one unchunked pass | 18.6 s | 1.11× | — |
| 128K FMHA, one layer | validation | our reading: full 128K causal MLA, 128 heads, FP8, TP1 | 277 ms | 0.93× | — |
| 128K KV capacity, DEP16 | validation | mem-fraction 0.70; "320 concurrent at DEP16" | 20 req/GPU | 0.95× | — |
| V3 prefill, 2K, BF16 attn + FP8 MoE | needs feature | FP8 experts need ≈ 163 GB/GPU on 4 GPUs; LMSYS offloads weights to Grace | 18,471 | — | — |

Sources: [LMSYS GB200 Part II](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/) (Sept 2025), [LMSYS GB300 long context](https://www.lmsys.org/blog/2026-02-19-gb300-longctx/) (Feb 2026),
[reproduction recipes](https://github.com/YAMY1234/srt-slurm/tree/gb300_blog/recipes/gb300-128k8k-blog), [profiles](https://github.com/sgl-project/sglang/issues/18703).

Fitted knobs: `gemmEff` 0.36 (prior 0.6), `moeEff` 0.40 (prior 0.55), `attnEff` 0.24 (prior 0.3). Two warnings go with these numbers:
- **`gemmEff` sits 2σ below its prior.** Only the 2K prefill anchor constrains it, and leave-one-out predicts that anchor at 1.38×. It is absorbing 2K-prefill costs the model lacks, so treat short-prompt prefill as the least certain regime.
- **The anchors span two software snapshots** (SGLang Sept 2025 and Feb 2026). One set of knobs has to cover both, and kernel improvements in between make that fit slightly inconsistent.

Earlier fits were wrong for identifiable reasons: prompts weren't packed into prefill chunks, the prefill instance size was wrong, prefill attention was costed in BF16,
and the long-context number was read as per decode GPU when it is per GPU over all GPUs.

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
| `gemmEff` | 0.36 | dense GEMM ceiling (fraction of dense peak), times tile padding: rows are rounded up to multiples of `gemmTileM` = 64. **Fitted** |
| `moeEff` | 0.40 | grouped / expert GEMM ceiling, same padding on rows per expert. **Fitted** |
| `attnEff` | 0.24 | **decode** attention tensor-core efficiency against the peak of its math dtype (FP8 when KV is FP8). **Fitted**; no public FP8-math MLA decode benchmark |
| `kvBwEff` | 0.78 | HBM fraction reached streaming KV, from the latency-vs-context slope in [FlashInfer #4390](https://github.com/flashinfer-ai/flashinfer/issues/4390) (trtllm-gen MLA decode, B300) |
| `attnEffPrefill` / `softmaxSerial` | 0.80 / 0.26 | prefill attention: FP8 MMA efficiency, and the share of the shorter of (MMA, exp) left un-overlapped. Solved from the GB200 / GB300 FMHA pair (277 / 205 ms) with exp at the MUFU rate |
| `attnFoldQ` | false | speculative verify re-reads KV per query row (trtllm-gen); true folds query rows into one pass (cute-dsl `fold_sq`) |
| `attnMath` | auto | attention matmul dtype; auto = FP8 when the KV cache is FP8 (LMSYS: FP8 attention for prefill and decode) |
| `glueHid` / `glueExp` | 8 / 2 | memory-bound glue kernels: BF16 hidden-vector passes per token per layer (fused add+RMSNorm, quant, RoPE), plus passes per routed (token, expert) for FC2 output and combine |
| `hbmEff` | 0.85 | achievable HBM bandwidth fraction |
| `overlap` | 0.80 | share of hideable communication actually hidden by dual-batch overlap |
| `alphaUs` | 10 | base collective latency; per collective: all-reduce one-shot 0.8α, ring step 0.4α, all-to-all 1.5α, p2p 0.5α |
| `linkEff` | 0.70 | NVLink efficiency for all-to-all / p2p |
| `floorUs` | 30 | per-layer kernel-launch floor with CUDA graphs (doubled with DBO) |
| `overheadMs` | 0.3 | per-step scheduler / sampling overhead |
| `memUtil` | 0.90 | SGLang `--mem-fraction-static`: this share of HBM holds weights + DeepEP buffers + the KV pool (the long-context recipes use 0.70–0.72) |
| hardware | see `HARDWARE.gb200` | 186 GB usable HBM, 8 TB/s, dense 2.5 / 5 / 10 PF (BF16 / FP8 / FP4), NVLink 900 GB/s per direction — NVIDIA GB200 NVL72 datasheet (sparse ÷ 2); SFU exp rate 16 / clk / SM × 148 SMs × 2.06 GHz (clock implied by the BF16 peak) |

## Modeling choices worth knowing

- **Mean-field steady state.** Average decode context = context length + 512 (output length is fixed at 1,024 tokens; no prefix cache); no queueing or tails.
- **Layer averaging.** Dense-first layers and MoE layers are costed separately; CSA/HCA layers are averaged per layer. Pipeline stages get `ceil(layers/PP)` layers of the average mix.
- **Prefill.** Chunk size is per attention rank. Prompts shorter than the chunk are packed `floor(chunk / ISL)` per forward pass; they finish together, so TTFT covers the whole pack. Throughput assumes a full pipeline; latency adds `(PP−1)` chunk times. Attention FLOPs use 2·(d_qk + d_v)·pairs; MLA prefill runs un-absorbed.
- **Attention** (`attention.js`) takes the slowest of tensor-core math, one softmax exp per score on the SFUs, and KV streaming, plus a small fixed cost per call in decode. Decode MLA runs absorbed (KV read = latent, FLOPs ∝ 576 + 512 per head per token). Prefill MLA runs un-absorbed, where exp throughput rather than FP8 MMA limits it on GB200.
- **MoE.** Touched experts `1 − (1 − k/E)^b`; max-rank imbalance `μ + √(2μ ln EP)` plus a skew term that EPLB (+12.5% redundant slots) mostly removes; dispatch sends one copy per destination rank. All-to-all bytes are scaled by the max-rank imbalance, because the hottest receiver finishes last.
- **Speculative decoding (MTP).** Verify step processes γ+1 tokens per request (raising expert touch), drafts run γ layer-steps, `E = (1 − α^(γ+1)) / (1 − α)`.
- **DeepSeek-V4.** Dimensions, `index_topk` (Pro 1024, Flash 512) and the per-layer schedule come from HF `config.json` `compress_ratios`: Pro has 30 CSA + 31 HCA layers, Flash 21 CSA + 20 HCA + 2 layers with ratio 0, which we **assume** are sliding-window only. The FP8/BF16 KV entry layout is **assumed**. No anchor covers V4, so 200K–1M results are extrapolation. The four mHC residual streams only enter through the 4× wider pipeline hop; hash-routed layers are costed as ordinary MoE layers.

## KV offload (Grace memory and storage)

What deployments actually do, and what the tool models (`memory.js`, `attention.js`, `prefill.js`).

**Controls.**
- Prefix cache: Workload → Prefix cache.
- Decode KV placement: Parallelism card → KV cache, shown while editing a decode pool. The two options are *HBM only* and *Spill to LPDDR5X when HBM is full*.
- Assumptions: Grace LPDDR5X per GPU and host link efficiency; for sparse-attention models, the hot-buffer size and miss rate.
- Auto-tune picks the placement itself. It tunes once per option and keeps spill only if spill raises the best feasible throughput (goodput for a split rack) by more than 2% and the chosen batch actually overflows HBM.

| Mechanism | In production? | Tool behaviour |
|---|---|---|
| **Prefix / context cache** in host memory or storage | Yes, widely: SGLang HiCache (GPU → host → 3FS / Mooncake / NIXL), NVIDIA Dynamo KVBM (G1 GPU → G2 CPU → G3 disk → G4 remote), Mooncake, LMCache | The cached share of the prompt skips prefill compute; new tokens still attend over it. Its KV loads layer by layer at the tier's bandwidth, overlapped with compute: stage time = max(compute, load) + one layer's load. Prefill throughput counts the whole prompt. |
| **Spill, sparse (CSA) attention** | Yes: SGLang HiSparse (`--enable-hisparse`, DeepSeek-V3.2, GLM-5, DeepSeek-V4; decode instances; LRU hot buffer 2–6K slots) | **Minimal spill.** Once HBM is full, only as many requests as needed move their CSA compressed entries to LPDDR5X. Each such request keeps indexer keys, HCA, window and a hot buffer in HBM, and every other request stays fully in HBM. Offloaded requests swap in their top-k misses **serially** each step, because the misses depend on this step's indexer output. HiSparse itself mirrors *every* request in host memory: offloading only the overflow is our extension. It keeps HBM full when LPDDR5X is the binding capacity (V4-Pro at 624K on DEP72: 74 → 95 requests/rank, +12.5% auto-tuned throughput). |
| **Spill, dense attention** | No production stack streams dense-attention KV from host each step; research systems (FlexGen, InfiniGen) do | KV that doesn't fit in HBM moves to LPDDR5X and is **streamed every step**, prefetched and overlapped with the HBM part of attention. Auto-tune rejects it wherever it doesn't pay. |
| Weight offload | LMSYS GB200 Part II "Scaling Down by Offloading" (prefill weights, prefetched) | Not modeled yet: this is why the FP8-MoE 4-GPU prefill anchor is "needs feature". |

Hardware (`HARDWARE.gb200.host`): 480 GB LPDDR5X per Grace at ~512 GB/s, shared by 2 GPUs; NVLink-C2C 450 GB/s each way.
Public docs don't say whether C2C is per GPU or per superchip. Two GPUs streaming at once share the LPDDR5X either way, so the per-GPU peak is taken as min(C2C, LPDDR5X) / 2 = 225 GB/s,
times `hostBwEff` 0.8 (GH200 nvbandwidth measures host→device at 416 of 450 GB/s). `hostGB` = 200 GB per GPU after OS, runtime and pinned buffers.

Checks (`tests/offload.test.mjs`):
- HiSparse trend: V4-Flash at 200K/20K on 2 GPUs gains 2.77× throughput at the capacity-limited batch, against "up to 3×" reported on 2×B200.
- Dense spill lowers throughput for V3 at 128K even though it doubles concurrency.
- A prefix load from Grace memory hides behind compute, but a slow storage tier can become the bottleneck.

Not modeled: the working-set size and hit rate of the prefix cache (the user sets the hit rate; the tool reports how many prompts' KV fit in Grace memory), eviction dynamics, and HiSparse's miss rate as a function of buffer size.

## Not implemented yet (PLAN milestones M6+)

Aggregated / chunked-prefill-with-decode mode (removed from the product for now), weight offload to Grace, attention-FFN disaggregation, DWDP, discrete-event tails, HF `config.json` import,
calibration mode (fit efficiency knobs to user measurements), sensitivity tornado, other hardware, launch-command export. Uneven pipeline stage assignment is approximated.
