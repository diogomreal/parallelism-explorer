export const DT_BYTES = { bf16: 2, fp8: 1, fp4: 0.5625 };

export const HARDWARE = {
  gb200: { name: 'GB200 NVL72', gpus: 72, hbmGB: 186, bw: 8e12, flops: { bf16: 2.5e15, fp8: 5e15, fp4: 10e15 }, link: 900e9, sfu: 16 * 148 * 2.06e9, tdpW: 1200,
    // Grace host tier: one Grace per 2 GPUs, 480 GB LPDDR5X at ~512 GB/s; NVLink-C2C 450 GB/s each way. Whether C2C is per GPU or per superchip is not
    // documented publicly; either way two GPUs streaming at once share the LPDDR5X, so per-GPU peak = min(C2C, LPDDR5X) / 2.
    host: { gbPerGrace: 480, gpusPerGrace: 2, lpddrBw: 512e9, c2c: 450e9 }, nic: 50e9, provenance: 'vendor / verify' },
};

// DeepSeek-V4 numbers: HF config.json (deepseek-ai/DeepSeek-V4-Pro, -Flash), checked 2026-09-23: dims, index_topk, o_groups / o_lora_rank and the
// per-layer compress_ratios. Still ASSUMED: a ratio-0 layer attends only its sliding window; the FP8 nope / BF16 rope KV entry layout.
const V4 = { layers: 0, vocab: 129280, attn: 'hybrid', kvHeads: 1, headDim: 512, ropeDim: 64, oLora: 1024, csaRatio: 4, hcaRatio: 128, window: 128, idxTopk: 512, idxHeads: 64, idxDim: 128, denseLayers: 0, denseInter: 0, moe: true, topK: 6, shared: 1, wDtype: 'fp8', expDtype: 'fp4', kvDtype: 'fp8',
  hcMult: 4, mtp: 1, swaLayers: 0,
  notes: 'Simplified: hash-routed first 3 layers are treated as ordinary MoE layers; mHC only enters via the 4× wider pipeline hop.' };
export const MODELS = {
  // Layer schedule from config.json compress_ratios (4 = CSA, 128 = HCA, 0 = sliding window only; the trailing entry is the MTP layer).
  deepseek_v4_pro: { ...V4, name: 'DeepSeek-V4-Pro', layers: 61, hidden: 7168, heads: 128, qLora: 1536, oGroups: 16, experts: 384, expertInter: 3072, idxTopk: 1024, csaLayers: 30, hcaLayers: 31 },
  deepseek_v4_flash: { ...V4, name: 'DeepSeek-V4-Flash', layers: 43, hidden: 4096, heads: 64, qLora: 1024, oGroups: 8, experts: 256, expertInter: 2048, idxTopk: 512, csaLayers: 21, hcaLayers: 20, swaLayers: 2 },
  deepseek_v3: { name: 'DeepSeek-V3 / R1', mtp: 1, layers: 61, denseLayers: 3, hidden: 7168, vocab: 129280, attn: 'mla', heads: 128, kvHeads: 128, headDim: 128, qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128, denseInter: 18432, moe: true, experts: 256, topK: 8, expertInter: 2048, shared: 1, wDtype: 'fp8', expDtype: 'fp4', kvDtype: 'fp8' },
  kimi_k2: { name: 'Kimi K2 (1T MoE)', layers: 61, denseLayers: 1, hidden: 7168, vocab: 163840, attn: 'mla', heads: 64, kvHeads: 64, headDim: 128, qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128, denseInter: 18432, moe: true, experts: 384, topK: 8, expertInter: 2048, shared: 1, wDtype: 'fp8', expDtype: 'fp8', kvDtype: 'fp8' },
  qwen3_235b: { name: 'Qwen3-235B-A22B', layers: 94, denseLayers: 0, hidden: 4096, vocab: 151936, attn: 'gqa', heads: 64, kvHeads: 4, headDim: 128, denseInter: 12288, moe: true, experts: 128, topK: 8, expertInter: 1536, shared: 0, wDtype: 'fp8', expDtype: 'fp8', kvDtype: 'fp8' },
  llama_405b: { name: 'Llama-3.1-405B (dense)', layers: 126, denseLayers: 126, hidden: 16384, vocab: 128256, attn: 'gqa', heads: 128, kvHeads: 8, headDim: 128, denseInter: 53248, moe: false, experts: 0, topK: 0, expertInter: 0, shared: 0, wDtype: 'fp8', expDtype: 'fp8', kvDtype: 'fp8' },
  hypo_10t: { name: 'Hypothetical 10T MoE (FP4 experts)', layers: 100, denseLayers: 0, hidden: 16384, vocab: 131072, attn: 'gqa', heads: 128, kvHeads: 8, headDim: 128, denseInter: 65536, moe: true, experts: 512, topK: 8, expertInter: 4096, shared: 0, wDtype: 'fp8', expDtype: 'fp4', kvDtype: 'fp8' },   // FP8 experts + redundant slots don't fit on GB200: try it
};

export const HYBRID_DEFAULT = { headDim: 512, kvHeads: 1, ropeDim: 64, qLora: 1024, oGroups: 8, oLora: 1024, csaRatio: 4, hcaRatio: 128, window: 128, idxTopk: 512, idxHeads: 64, idxDim: 128 };
// gpuHr: $/GPU-hour (blank/0 hides all cost figures). specGamma: MTP / draft tokens per step (0 = off); specAlpha: per-token acceptance.
// KV offload (see docs/validation.md "KV offload"):
//   prefixHit: share of each prompt whose KV is already cached (0 = no prefix cache); prefixTier: where hits are loaded from ('host' Grace memory | 'storage').
//   kvOffload: decode KV placement. 'none' = HBM only; 'spill' = once HBM is full, KV moves to Grace LPDDR5X. How depends on the attention:
//   dense attention streams the spilled KV every step; sparse (CSA) attention keeps the CSA entries in LPDDR5X and swaps in each step's top-k
//   misses into a per-request HBM hot buffer (SGLang HiSparse). Auto-tune picks the placement.
export const WORKLOAD_DEFAULT = { isl: 2048, osl: 1024, minTokUser: 20, ttftMs: 2000, gpuHr: 3.5, batchPerRank: 384, chunk: 8192, specGamma: 0, specAlpha: 0.85,
  prefixHit: 0, prefixTier: 'host', kvOffload: 'none' };
// Every constant below is an assumption (provenance 'assumed') until calibrated against measurements; see docs/validation.md.
export const ASSUMP_DEFAULT = {
  // gemmEff, moeEff, attnEff: fitted by scripts/calibrate.mjs (npm run calibrate) to the anchors in tests/anchors.data.mjs, with priors
  gemmEff: 0.36,     // dense GEMM ceiling (fraction of dense peak, before tile padding). Fit is 2σ below its 0.6 prior: it also absorbs 2K-prefill costs the model lacks
  moeEff: 0.40,      // grouped / expert GEMM ceiling
  hbmEff: 0.85,      // achievable fraction of HBM bandwidth
  attnEff: 0.24,     // DECODE attention tensor-core efficiency (fraction of the math dtype's peak). Fitted: no public FP8-math MLA decode benchmark (FlashMLA BF16-math ≈ 0.31 of BF16 peak on B200)
  attnEffPrefill: 0.8, // PREFILL attention tensor-core efficiency; with the SFU exp term below it reproduces LMSYS's GB200 / GB300 128K FMHA timings (277 / 205 ms)
  softmaxSerial: 0.26, // prefill: share of the shorter of (MMA, exp) time that does not overlap with the longer. From the same GB200 / GB300 pair
  kvBwEff: 0.78,     // fraction of HBM bandwidth an attention kernel reaches streaming KV (FlashInfer #4390 trtllm-gen MLA decode latency slope on B300)
  attnFixedUs: 5,    // fixed cost per decode-attention call (split-KV combine, scheduling)
  attnFoldQ: false,  // speculative verify: false = KV re-read per query row (trtllm-gen), true = query rows folded into one pass (cute-dsl fold_sq)
  attnMath: 'auto',  // attention matmul dtype: 'auto' follows the KV cache (FP8 KV → FP8 math), or 'bf16' / 'fp8'
  overlap: 0.8,      // fraction of the hideable comm actually hidden by dual-batch overlap
  alphaUs: 10,       // base collective latency (per-collective multipliers in comm.js)
  overheadMs: 0.3,   // per-step scheduler / sampling overhead
  memUtil: 0.9,      // gpu-memory-utilization cap (≈ SGLang --mem-fraction-static: the long-context LMSYS recipes use 0.70–0.72)
  floorUs: 30,       // per-layer kernel-launch floor with CUDA graphs
  linkEff: 0.7,      // achievable fraction of NVLink bandwidth for all-to-all / p2p
  glueHid: 8,        // memory-bound "glue" passes over the BF16 hidden vector per token per layer (fused add+RMSNorm ×2, activation quant, RoPE, residual): ~8 with fused norms
  glueExp: 2,        // extra hidden-vector passes per routed (token, expert) assignment: FC2 output write + combine/weighted-sum read
  gemmTileM: 64,
  hostGB: 200,       // Grace memory per GPU available for KV (480 GB per Grace, 2 GPUs, minus OS, runtime and pinned staging)
  hostBwEff: 0.8,    // achieved fraction of the per-GPU host bandwidth (GH200 nvbandwidth: host→device 416 of 450 GB/s, device→host 295)
  storageGBs: 25,    // per-GPU bandwidth of the storage tier for prefix-cache hits (NIC / NVMe / 3FS-class), GB/s
  sparseBuf: 4096,   // sparse-attention spill: HBM hot-buffer entries per request per CSA layer (HiSparse device_buffer_size: 2048–6144)
  sparseMiss: 0.1,   // sparse-attention spill: share of each step's top-k not already in the hot buffer (assumed; FreeKV: > 80% step-to-step overlap)     // GEMM rows are padded to whole tiles of this many rows (tile quantization on the token dimension)
};

