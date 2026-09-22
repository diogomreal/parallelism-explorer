export const DT_BYTES = { bf16: 2, fp8: 1, fp4: 0.5625 };

export const HARDWARE = {
  gb200: { name: 'GB200 NVL72', gpus: 72, hbmGB: 186, bw: 8e12, flops: { bf16: 2.5e15, fp8: 5e15, fp4: 10e15 }, link: 900e9, tdpW: 1200, nic: 50e9, provenance: 'vendor / verify' },
};

// DeepSeek-V4 numbers: HF transformers DeepseekV4Config + vLLM/NVIDIA model cards. Layer schedule, o_lora_rank (Pro) and
// the "layers of each attention type" split are ASSUMED (2 HCA bootstrap layers, then alternating CSA / HCA): verify against config.json.
const V4 = { layers: 0, vocab: 129280, attn: 'hybrid', kvHeads: 1, headDim: 512, ropeDim: 64, oLora: 1024, csaRatio: 4, hcaRatio: 128, window: 128, idxTopk: 512, idxHeads: 64, idxDim: 128, denseLayers: 0, denseInter: 0, moe: true, topK: 6, shared: 1, wDtype: 'fp8', expDtype: 'fp4', kvDtype: 'fp8',
  hcMult: 4, mtp: 1,
  notes: 'Simplified: hash-routed first 3 layers are treated as ordinary MoE layers; mHC only enters via the 4× wider pipeline hop.' };
export const MODELS = {
  deepseek_v4_pro: { ...V4, name: 'DeepSeek-V4-Pro', layers: 61, hidden: 7168, heads: 128, qLora: 1536, oGroups: 16, experts: 384, expertInter: 3072 },
  deepseek_v4_flash: { ...V4, name: 'DeepSeek-V4-Flash', layers: 43, hidden: 4096, heads: 64, qLora: 1024, oGroups: 8, experts: 256, expertInter: 2048 },
  deepseek_v3: { name: 'DeepSeek-V3 / R1', mtp: 1, layers: 61, denseLayers: 3, hidden: 7168, vocab: 129280, attn: 'mla', heads: 128, kvHeads: 128, headDim: 128, qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128, denseInter: 18432, moe: true, experts: 256, topK: 8, expertInter: 2048, shared: 1, wDtype: 'fp8', expDtype: 'fp4', kvDtype: 'fp8' },
  kimi_k2: { name: 'Kimi K2 (1T MoE)', layers: 61, denseLayers: 1, hidden: 7168, vocab: 163840, attn: 'mla', heads: 64, kvHeads: 64, headDim: 128, qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128, denseInter: 18432, moe: true, experts: 384, topK: 8, expertInter: 2048, shared: 1, wDtype: 'fp8', expDtype: 'fp8', kvDtype: 'fp8' },
  qwen3_235b: { name: 'Qwen3-235B-A22B', layers: 94, denseLayers: 0, hidden: 4096, vocab: 151936, attn: 'gqa', heads: 64, kvHeads: 4, headDim: 128, denseInter: 12288, moe: true, experts: 128, topK: 8, expertInter: 1536, shared: 0, wDtype: 'fp8', expDtype: 'fp8', kvDtype: 'fp8' },
  llama_405b: { name: 'Llama-3.1-405B (dense)', layers: 126, denseLayers: 126, hidden: 16384, vocab: 128256, attn: 'gqa', heads: 128, kvHeads: 8, headDim: 128, denseInter: 53248, moe: false, experts: 0, topK: 0, expertInter: 0, shared: 0, wDtype: 'fp8', expDtype: 'fp8', kvDtype: 'fp8' },
  hypo_10t: { name: 'Hypothetical 10T MoE (FP4 experts)', layers: 100, denseLayers: 0, hidden: 16384, vocab: 131072, attn: 'gqa', heads: 128, kvHeads: 8, headDim: 128, denseInter: 65536, moe: true, experts: 512, topK: 8, expertInter: 4096, shared: 0, wDtype: 'fp8', expDtype: 'fp4', kvDtype: 'fp8' },   // FP8 experts + redundant slots don't fit on GB200: try it
};

export const HYBRID_DEFAULT = { headDim: 512, kvHeads: 1, ropeDim: 64, qLora: 1024, oGroups: 8, oLora: 1024, csaRatio: 4, hcaRatio: 128, window: 128, idxTopk: 512, idxHeads: 64, idxDim: 128 };
// gpuHr: $/GPU-hour (blank/0 hides all cost figures). specGamma: MTP / draft tokens per step (0 = off); specAlpha: per-token acceptance.
export const WORKLOAD_DEFAULT = { isl: 2048, osl: 1024, minTokUser: 20, ttftMs: 2000, gpuHr: 3.5, batchPerRank: 384, chunk: 8192, specGamma: 0, specAlpha: 0.85 };
// Every constant below is an assumption (provenance 'assumed') until calibrated against measurements; see docs/validation.md.
export const ASSUMP_DEFAULT = {
  gemmEff: 0.6,      // dense GEMM MFU ceiling (fraction of dense peak at large M)
  moeEff: 0.55,      // grouped / expert GEMM efficiency ceiling (relative to peak)
  hbmEff: 0.85,      // achievable fraction of HBM bandwidth
  attnEff: 0.2,      // attention kernel MFU ceiling (fraction of peak); ramped by the same eta()/gemmHalfM row-count curve as gemmEff/moeEff, using batch size as the row count
  overlap: 0.8,      // fraction of the hideable comm actually hidden by dual-batch overlap
  alphaUs: 10,       // base collective latency (per-collective multipliers in comm.js)
  overheadMs: 0.3,   // per-step scheduler / sampling overhead
  memUtil: 0.9,      // gpu-memory-utilization cap
  floorUs: 30,       // per-layer kernel-launch floor with CUDA graphs
  linkEff: 0.7,      // achievable fraction of NVLink bandwidth for all-to-all / p2p
  gemmHalfM: 24,     // rows (batch/chunk tokens) at which GEMM efficiency reaches half its ceiling
  attnHalfM: 4,      // rows at which attention-kernel efficiency reaches half its ceiling: much smaller than gemmHalfM, since flash-decoding kernels get extra parallelism from KV-block splitting that a plain GEMM doesn't, so they saturate at a much smaller batch
};

