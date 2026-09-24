// Published measurements the model is checked against. One registry, read by tests/anchors.test.mjs (and later the calibration fit).
// Every entry quotes its source; conditions the source does not state are listed in `assumed`.
// status: 'calibration' (used to fit knobs, must sit in the band), 'validation' (not fitted, must sit in the band),
//         'gap' (known miss, reported as a test TODO until the model is fixed), 'needs-config' (setup too underspecified to evaluate).
import { C, H, W0, mk, cfg } from './util.mjs';

const LMSYS2 = 'https://www.lmsys.org/blog/2025-09-25-gb200-part-2/';
const LMSYS_LONG = 'https://www.lmsys.org/blog/2026-02-19-gb300-longctx/';
const V3_LOWP = {};                                           // FP8 attention / KV + NVFP4 MoE (preset default)
const V3_HIGHP = { kvDtype: 'bf16', expDtype: 'fp8' };        // BF16 attention / KV + FP8 MoE

// Decode on a fixed layout. batch 'fill' = the largest batch per rank that fits (the sources size batches "to make KV cache roughly full").
function decode(A, { model = {}, gpus, layout, isl, osl, batch }) {
  const M = mk('deepseek_v3', model), P = C.normalize(M, gpus, cfg(layout)), W = { ...W0, isl, osl };
  const cap = C.memoryPerGpu(M, H, W, A, P, P.D).perRankMax;
  const ba = batch === 'fill' ? cap : batch, r = C.evalDecode(M, H, W, A, P, ba);
  return { value: r.perGpu, detail: `batch ${ba}/rank (fits ${cap})${r.oom ? ' OOM' : ''}` };
}
// Long-context disaggregated deployment: nPre prefill instances (PP4 × TP1, one unchunked pass) + one DEP decode instance; output tok/s over all GPUs.
function longE2E(A, { nPre = 3, dGpus = 8, maxBatch = 32, isl = 128000, osl = 8000 } = {}) {
  const M = mk('deepseek_v3'), W = { ...W0, isl, osl, chunk: isl }, Am = { ...A, memUtil: 0.72 };
  const Pd = C.normalize(M, dGpus, cfg({ ep: dGpus, eplb: true })), cap = C.memoryPerGpu(M, H, W, Am, Pd, Pd.D).perRankMax, ba = Math.min(cap, maxBatch);
  const d = C.evalDecode(M, H, W, Am, Pd, ba), pre = C.evalPrefill(M, H, W, Am, C.normalize(M, 4, cfg({ pp: 4 })), 4);
  const rp = pre.reqPerS * nPre, rd = d.total / osl, gpus = nPre * 4 + dGpus;
  return { value: Math.min(rp, rd) * osl / gpus, detail: `decode batch ${ba}/rank, ${d.perGpu.toFixed(0)} tok/s/decode GPU, ${d.tokUser.toFixed(1)} tok/s/user; ${rp < rd ? 'prefill' : 'decode'}-limited` };
}
function capacity(A, { gpus, layout, isl, osl }) {
  const M = mk('deepseek_v3'), P = C.normalize(M, gpus, cfg(layout));
  return { value: C.memoryPerGpu(M, H, { ...W0, isl, osl }, A, P, P.D).perRankMax, detail: 'max requests per rank' };
}

export const ANCHORS = [
  { id: 'lmsys2-decode-lowp', status: 'calibration', src: LMSYS2, unit: 'output tok/s/GPU', measured: 13386,
    quote: '26,156 input and 13,386 output tokens per second per NVIDIA Blackwell GPU … 2000-token input sequences; for decode, we use 48 ranks; batch size … 1408 for 2k ISL',
    assumed: 'OSL 100 (sets mean context), EPLB + DBO on, no MTP',
    run: (A) => decode(A, { model: V3_LOWP, gpus: 48, layout: { ep: 48, eplb: true, overlap: true }, isl: 2000, osl: 100, batch: 1408 }) },
  { id: 'lmsys2-prefill-lowp', status: 'calibration', src: LMSYS2, unit: 'input tok/s/GPU', measured: 26156,
    quote: 'for prefill, we use 4 ranks per instance',
    assumed: 'layout not stated: DP-attn 4 + EP4 with two-batch overlap and 16K-token chunks (SGLang PD prefill defaults of the time). Range over plausible setups: 1.13–1.39× at the step-1 knobs',
    run: (A) => { const M = mk('deepseek_v3'), r = C.evalPrefill(M, H, { ...W0, isl: 2000, osl: 100, chunk: 16384 }, A, C.normalize(M, 4, cfg({ ep: 4, overlap: true, eplb: true })), 4); return { value: r.thr, detail: `DPa4·EP4, ${r.pack} prompts per pass` }; } },
  { id: 'lmsys2-decode-highp', status: 'validation', src: LMSYS2, unit: 'output tok/s/GPU', measured: 9087,
    quote: 'traditional BF16 attention and FP8 MoE, SGLang still achieves 18,471 input and 9,087 output tokens per second',
    assumed: 'same layout as the low-precision run; batch fills the (halved) KV budget',
    run: (A) => decode(A, { model: V3_HIGHP, gpus: 48, layout: { ep: 48, eplb: true, overlap: true }, isl: 2000, osl: 100, batch: 'fill' }) },
  { id: 'lmsys2-prefill-highp', status: 'needs-config', src: LMSYS2, unit: 'input tok/s/GPU', measured: 18471,
    quote: 'traditional BF16 attention and FP8 MoE, SGLang still achieves 18,471 input … tokens per second',
    assumed: 'best modeled 4-GPU layout',
    why: 'FP8 experts need ≈ 163 GB/GPU of weights on a 4-GPU instance: LMSYS offloads weights to Grace memory ("Scaling Down by Offloading"), which the model does not implement',
    run: null },
  // Long context (128K/8K). Setups from the LMSYS reproduction recipes (github.com/YAMY1234/srt-slurm, branch gb300_blog, recipes/gb300-128k8k-blog).
  // Their "Output TPS/GPU" divides output tokens/s by ALL GPUs, prefill + decode (analysis/srtlog/run_loader.py: tps / total_gpus).
  { id: 'lmsys-long-e2e', status: 'calibration', src: LMSYS_LONG, unit: 'output tok/s/GPU (all GPUs)', measured: 147.9,
    quote: 'GB200 without MTP: 147.9 TPS/GPU; recipe gb200-maxthroughput-ctx3_ctx_pp4_gen1_dep8_batch32_eplb0_mtp0: 3 prefill workers (PP4, TP1) + 1 decode worker (DP-attn 8, EP8), mem-fraction-static 0.72, 32 redundant experts, no two-batch overlap',
    assumed: 'DeepSeek-R1 NVFP4 ≈ V3 preset; decode batch = min(32, KV capacity); steady state rate-matched between the pools',
    run: (A) => longE2E(A) },
  { id: 'lmsys-long-ttft', status: 'validation', src: LMSYS_LONG, unit: 's', measured: 18.6,
    quote: 'GB200 without dynamic chunking: 18.6 s TTFT for 128K; recipe ctx1_ctx_pp4_gen1_tp4_batch1: one prefill worker, PP4 × TP1, chunked-prefill-size -1',
    assumed: 'ISL 128,000 in one pass per stage; TTFT ≈ prefill latency + KV hand-off',
    run: (A) => { const M = mk('deepseek_v3'), r = C.evalPrefill(M, H, { ...W0, isl: 128000, osl: 8000, chunk: 128000 }, A, C.normalize(M, 4, cfg({ pp: 4 })), 4); return { value: r.ttft, detail: `${r.nCh} chunk × PP4` }; } },
  { id: 'lmsys-long-fmha', status: 'validation', src: 'https://github.com/sgl-project/sglang/issues/18703', unit: 's per layer', measured: 0.277,
    quote: 'GB200 FMHA latency 277 ms (GB300: 205 ms, 1.35× from 2× SFU throughput)',
    assumed: 'OUR READING: one layer of full 128K causal MLA prefill attention, 128 heads on one GPU (TP1), FP8 math. The source does not state the shape',
    run: (A) => { const M = mk('deepseek_v3'), a = C.prefillAttention(1, 128000, 0, M, H, A, C.normalize(M, 1, cfg())); return { value: a.t, detail: `MMA ${(a.tMma * 1e3).toFixed(0)} ms, exp ${(a.tExp * 1e3).toFixed(0)} ms` }; } },
  { id: 'lmsys-long-capacity', status: 'validation', src: LMSYS_LONG, unit: 'requests/GPU', measured: 20, tol: 0.25,
    quote: '320 concurrent requests at DEP16 on GB200 (= 20 per GPU; the same post also says "~24 req/GPU"); DEP16 recipes use mem-fraction-static 0.70',
    assumed: '128K/8K, FP8 KV, no Grace offload',
    run: (A) => capacity({ ...A, memUtil: 0.7 }, { gpus: 16, layout: { ep: 16, eplb: true }, isl: 128000, osl: 8000 }) },
];
