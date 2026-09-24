// Attention-core cost (decode and prefill). Per GPU, SI units. Three hardware limits, each with a published calibration point:
//   tensor-core math  FLOPs / (peak[math dtype] · eff)
//   softmax exp       one exp per (query row, head, attended key) on the SFUs (MUFU: 16 / clk / SM → H.sfu)
//   KV streaming      bytes / (HBM · kvBwEff)
// Sources (docs/validation.md): FlashInfer #4390 trtllm-gen MLA decode latency slope ≈ 0.78 × HBM; LMSYS GB200 / GB300 long-context FMHA
// (277 ms / 205 ms for a 128K causal layer, FP8) → FP8 MMA ≈ 0.8 of peak, exp at the MUFU rate, 26% of the shorter of the two not overlapped.

// Math dtype of the attention matmuls. 'auto' follows the KV cache (FP8 KV → FP8 Q·K / P·V, as in the TRT-LLM kernels SGLang uses on Blackwell).
export const attnMathDtype = (M, A) => (A.attnMath && A.attnMath !== 'auto' ? A.attnMath : M.kvDtype === 'bf16' ? 'bf16' : 'fp8');

// Decode: nReq requests per attention rank, q query tokens each (q > 1 = speculative verify), mean context ctx.
// off (optional) = KV offload for this batch: { hostFrac, hostBw, sparse: { fetchB } } (see memory.js).
//   spill:  hostFrac of the KV bytes stream from Grace memory each step. The data is known ahead, so it is prefetched layer by layer and
//           overlaps the HBM part: memory time = max(HBM part, host part).
//   sparse: each step's top-k misses swap in from Grace memory before attention can run: that fetch is serial (it depends on this step's indexer).
export function decodeAttention(nReq, q, ctx, M, H, A, P, off = null) {
  const D = P.D, kvm = D.kv(ctx, P.tpA), Hl = M.heads / P.tpA, ents = M.attn === 'hybrid' ? kvm.ent : ctx;
  const reread = q > 1 && !A.attnFoldQ ? q : 1;                                     // trtllm-gen re-reads KV per query row; fold-q kernels read once
  const bytes = nReq * kvm.read * reread / P.cp;
  const perTok = M.attn === 'mla' ? 2 * Hl * ctx * (M.kvLora + M.rope + M.kvLora)   // absorbed MLA: QK over 576 dims, PV over 512
    : M.attn === 'hybrid' ? 4 * Hl * kvm.ent * M.headDim + kvm.idxFl : 4 * Hl * ctx * M.headDim;
  const flops = perTok * nReq * q / P.cp, exps = Hl * ents * nReq * q / P.cp;
  const spill = off && !off.sparse ? off.hostFrac : 0;
  const tc = flops / (H.flops[attnMathDtype(M, A)] * A.attnEff), tm = bytes * (1 - spill) / (H.bw * A.kvBwEff), te = exps / H.sfu;
  const t = Math.max(tc, tm, te) + A.attnFixedUs * 1e-6;
  // host traffic and the part of it that is not hidden behind the attention kernel
  const hostBytes = off ? (off.sparse ? nReq * off.sparse.fetchB * q / P.cp : bytes * spill) : 0;
  const tHost = off && hostBytes > 0 ? hostBytes / off.hostBw : 0;
  const hostExposed = off?.sparse ? tHost : Math.max(0, tHost - t);
  return { flops, bytes: bytes * (1 - spill), exps, t, hostBytes, tHost, hostExposed, bound: tm >= tc && tm >= te ? 'memory' : 'compute' };
}

// Prefill: nReq requests per attention rank, each adding `size` new tokens on top of `prior` cached ones (causal).
export function prefillAttention(nReq, size, prior, M, H, A, P) {
  const D = P.D, kvm = D.kv(prior + size / 2, P.tpA), Hl = M.heads / P.tpA;
  const keys = M.attn === 'hybrid' ? kvm.ent : prior + size / 2;                     // mean attended keys per new token
  const perPair = M.attn === 'mla' ? 2 * (M.nope + M.rope + M.vDim) : 4 * M.headDim;  // prefill runs MLA un-absorbed (MHA-style)
  const flops = nReq * (Hl * size * keys * perPair + (M.attn === 'hybrid' ? kvm.idxFl * size : 0)) / P.cp;
  const exps = nReq * Hl * size * keys / P.cp;
  const tMma = flops / (H.flops[attnMathDtype(M, A)] * A.attnEffPrefill), tExp = exps / H.sfu;
  const tc = Math.max(tMma, tExp) + A.softmaxSerial * Math.min(tMma, tExp);
  const tm = nReq * kvm.read / P.cp / (H.bw * A.kvBwEff);
  return { flops, exps, tMma, tExp, t: Math.max(tc, tm) };
}
