// Per-GPU memory model (PLAN §6.2): weights, KV per request, activations, reserved → free HBM → max concurrent requests.

export function memoryPerGpu(M, H, W, A, P, D) {
  const L = M.layers, Ls = Math.ceil(L / P.pp);                    // worst-case pipeline stage
  const ctx = W.isl + W.osl / 2;                                   // mean decode context: the full prompt (cached or not) plus half the output
  const wL = (n) => (n * Ls) / L;                                  // scale a whole-model layer count to one stage
  // attention + dense FFN + shared experts + router: sharded by TP, replicated across attention DP and CP ranks
  const perLayerDense = D.attnP * L + D.denseP * D.denseLayers + (D.sharedP + D.routerP) * D.moeLayers;
  const mtpLayer = W.specGamma > 0 && M.mtp ? D.attnP + D.sharedP + D.routerP : 0;      // draft (MTP) layer lives on the last stage
  const wAttn = ((perLayerDense * Ls) / L + mtpLayer) / P.tpA * D.bW;
  // routed experts: (slots / EP) per GPU, each sharded TP_moe ways; slots include redundant experts
  const wExpLayers = M.moe ? (P.slots / P.ep) * D.expertP / P.tpM * D.bE : 0;
  const wExp = wL(D.moeLayers) * wExpLayers + (mtpLayer && M.moe ? wExpLayers : 0);
  const wEmb = (P.pp === 1 ? 2 : 1) * (M.vocab * M.hidden * D.bW) / P.g;   // vocab-parallel over the stage's GPUs
  const hbm = H.hbmGB * 1e9;
  // SGLang semantics: --mem-fraction-static (memUtil) of HBM holds weights + KV pool; the rest is left for activations, CUDA graphs, NCCL.
  const reserved = (1 - A.memUtil) * hbm;
  // DeepEP dispatch/combine buffers come out of the static pool: ~ EP × 128 tokens × d × (dispatch + combine bytes), 2× buffered
  const disp = M.moe && P.ep > 1 ? 2 * P.ep * 128 * M.hidden * (D.dispatchB + D.combineB) : 0;
  const act = disp;
  const free = Math.max(0, hbm - wAttn - wExp - wEmb - reserved - act);
  const m = Math.max(P.microbatches, 1);
  const kvReq = (D.kv(ctx + 8, P.tpA).store * Ls) / P.cp;          // + half a 16-token page of allocation waste
  const perRankMax = Math.floor(free / kvReq / m);
  return { hbm, wAttn, wExp, wEmb, reserved, act, free, kvReq, kvUsedFor: (ba) => ba * m * kvReq, perRankMax, bMaxRep: perRankMax * P.dpA * m, ctx };
}
