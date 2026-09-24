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
  const kvm = D.kv(ctx + 8, P.tpA), per = (x) => (x * Ls) / P.cp;  // + half a 16-token page of allocation waste; per request, this stage, this rank
  const kvReq = per(kvm.store);
  // KV placement (docs/validation.md "KV offload"). kvDev / kvHost: bytes per request in HBM / in Grace memory at full placement.
  // Both offload mechanisms engage only once the batch no longer fits in HBM (hbmMax).
  const mode = offloadMode(M, W), hostCap = mode === 'none' ? 0 : A.hostGB * 1e9;
  const hbmMax = Math.floor(free / kvReq / m);
  let kvDev = kvReq, kvHost = 0;
  if (mode === 'sparse') {                                          // CSA entries live in host; HBM keeps a hot buffer of sparseBuf entries per CSA layer
    const bufB = (D.csaL * Math.min(A.sparseBuf, (ctx + 8) / M.csaRatio) * D.entryB) / M.layers;
    kvHost = per(kvm.csaMain); kvDev = kvReq - kvHost + per(bufB);
  }
  // Minimal spill: move only what does not fit. Dense attention spills a byte share of every request's KV; sparse attention moves whole requests'
  // CSA entries to LPDDR5X (each saves kvReq − kvDev of HBM) and keeps every other request fully in HBM. HiSparse itself mirrors every request
  // in host memory; offloading only the overflow is our extension (docs/validation.md "KV offload").
  const save = kvReq - kvDev, hostReqMax = kvHost > 0 ? Math.floor(hostCap / kvHost) : 0;
  const offloaded = (ba) => (mode !== 'sparse' ? 0 : Math.min(ba * m, Math.max(0, Math.ceil(Math.max(0, ba * m * kvReq - free) / save - 1e-9))));   // requests (× microbatches)
  const perRankMax = mode === 'spill' ? Math.floor((free + hostCap) / kvReq / m)
    : mode === 'sparse' ? Math.max(hbmMax, Math.floor(Math.min(free / kvDev, (free + hostReqMax * save) / kvReq) / m)) : hbmMax;
  // Share of the batch's KV bytes that sit in host memory at batch ba, and the share of requests that have to swap in (sparse).
  const hostFrac = (ba) => (mode === 'spill' ? Math.max(0, ba * m * kvReq - free) / Math.max(ba * m * kvReq, 1) : mode === 'sparse' ? offloaded(ba) * kvHost / Math.max(ba * m * kvReq, 1) : 0);
  const offloadedShare = (ba) => offloaded(ba) / Math.max(ba * m, 1);
  const kvUsedFor = (ba) => (mode === 'sparse' ? ba * m * kvReq - offloaded(ba) * save : ba * m * kvReq * (1 - hostFrac(ba)));
  const hostUsedFor = (ba) => (mode === 'sparse' ? offloaded(ba) * kvHost : ba * m * kvReq * hostFrac(ba));
  return { hbm, wAttn, wExp, wEmb, reserved, act, free, kvReq, kvDev, kvHost, mode, hostCap, hbmMax, hostFrac, kvUsedFor, hostUsedFor,
    offloadedFor: (ba) => offloaded(ba) / m, offloadedShare,
    oomAt: (ba) => kvUsedFor(ba) > free + 1 || hostUsedFor(ba) > hostCap + 1, perRankMax, bMaxRep: perRankMax * P.dpA * m, ctx };
}

// Spilling uses the mechanism the attention allows: HiSparse-style top-k swap-in for sparse (CSA) attention, per-step streaming for dense attention.
export const offloadMode = (M, W) => (W.kvOffload === 'spill' || W.kvOffload === 'sparse' ? (M.attn === 'hybrid' ? 'sparse' : 'spill') : 'none');
// Per-GPU bandwidth to Grace memory: the C2C link and the LPDDR5X are shared by the GPUs of one Grace.
export const hostBw = (H, A) => (Math.min(H.host.c2c, H.host.lpddrBw) / H.host.gpusPerGrace) * A.hostBwEff;
