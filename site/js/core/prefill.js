// Chunked-prefill model (PLAN §6.7): per-chunk roofline over the op list with M = chunk tokens, causal attention over the prior context,
// steady-state pipelined throughput, PP fill latency, KV hand-off.
import { collective, distinctDest } from './comm.js';
import { moeLoad } from './decode.js';
import { gemmEta as eta } from './gemm.js';
import { prefillAttention } from './attention.js';
import { hostBw } from './memory.js';

// One layer of one kind for a forward pass per attention rank of `nReq` requests, each contributing `size` new tokens on top of `prior`
// tokens already in its KV cache (nReq > 1 only when several short prompts are packed into one chunk, so prior is then 0).
// Attention is costed by prefillAttention (tensor-core math + softmax exp on the SFUs + KV streaming), with its own efficiency knob.
function layerChunk(kind, size, prior, M, H, A, P, nReq = 1) {
  const D = P.D, bw = H.bw * A.hbmEff, tG = nReq * size / P.cp;                 // CP shards the chunk's tokens across its ranks
  const peakW = H.flops[M.wDtype];
  const gemm = (params, rows, peak, ceil) => Math.max(2 * rows * params / (peak * eta(rows, ceil, A)), params * (peak === peakW ? D.bW : D.bE) / bw);
  const tProj = gemm(D.attnP / P.tpA, tG, peakW, A.gemmEff);
  const tAttn = prefillAttention(nReq, size, prior, M, H, A, P).t;
  const tDense = gemm((kind === 'moe' ? D.sharedP : D.denseP) / P.tpA, tG, peakW, A.gemmEff);
  let tExp = 0, tm = 0, load = null;
  if (kind === 'moe') {
    const b = nReq * size * P.dpA; load = moeLoad(M, P, b);
    const Me = Math.max(1, load.tokLoc / Math.max(load.touched, 1e-9));
    tExp = Math.max(2 * load.tokLoc * D.expertP / P.tpM / (H.flops[M.expDtype] * eta(Me, A.moeEff, A)), load.touched * D.expertP / P.tpM * D.bE / bw);
    if (P.ep > 1) {
      const perGpu = b / (P.dpA * P.tpA * P.cp) * load.imb, dest = distinctDest(P.ep, M.topK);   // the hottest receiver sets the all-to-all time
      tm += collective('a2a', perGpu * dest * M.hidden * D.dispatchB * P.tpM, P.ep, H, A) + collective('a2a', perGpu * dest * M.hidden * D.combineB * P.tpM, P.ep, H, A);
    }
    if (P.tpM > 1) tm += collective('rsag', load.tokLoc * M.hidden * 2, P.tpM, H, A);
  }
  if (P.tpA > 1) tm += 2 * collective('ar', tG * M.hidden * 2, P.tpA, H, A);
  if (P.cp > 1) { const ring = (P.cp - 1) * collective('p2p', nReq * D.kv(size, P.tpA).store / P.cp, 2, H, A); tm += Math.max(0, ring - tAttn); }   // ring KV exchange hides behind attention blocks
  const tGlue = (tG * A.glueHid + (load ? load.tokLoc * A.glueExp : 0)) * M.hidden * 2 / bw;   // memory-bound norms / quant / residual / combine
  const tc = tProj + tAttn + tDense + tExp + tGlue, hidden = P.overlap ? A.overlap * Math.min(tm, tc) : 0;
  return { tProj, tAttn, tDense, tExp, tGlue, tm, hidden, t: tc + tm - hidden + A.floorUs * 1e-6, load };
}

export function evalPrefill(M, H, W, A, P, gpus, recvSenders) {
  const D = P.D, L = M.layers, Ls = Math.ceil(L / P.pp);
  const isl = Math.max(W.isl, 1);
  // Prefix cache: the first `cached` tokens already have KV in a cache tier; only the rest is computed (attending over the whole prefix).
  const hit = Math.min(Math.max(W.prefixHit || 0, 0), 0.99), cached = Math.floor(hit * isl), fresh = isl - cached;
  const C = Math.max(1, Math.min(W.chunk, fresh));                             // tokens of one request per attention rank per iteration
  const pack = Math.max(1, Math.floor(W.chunk / fresh));                       // prompts shorter than the chunk are packed into one forward pass
  const kinds = [];
  if (D.moeLayers > 0) kinds.push(['moe', D.moeLayers / L]);
  if (D.denseLayers > 0) kinds.push(['dense', D.denseLayers / L]);
  const acc = { tProj: 0, tAttn: 0, tDense: 0, tExp: 0, tGlue: 0, tm: 0, hidden: 0, t: 0 };
  let nCh = 0, done = cached, tMax = 0;
  while (done < isl - 1e-9) {
    const size = Math.min(C, isl - done);
    let ts = 0;
    for (const [k, w] of kinds) { const r = layerChunk(k, size, done, M, H, A, P, pack); for (const key in acc) acc[key] += r[key] * w * Ls; ts += r.t * w * Ls; }
    tMax = Math.max(tMax, ts); done += size; nCh++;
  }
  const Tc = acc.t;                                                            // stage-seconds of compute to prefill `pack` requests per attention rank
  // Cached prefix KV loads into HBM layer by layer while earlier layers compute (HiCache / KVBM style); only the first layer's load is exposed
  // for sure, the rest hides unless the load is slower than the compute.
  const tierBw = W.prefixTier === 'storage' ? A.storageGBs * 1e9 : hostBw(H, A);
  const loadB = cached > 0 ? pack * D.kv(cached, P.tpA).store * Ls / P.cp : 0, tLoad = loadB / tierBw;
  const T = cached > 0 ? Math.max(Tc, tLoad) + tLoad / Ls : Tc;
  const hop = P.pp > 1 ? collective('p2p', pack * C * M.hidden * 2 * (M.hcMult || 1) / (P.tpA * P.cp), 2, H, A) : 0;
  const tAvg = T / nCh, fill = (P.pp - 1) * tAvg + P.pp * hop;
  const tPrefill = T + fill;                                                   // latency of one request through the pipeline
  const bubble = P.pp > 1 ? (P.pp - 1) / (nCh + P.pp - 1) : 0;
  const perReplica = P.dpA * pack / T;                                         // steady state: pipeline full, dpA · pack requests complete every T
  const reqPerS = perReplica * P.replicas, thr = reqPerS * isl / gpus;
  const kvBytes = D.kv(W.isl).store * L;
  const senders = Math.max(1, Math.min(P.tpA * P.cp, recvSenders || P.tpA * P.cp)) * Math.max(1, P.pp);
  const kvXfer = pack * kvBytes / (senders * H.link * A.linkEff) + 0.5 * A.alphaUs * 1e-6;   // in-rack NVLink hand-off of the packed requests, shards sent in parallel
  const ms = 1e3, scale = ms;
  const segs = [
    { key: 'attnProj', label: 'Linear projections + dense', ms: (acc.tProj + acc.tDense) * scale },
    { key: 'moe', label: 'Routed experts', ms: acc.tExp * scale },
    { key: 'kv', label: 'Attention (quadratic)', ms: acc.tAttn * scale },
    { key: 'glue', label: 'Glue kernels (norm / quant / residual)', ms: acc.tGlue * scale },
    { key: 'host', label: `Prefix KV load from ${W.prefixTier === 'storage' ? 'storage' : 'Grace memory'} (exposed)`, ms: (T - Tc) * scale },
    { key: 'comm', label: 'Communication (exposed)', ms: (acc.tm - acc.hidden) * scale },
    { key: 'hidden', label: 'Communication (hidden by overlap)', ms: acc.hidden * scale, hidden: true },
    { key: 'pp', label: 'PP fill / drain', ms: fill * scale },
    { key: 'over', label: 'Kernel floors', ms: A.floorUs * 1e-6 * Ls * nCh * scale },
  ].filter((s) => s.ms > 1e-6);
  const prefix = { hit, cached, tLoad, tier: W.prefixTier === 'storage' ? 'storage' : 'host', loadGB: loadB / 1e9,
    ctxPerHostGpu: Math.floor(A.hostGB * 1e9 / Math.max(D.kv(isl, P.tpA).store * Ls / P.cp, 1)) };   // prompts' KV that fit in one GPU's share of Grace memory
  return { tPrefill, tChunk: tAvg, nCh, C, pack, bubble, thr, prefix, reqPerS, kvBytes, kvXfer, segs, isl, ttft: tPrefill + kvXfer, tLin: acc.tProj + acc.tDense, tExp: acc.tExp, tAttn: acc.tAttn, commT: acc.tm, hidden: acc.hidden };
}
