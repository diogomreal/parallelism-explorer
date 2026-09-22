// Chunked-prefill model (PLAN §6.7): per-chunk roofline over the op list with M = chunk tokens, causal attention over the prior context,
// steady-state pipelined throughput, PP fill latency, KV hand-off.
import { collective, distinctDest } from './comm.js';
import { moeLoad } from './decode.js';

const eta = (rows, ceil, A) => ceil * rows / (rows + A.gemmHalfM);
const etaAttn = (rows, ceil, A) => ceil * rows / (rows + A.attnHalfM);   // attention saturates much faster than a GEMM (see attnHalfM)

// One layer of one kind for a chunk of `size` tokens per attention rank whose request already has `prior` tokens in the KV cache.
function layerChunk(kind, size, prior, M, H, A, P) {
  const D = P.D, bw = H.bw * A.hbmEff, tG = size / P.cp;                        // CP shards the chunk's tokens across its ranks
  const peakW = H.flops[M.wDtype], peakAttn = H.flops[M.kvDtype === 'bf16' ? 'bf16' : 'fp8'] * etaAttn(tG, A.attnEff, A), Hl = M.heads / P.tpA;
  const gemm = (params, rows, peak, ceil) => Math.max(2 * rows * params / (peak * eta(rows, ceil, A)), params * (peak === peakW ? D.bW : D.bE) / bw);
  const tProj = gemm(D.attnP / P.tpA, tG, peakW, A.gemmEff);
  const kvm = D.kv(prior + size / 2, P.tpA);
  const fl = M.attn === 'mla' ? 2 * (M.nope + M.rope + M.vDim) * Hl * size * (prior + size / 2)      // prefill runs MLA un-absorbed (MHA-style)
    : M.attn === 'hybrid' ? (4 * Hl * M.headDim * kvm.ent + kvm.idxFl) * size
    : 4 * Hl * M.headDim * size * (prior + size / 2);
  const tAttn = Math.max(fl / P.cp / peakAttn, kvm.read / P.cp / bw);
  const tDense = gemm((kind === 'moe' ? D.sharedP : D.denseP) / P.tpA, tG, peakW, A.gemmEff);
  let tExp = 0, tm = 0, load = null;
  if (kind === 'moe') {
    const b = size * P.dpA; load = moeLoad(M, P, b);
    const Me = Math.max(1, load.tokLoc / Math.max(load.touched, 1e-9));
    tExp = Math.max(2 * load.tokLoc * D.expertP / P.tpM / (H.flops[M.expDtype] * eta(Me, A.moeEff, A)), load.touched * D.expertP / P.tpM * D.bE / bw);
    if (P.ep > 1) {
      const perGpu = b / (P.dpA * P.tpA * P.cp), dest = distinctDest(P.ep, M.topK);
      tm += collective('a2a', perGpu * dest * M.hidden * D.dispatchB * P.tpM, P.ep, H, A) + collective('a2a', perGpu * dest * M.hidden * D.combineB * P.tpM, P.ep, H, A);
    }
    if (P.tpM > 1) tm += collective('rsag', load.tokLoc * M.hidden * 2, P.tpM, H, A);
  }
  if (P.tpA > 1) tm += 2 * collective('ar', tG * M.hidden * 2, P.tpA, H, A);
  if (P.cp > 1) { const ring = (P.cp - 1) * collective('p2p', D.kv(size, P.tpA).store / P.cp, 2, H, A); tm += Math.max(0, ring - tAttn); }   // ring KV exchange hides behind attention blocks
  const tc = tProj + tAttn + tDense + tExp, hidden = P.overlap ? A.overlap * Math.min(tm, tc) : 0;
  return { tProj, tAttn, tDense, tExp, tm, hidden, t: tc + tm - hidden + A.floorUs * 1e-6, load };
}

export function evalPrefill(M, H, W, A, P, gpus, recvSenders) {
  const D = P.D, L = M.layers, Ls = Math.ceil(L / P.pp);
  const isl = Math.max(W.isl, 1);                                              // no prefix cache: the whole prompt is prefilled
  const C = Math.max(1, Math.min(W.chunk, isl));                               // chunk tokens per attention rank per iteration
  const kinds = [];
  if (D.moeLayers > 0) kinds.push(['moe', D.moeLayers / L]);
  if (D.denseLayers > 0) kinds.push(['dense', D.denseLayers / L]);
  const acc = { tProj: 0, tAttn: 0, tDense: 0, tExp: 0, tm: 0, hidden: 0, t: 0 };
  let nCh = 0, done = 0, tMax = 0;
  while (done < isl - 1e-9) {
    const size = Math.min(C, isl - done);
    let ts = 0;
    for (const [k, w] of kinds) { const r = layerChunk(k, size, done, M, H, A, P); for (const key in acc) acc[key] += r[key] * w * Ls; ts += r.t * w * Ls; }
    tMax = Math.max(tMax, ts); done += size; nCh++;
  }
  const T = acc.t;                                                             // stage-seconds to prefill one request per attention rank
  const hop = P.pp > 1 ? collective('p2p', C * M.hidden * 2 * (M.hcMult || 1) / (P.tpA * P.cp), 2, H, A) : 0;
  const tAvg = T / nCh, fill = (P.pp - 1) * tAvg + P.pp * hop;
  const tPrefill = T + fill;                                                   // latency of one request through the pipeline
  const bubble = P.pp > 1 ? (P.pp - 1) / (nCh + P.pp - 1) : 0;
  const perReplica = P.dpA / T;                                                // steady state: pipeline full, dpA requests complete every T
  const reqPerS = perReplica * P.replicas, thr = reqPerS * isl / gpus;
  const kvBytes = D.kv(W.isl).store * L;
  const senders = Math.max(1, Math.min(P.tpA * P.cp, recvSenders || P.tpA * P.cp)) * Math.max(1, P.pp);
  const kvXfer = kvBytes / (senders * H.link * A.linkEff) + 0.5 * A.alphaUs * 1e-6;   // in-rack NVLink hand-off, shards sent in parallel
  const ms = 1e3, scale = ms;
  const segs = [
    { key: 'attnProj', label: 'Linear projections + dense', ms: (acc.tProj + acc.tDense) * scale },
    { key: 'moe', label: 'Routed experts', ms: acc.tExp * scale },
    { key: 'kv', label: 'Attention (quadratic)', ms: acc.tAttn * scale },
    { key: 'comm', label: 'Communication (exposed)', ms: (acc.tm - acc.hidden) * scale },
    { key: 'hidden', label: 'Communication (hidden by overlap)', ms: acc.hidden * scale, hidden: true },
    { key: 'pp', label: 'PP fill / drain', ms: fill * scale },
    { key: 'over', label: 'Kernel floors', ms: A.floorUs * 1e-6 * Ls * nCh * scale },
  ].filter((s) => s.ms > 1e-6);
  return { tPrefill, tChunk: tAvg, nCh, C, bubble, thr, reqPerS, kvBytes, kvXfer, segs, isl, ttft: tPrefill + kvXfer, tLin: acc.tProj + acc.tDense, tExp: acc.tExp, tAttn: acc.tAttn, commT: acc.tm, hidden: acc.hidden };
}
