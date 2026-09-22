// Decode step model (PLAN §6.1, §6.3, §6.4, §6.6): per-layer op list → roofline times → overlap → pipeline → TPOT.
// Everything is per GPU, SI units. `ba` = requests per attention rank in one microbatch.
import { collective, distinctDest } from './comm.js';
import { memoryPerGpu } from './memory.js';

const eta = (rows, ceil, A) => ceil * rows / (rows + A.gemmHalfM);         // GEMM efficiency rises with rows (small-M penalty)
const etaAttn = (rows, ceil, A) => ceil * rows / (rows + A.attnHalfM);   // attention efficiency rises with rows too, but saturates much faster than a GEMM (see attnHalfM)

// Expected experts read and max-rank load imbalance for b tokens in the step (PLAN §6.3).
export function moeLoad(M, P, b) {
  const mu = Math.max(b * M.topK / P.ep, 1e-9);
  let imb = 1;
  if (P.ep > 1) {
    const spread = Math.sqrt(2 * Math.log(Math.max(P.ep, 2)) / Math.max(mu, 1));                 // max of EP Poisson-ish loads
    imb = Math.min(3, 1 + spread * (P.eplb ? 0.55 : 1) + (P.eplb ? 0.04 : 0.22) * Math.min(1, Math.log2(P.ep + 1) / 6));   // + persistent skew EPLB mostly removes
  }
  const local = P.slots / P.ep;
  const touched = Math.min(local, local * (1 - Math.pow(1 - M.topK / Math.max(M.experts, 1), b)));   // 1 − (1 − k/E)^b of the local experts get a token
  return { mu, imb, tokLoc: mu * imb, local, touched };
}

// One layer of a given kind for one (micro)batch: returns op + collective lists and their totals.
function layerOps(kind, nReq, q, ctx, M, H, W, A, P) {
  const D = P.D, bw = H.bw * A.hbmEff, tok = nReq * q, ops = [], comm = [];
  const add = (name, cat, fl, by, t, extra) => ops.push({ name, cat, flops: fl, bytes: by, t, ai: fl / Math.max(by, 1), ...extra });
  const peakW = H.flops[M.wDtype];
  // attention projections
  { const params = D.attnP / P.tpA, fl = 2 * tok * params, by = params * D.bW, tc = fl / (peakW * eta(tok, A.gemmEff, A)), tm = by / bw;
    add('Attn projections', 'attnProj', fl, by, Math.max(tc, tm), { bound: tc > tm ? 'compute' : 'memory' }); }
  // attention core: read the KV cache; absorbed-MLA / GQA / hybrid FLOPs
  { const kvm = D.kv(ctx, P.tpA), Hl = M.heads / P.tpA, by = nReq * kvm.read / P.cp;
    const perTok = M.attn === 'mla' ? 2 * Hl * ctx * (M.kvLora + M.rope + M.kvLora) : M.attn === 'hybrid' ? 4 * Hl * kvm.ent * M.headDim + kvm.idxFl : 4 * Hl * ctx * M.headDim;
    const fl = perTok * tok / P.cp, peak = H.flops[M.kvDtype === 'bf16' ? 'bf16' : 'fp8'] * etaAttn(tok, A.attnEff, A), tc = fl / peak, tm = by / bw;
    add('Attention (KV read)', 'kv', fl, by, Math.max(tc, tm), { bound: tc > tm ? 'compute' : 'memory' }); }
  // dense FFN, or shared expert(s) + router in MoE layers
  { const params = (kind === 'moe' ? D.sharedP : D.denseP) / P.tpA + (kind === 'moe' ? D.routerP : 0);      // router is replicated
    if (params > 0) { const fl = 2 * tok * params, by = params * D.bW, tc = fl / (peakW * eta(tok, A.gemmEff, A)), tm = by / bw;
      add(M.moe ? 'Shared / dense FFN' : 'Dense FFN', 'dense', fl, by, Math.max(tc, tm), { bound: tc > tm ? 'compute' : 'memory' }); } }
  // routed experts
  let load = null;
  if (kind === 'moe') {
    const b = tok * P.dpA;                                               // tokens of the whole replica-stage microbatch
    load = moeLoad(M, P, b);
    const fl = 2 * load.tokLoc * D.expertP / P.tpM, by = load.touched * D.expertP / P.tpM * D.bE;
    const Me = Math.max(1, load.tokLoc / Math.max(load.touched, 1e-9));  // GEMM rows per expert
    const tc = fl / (H.flops[M.expDtype] * eta(Me, A.moeEff, A)), tm = by / bw;
    add('Routed experts', 'moe', fl, by, Math.max(tc, tm), { bound: tc > tm ? 'compute' : 'memory' });
    if (P.ep > 1) {                                                      // dispatch (FP8) + combine (BF16), one send per destination rank
      const perGpu = b / (P.dpA * P.tpA * P.cp), dest = distinctDest(P.ep, M.topK);
      const Sd = perGpu * dest * M.hidden * D.dispatchB * P.tpM, Sc = perGpu * dest * M.hidden * D.combineB * P.tpM;
      comm.push({ name: 'EP dispatch', kind: 'a2a', n: P.ep, bytes: Sd, t: collective('a2a', Sd, P.ep, H, A) });
      comm.push({ name: 'EP combine', kind: 'a2a', n: P.ep, bytes: Sc, t: collective('a2a', Sc, P.ep, H, A) });
    }
    if (P.tpM > 1) { const S = load.tokLoc * M.hidden * 2; comm.push({ name: 'Expert-TP reduce', kind: 'rs', n: P.tpM, bytes: S, t: collective('rsag', S, P.tpM, H, A) }); }
  }
  if (P.tpA > 1) { const S = tok * M.hidden * 2; comm.push({ name: 'TP all-reduce ×2', kind: 'ar', n: P.tpA, bytes: 2 * S, t: 2 * collective('ar', S, P.tpA, H, A) }); }
  if (P.cp > 1) {                                                        // partial attention output + log-sum-exp across CP ranks
    const v = M.attn === 'mla' ? M.kvLora : M.headDim, S = tok * (M.heads / P.tpA) * (v + 1) * 2;
    comm.push({ name: 'CP partial-attn reduce', kind: 'ar', n: P.cp, bytes: S, t: collective('ar', S, P.cp, H, A) });
  }
  return { ops, comm, tc: ops.reduce((s, o) => s + o.t, 0), tm: comm.reduce((s, c) => s + c.t, 0), load };
}

// Layer time for one kind: with dual-batch overlap the batch is split in two halves that ping-pong compute and comm.
function layerTime(kind, nReq, q, ctx, M, H, W, A, P, dbo) {
  const floor = A.floorUs * 1e-6 * (dbo ? 2 : 1);
  if (!dbo) { const r = layerOps(kind, nReq, q, ctx, M, H, W, A, P); return { ...r, t: r.tc + r.tm + floor, hidden: 0, exposed: r.tm, floor }; }
  const h = layerOps(kind, nReq / 2, q, ctx, M, H, W, A, P), hidden = A.overlap * 2 * Math.min(h.tc, h.tm);
  const scale = (x) => ({ ...x, flops: (x.flops || 0) * 2, bytes: x.bytes * 2, t: x.t * 2 });             // both halves: weights are re-read per microbatch
  return { ops: h.ops.map(scale), comm: h.comm.map(scale), tc: 2 * h.tc, tm: 2 * h.tm, load: h.load, t: 2 * h.tc + 2 * h.tm - hidden + floor, hidden, exposed: 2 * h.tm - hidden, floor };
}

const lmHeadTime = (tok, M, H, A, P, D) => {
  const b = tok * P.dpA, fl = 2 * b * M.vocab * M.hidden / P.g, by = M.vocab * M.hidden * D.bW / P.g;
  return Math.max(fl / (H.flops[M.wDtype] * eta(b, A.gemmEff, A)), by / (H.bw * A.hbmEff));
};

export function evalDecode(M, H, W, A, P, ba) {
  const D = P.D, L = M.layers, Ls = Math.ceil(L / P.pp), gamma = W.specGamma || 0, q = gamma + 1;
  const alpha = Math.min(W.specAlpha ?? 0.85, 0.999), E = gamma > 0 ? (1 - Math.pow(alpha, gamma + 1)) / (1 - alpha) : 1;   // expected tokens per verify step
  const ctx = W.isl + W.osl / 2 + q / 2;
  const dbo = !!P.overlap && ba >= 2;
  const kinds = [];
  if (D.moeLayers > 0) kinds.push(['moe', D.moeLayers / L]);
  if (D.denseLayers > 0) kinds.push(['dense', D.denseLayers / L]);
  const per = kinds.map(([k, w]) => ({ k, w, r: layerTime(k, ba, q, ctx, M, H, W, A, P, dbo) }));

  // merge the per-kind op lists into one layer-average list (what the UI table / roofline show)
  const byName = new Map(), commBy = new Map();
  for (const { w, r } of per) {
    for (const o of r.ops) { const a = byName.get(o.name) || { ...o, flops: 0, bytes: 0, t: 0, _bt: -1 }; a.flops += o.flops * w; a.bytes += o.bytes * w; a.t += o.t * w; if (o.t * w > a._bt) { a.bound = o.bound; a._bt = o.t * w; } byName.set(o.name, a); }
    for (const c of r.comm) { const a = commBy.get(c.name) || { ...c, bytes: 0, t: 0 }; a.bytes += c.bytes * w; a.t += c.t * w; commBy.set(c.name, a); }
  }
  const ops = [...byName.values()].map((o) => ({ ...o, ai: o.flops / Math.max(o.bytes, 1) }));
  const comm = [...commBy.values()];
  const tLayer = per.reduce((s, x) => s + x.w * x.r.t, 0);
  const hidden = per.reduce((s, x) => s + x.w * x.r.hidden, 0), exposed = per.reduce((s, x) => s + x.w * x.r.exposed, 0);
  const commT = comm.reduce((s, c) => s + c.t, 0), floor = per.reduce((s, x) => s + x.w * x.r.floor, 0);
  const load = per.find((x) => x.k === 'moe')?.r.load || { imb: 1, touched: 0, tokLoc: 0 };

  const tLm = lmHeadTime(ba * q, M, H, A, P, D);
  const tStage = Ls * tLayer + tLm;                                        // slowest stage: the last one carries the LM head
  const tok = ba * q, hop = P.pp > 1 ? collective('p2p', tok * M.hidden * 2 * (M.hcMult || 1) / (P.tpA * P.cp), 2, H, A) : 0;
  const m = P.microbatches, f = Math.max(m, P.pp);
  let draft = 0;                                                            // MTP / draft layer, run γ times sequentially on 1 token per request
  if (gamma > 0) { const kd = D.moeLayers > 0 ? 'moe' : 'dense'; draft = gamma * (layerTime(kd, ba, 1, ctx, M, H, W, A, P, dbo).t + lmHeadTime(ba, M, H, A, P, D)); }
  const step = f * tStage + P.pp * hop + A.overheadMs * 1e-3 + draft;    // seconds per verify step
  const tpot = step / E;                                                    // effective seconds per output token per user
  const b = ba * P.dpA, thrRep = m * b * E / step, total = thrRep * P.replicas, perGpu = total / P.gpus;

  // step-time breakdown per emitted token (ms). Time inside f·t_stage is scaled by m; the (f − m) idle stage-slots are pipeline bubbles.
  const cats = { attnProj: 0, kv: 0, dense: 0, moe: 0 };
  ops.forEach((o) => { cats[o.cat] += o.t * Ls * m / E; });
  const ms = 1e3;
  const segs = [
    { key: 'attnProj', label: 'Attn projections', ms: cats.attnProj * ms },
    { key: 'kv', label: 'Attention core (KV)', ms: cats.kv * ms },
    { key: 'dense', label: 'Dense / shared FFN', ms: cats.dense * ms },
    { key: 'moe', label: 'Routed experts', ms: cats.moe * ms },
    { key: 'comm', label: 'Communication (exposed)', ms: exposed * Ls * m / E * ms },
    { key: 'hidden', label: 'Communication (hidden by overlap)', ms: hidden * Ls * m / E * ms, hidden: true },
    { key: 'pp', label: 'PP hops + bubbles', ms: ((f - m) * tStage + P.pp * hop) / E * ms },
    { key: 'spec', label: 'Speculation (draft steps)', ms: draft / E * ms },
    { key: 'over', label: 'Framework + kernel floors', ms: ((A.overheadMs * 1e-3) / E + (floor * Ls + tLm) * m / E) * ms },
  ].filter((s) => s.ms > 1e-6);
  const mem = memoryPerGpu(M, H, W, A, P, D);
  const oom = mem.kvUsedFor(ba) > mem.free;
  return {
    tpot, tokUser: 1 / tpot, total, perGpu, b, ba, ops, comm, segs, imb: load.imb, touched: load.touched, tokLoc: load.tokLoc, exposed, hidden, commT, tStage, tLayer, mem, oom,
    stepMs: step * 1e3, E, concurrency: m * b * P.replicas,
    bottleneck: bottleneck(segs, ops, mem, oom, M),
  };
}

const gb = (x) => (x / 1e9).toFixed(0) + ' GB';
function bottleneck(segs, ops, mem, oom, M) {
  if (mem.free <= 0) return { kind: 'capacity', text: `Weights alone don’t fit: ${gb(mem.wAttn + mem.wExp + mem.wEmb)} per GPU (attention ${gb(mem.wAttn)}, experts ${gb(mem.wExp)}) against ${gb(mem.hbm)} of HBM. Use fewer redundant expert slots, a larger EP/PP, or FP4 experts.` };
  if (oom) return { kind: 'capacity', text: `Out of memory: KV for this batch needs more than the ${gb(mem.free)} free per GPU. Lower the batch, raise DP-attn / CP, or use FP8 KV.` };
  const vis = segs.filter((s) => !s.hidden), total = vis.reduce((a, s) => a + s.ms, 0), top = [...vis].sort((a, b) => b.ms - a.ms)[0];
  const pct = Math.round((top.ms / total) * 100);
  const bound = top.key === 'kv' ? ops.find((o) => o.cat === 'kv')?.bound : top.key === 'moe' ? ops.find((o) => o.cat === 'moe')?.bound : null;
  const why = {
    kv: `Attention over the KV cache dominates (${pct}% of the step, ${bound}-bound). More CP/TP shards KV reads; a smaller batch or FP8 KV cuts them.`,
    moe: `Routed experts dominate (${pct}%, ${bound}-bound). ${bound === 'compute' ? 'Experts are past the ridge point: more EP won’t add throughput, only FP4 weights or a smaller batch help.' : 'Expert weights are re-read each step: raise the batch or widen EP so each expert sees more tokens.'}`,
    attnProj: `Attention projections dominate (${pct}%). Weights are replicated per DP-attn rank: increase TP or shrink DP to spread them.`,
    dense: `Dense / shared FFN dominates (${pct}%). Increase TP to multiply bandwidth for these layers.`,
    comm: `Exposed communication dominates (${pct}%). Try dual-batch overlap, fewer TP ranks, or FP8 dispatch.`,
    pp: `Pipeline hops/bubbles dominate (${pct}%). Reduce PP or add microbatches.`,
    spec: `Draft steps dominate (${pct}%): lower the draft length or acceptance is too low to pay for them.`,
    over: `Fixed overheads (launch floors, LM head, scheduler) dominate (${pct}%): typical at tiny batch or many layers per stage.`,
  };
  return { kind: top.key, text: why[top.key] };
}
