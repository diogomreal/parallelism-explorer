// Parallel-layout logic: valid-value snapping, folding constraints, redundant expert slots.
import { derive } from './model.js';

// Per pool: gpus = replicas × pp × g.  Attention: TP × DP × CP = g.  MoE: EP × TP_moe = g.

export const divisors = (n) => { const r = []; for (let i = 1; i <= n; i++) if (n % i === 0) r.push(i); return r; };
const nearest = (list, x) => list.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a), list[0]);

// poolMode: 'decode' / 'prefill' = the whole rack serves just that phase; 'split' = one rack shared by a prefill pool and a decode pool.
export function poolGpus(state, pool) {
  if (state.poolMode !== 'split') return 72;
  return pool === 'prefill' ? state.split : 72 - state.split;
}

export function layoutOptions(M, gpus, P) {
  const pp = divisors(gpus).filter((x) => x <= 16 && x <= M.layers);
  const ppV = pp.includes(P.pp) ? P.pp : nearest(pp, P.pp);
  const rep = divisors(gpus / ppV);
  const repV = rep.includes(P.replicas) ? P.replicas : nearest(rep, P.replicas);
  const g = gpus / ppV / repV;
  const maxTp = Math.min(g, M.heads, 64);
  const tp = divisors(g).filter((x) => x <= maxTp && M.heads % x === 0);
  const tpV = tp.includes(P.tpA) ? P.tpA : nearest(tp, P.tpA);
  const cp = divisors(g / tpV);
  const cpV = cp.includes(P.cp) ? P.cp : nearest(cp, P.cp);
  const ep = divisors(g);
  const epV = ep.includes(P.ep) ? P.ep : nearest(ep, P.ep);
  return { pp, ppV, rep, repV, g, tp, tpV, cp, cpV, ep, epV, dpA: g / tpV / cpV, tpM: g / epV };
}

// Expert slots: divisible by EP; with EPLB, 1/8 extra redundant experts for hot-expert replication (DeepSeek: 256 + 32).
export function expertSlots(M, ep, eplb) {
  if (!M.moe) return 0;
  const want = eplb && ep > 1 ? M.experts + Math.ceil(M.experts / 8) : M.experts;
  return Math.ceil(want / ep) * ep;
}

// Snap a requested config to a valid one for this pool size; returns it with derived fields, notes and the option lists for sliders.
export function normalize(M, gpus, P, D = derive(M)) {
  const o = layoutOptions(M, gpus, P);
  const slots = expertSlots(M, o.epV, P.eplb);
  const notes = [];
  if (M.moe && slots > M.experts) notes.push({ lvl: P.eplb && slots - M.experts <= Math.ceil(M.experts / 8) + o.epV ? 'info' : 'warn', text: `${M.experts} experts over EP=${o.epV}: +${slots - M.experts} redundant slots${P.eplb ? ' (EPLB)' : ' (divisibility)'}` });
  if (M.attn !== 'gqa' && o.tpV > 1) notes.push({ lvl: 'warn', text: `${M.attn === 'mla' ? 'MLA' : 'Single-KV-head attention'} + TP=${o.tpV}: KV is replicated ${o.tpV}× (prefer DP-attn / CP)` });
  if (M.attn === 'gqa' && o.tpV > M.kvHeads) notes.push({ lvl: 'warn', text: `TP=${o.tpV} > ${M.kvHeads} KV heads: KV replicated ${o.tpV / M.kvHeads}×` });
  if (P.microbatches < o.ppV) notes.push({ lvl: 'warn', key: 'mb', text: `microbatches (${P.microbatches}) < PP (${o.ppV}): pipeline bubbles` });
  else if (P.microbatches > Math.max(o.ppV, 1) && P.microbatches > 1) notes.push({ lvl: 'info', key: 'mb', text: `microbatches (${P.microbatches}) > PP (${o.ppV}): adds latency and KV, not throughput` });
  if (M.moe && o.epV === 1 && o.g > 1) notes.push({ lvl: 'info', text: 'EP=1: experts only sharded by TP inside the expert' });
  return { ...P, pp: o.ppV, replicas: o.repV, tpA: o.tpV, cp: o.cpV, ep: o.epV, dpA: o.dpA, tpM: o.tpM, g: o.g, gpus, slots, redundant: slots - (M.moe ? M.experts : 0), used: o.ppV * o.repV * o.g, notes, opts: o, D };
}
