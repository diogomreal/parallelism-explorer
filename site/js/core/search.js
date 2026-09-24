// Configuration search (PLAN §8): enumerate valid layouts, find the best batch under an SLO, Pareto frontier, prefill/decode split.
import { divisors, normalize } from './layout.js';
import { derive } from './model.js';
import { evalDecode } from './decode.js';
import { evalPrefill } from './prefill.js';
import { memoryPerGpu } from './memory.js';

export function enumerateConfigs(M, gpus, dboOpts = [false, true], eplbOpts = [true]) {
  const seen = new Set(), out = [];
  const pps = divisors(gpus).filter((x) => x <= 8 && x <= M.layers);
  for (const pp of pps) for (const rep of divisors(gpus / pp)) {
    const g = gpus / pp / rep; if (g < 1) continue;
    for (const tpA of divisors(g)) { if (tpA > 8 || M.heads % tpA) continue;
      for (const cp of divisors(g / tpA)) { if (cp > 4) continue;
        const eps = M.moe ? divisors(g).filter((e) => e === 1 || e >= g / 4) : [1];
        for (const ep of eps) {
          for (const dbo of dboOpts) for (const eplb of (M.moe && ep > 1 ? eplbOpts : [false])) {
            for (const mb of (pp === 1 ? [1] : [pp, 2 * pp])) {
              const key = [pp, rep, tpA, cp, ep, dbo, eplb, mb].join();
              if (seen.has(key)) continue; seen.add(key);
              out.push({ pp, replicas: rep, tpA, cp, ep, microbatches: mb, overlap: dbo, eplb });
            }}}}}}
  return out;
}

const point = (ba, r, cfg, P) => ({ ba, tokUser: r.tokUser, perGpu: r.perGpu, total: r.total, tpot: r.tpot, cfg, P });
export const batchGrid = (cap) => { const g = []; for (let b = 1; b <= cap; b = Math.max(b + 1, Math.round(b * 1.7))) g.push(b); if (cap >= 1 && g[g.length - 1] !== cap) g.push(cap); return g; };

// All (batch → metrics) points of one layout on a pool, up to the memory cap.
export function sweepConfig(M, H, W, A, cfg, gpus, batches) {
  const P = normalize(M, gpus, cfg), mem = memoryPerGpu(M, H, W, A, P, P.D), cap = Math.min(mem.perRankMax, 4096);
  const pts = [];
  for (const ba of (batches || batchGrid(cap))) { if (ba > cap) break; pts.push(point(ba, evalDecode(M, H, W, A, P, ba), cfg, P)); }
  return pts;
}

// Largest batch (integer, per attention rank) that still meets tokens/s/user ≥ slo: TPOT is non-decreasing in batch, so bisect.
export function bestAtSlo(M, H, W, A, cfg, gpus, slo) {
  const P = normalize(M, gpus, cfg), mem = memoryPerGpu(M, H, W, A, P, P.D), cap = Math.min(mem.perRankMax, 4096);
  if (cap < 1) return null;
  const at = (ba) => evalDecode(M, H, W, A, P, ba);
  let r = at(1); if (r.tokUser < slo) return null;
  let lo = 1, hi = cap, best = 1, rb = r;
  const rc = at(cap);
  if (rc.tokUser >= slo) return point(cap, rc, cfg, P);
  while (hi - lo > 1) { const mid = (lo + hi) >> 1, rm = at(mid); if (rm.tokUser >= slo) { lo = mid; rb = rm; } else hi = mid; }
  return point(lo, rb, cfg, P);
}

// Decode-pool search as a generator (yields progress in [0,1] so the UI can stay responsive); returns {pts, front, n}.
export function* searchDecode(M, H, W, A, gpus, { eplb = true, slo = 0, chunk = 40 } = {}) {
  const cfgs = enumerateConfigs(M, gpus, [false, true], [eplb]), pts = [];
  for (let i = 0; i < cfgs.length; i += chunk) {
    for (const c of cfgs.slice(i, i + chunk)) {
      for (const p of sweepConfig(M, H, W, A, c, gpus)) pts.push(p);
      if (slo > 0) { const b = bestAtSlo(M, H, W, A, c, gpus, slo); if (b) pts.push(b); }
    }
    yield Math.min(1, (i + chunk) / cfgs.length);
  }
  return { pts, front: frontier(pts), n: cfgs.length };
}

export function frontier(pts) {
  const s = [...pts].sort((a, b) => b.tokUser - a.tokUser || b.perGpu - a.perGpu);
  const out = []; let best = -1;
  for (const p of s) if (p.perGpu > best * 1.0001) { out.push(p); best = p.perGpu; }
  return out.reverse();
}

export const cfgLabel = (c) => { const P = c.P || c; return `PP${P.pp}·TP${P.tpA}·CP${P.cp}·DPa${P.dpA}·EP${P.ep}`; };
export function family(c) {
  const P = c.P || c;
  if (P.pp >= 4) return 'Deep PP';
  if (P.cp > 1) return 'TP + CP';
  if (P.tpA >= 4) return 'TP + EP';
  if (P.replicas >= 2) return 'Small EP groups × replicas';
  return 'DP-attn + wide EP';
}

// Rank candidate points for an objective; best of each layout family first, then the runners-up (PLAN §8.1/8.4).
export function pickTop(pts, { objective, slo, minGpu }) {
  let feas, score;
  if (objective === 'thr') { feas = pts; score = (p) => p.total; }
  else if (objective === 'slo') { feas = pts.filter((p) => p.tokUser >= slo); score = (p) => p.perGpu; }
  else { feas = pts.filter((p) => p.perGpu >= minGpu); score = (p) => p.tokUser; }
  const sorted = [...feas].sort((a, b) => score(b) - score(a));
  const seen = new Set(), fams = new Set(), top = [], rest = [];
  for (const p of sorted) {
    const k = cfgLabel(p) + p.cfg.microbatches + p.cfg.overlap + p.cfg.eplb; if (seen.has(k)) continue; seen.add(k);
    if (!fams.has(family(p))) { fams.add(family(p)); top.push(p); } else if (rest.length < 6) rest.push(p);
    if (top.length >= 5 && rest.length >= 6) break;
  }
  top.push(...rest.slice(0, Math.max(0, 8 - top.length)));
  return { top, feasible: feas.length, score };
}

// Does the winner survive perturbed assumptions? Re-run the top candidates under 8 scenarios (PLAN §8.4).
export function robustness(M, H, W, A, top, gpus, { objective, slo }, score) {
  const scenarios = [{ gemmEff: Math.min(0.95, A.gemmEff * 1.3) }, { gemmEff: A.gemmEff * 0.85 }, { hbmEff: 0.7 }, { alphaUs: A.alphaUs * 2 }, { overlap: 0 }, { overlap: 1 }, { attnEff: A.attnEff * 0.7 }, { overheadMs: A.overheadMs * 3 }];
  let wins = 0;
  if (top.length) scenarios.forEach((sc) => {
    const A2 = { ...A, ...sc };
    const res = top.slice(0, 5).map((p) => { const c = sweepConfig(M, H, W, A2, p.cfg, gpus).filter((q) => objective !== 'slo' || q.tokUser >= slo); return c.length ? Math.max(...c.map(score)) : -1; });
    if (res[0] >= Math.max(...res) * 0.995) wins++;
  });
  return { wins, n: scenarios.length };
}

// ------------------------------------------------------------------ prefill side + rate-matched split
export function enumeratePrefill(M, gpus) {
  const out = [];
  for (const pp of divisors(gpus).filter((x) => x <= 4 && x <= M.layers)) for (const rep of divisors(gpus / pp)) {
    const g = gpus / pp / rep;
    for (const tpA of divisors(g)) { if (tpA > 8 || M.heads % tpA) continue;
      for (const cp of divisors(g / tpA)) { if (cp > 4) continue;
        for (const ep of (M.moe ? divisors(g).filter((e) => e === 1 || e >= g / 4) : [1]))
          for (const dbo of [false, true]) out.push({ pp, replicas: rep, tpA, cp, ep, microbatches: 1, overlap: dbo, eplb: M.moe && ep > 1 });
      }}}
  return out;
}

// Best prefill layouts + chunk sizes on a pool: maximize requests/s subject to prefill latency ≤ budget seconds. Returns the top n distinct layouts.
export function topPrefill(M, H, W, A, gpus, budget, n = 8, chunks = [2048, 4096, 8192, 16384]) {
  const D = derive(M), rows = [];
  for (const cfg of enumeratePrefill(M, gpus)) {
    const P = normalize(M, gpus, cfg, D);
    if (memoryPerGpu(M, H, W, A, P, D).free <= 0) continue;
    let best = null;
    for (const chunk of chunks) {
      const pre = evalPrefill(M, H, { ...W, chunk }, A, P, gpus);
      if (pre.ttft <= budget && (!best || pre.reqPerS > best.reqPerS)) best = { cfg, P, chunk, reqPerS: pre.reqPerS, thr: pre.thr, ttft: pre.ttft, nCh: pre.nCh, bubble: pre.bubble };
    }
    if (best) rows.push(best);
  }
  rows.sort((a, b) => b.reqPerS - a.reqPerS);
  const seen = new Set(), out = [];
  for (const r of rows) { const k = cfgLabel(r) + r.P.replicas; if (seen.has(k)) continue; seen.add(k); out.push(r); if (out.length >= n) break; }
  return out;
}
export const bestPrefill = (M, H, W, A, gpus, budget, chunks) => topPrefill(M, H, W, A, gpus, budget, 1, chunks)[0] || null;

// Rate-matched prefill/decode split of the rack: for each candidate prefill size, take the best layout on each side, pick the split that
// maximizes min(prefill req/s, decode req/s). Generator: yields progress; returns candidates sorted by goodput.
export function* optimizeSplit(M, H, W, A, { sizes, eplb = true } = {}) {
  const gps = sizes || [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56];
  const rows = [], decCache = new Map();
  const slo = W.minTokUser;
  for (let i = 0; i < gps.length; i++) {
    const gp = gps[i], gd = 72 - gp;
    if (!decCache.has(gd)) {
      let best = null;
      for (const c of enumerateConfigs(M, gd, [false, true], [eplb])) { const p = bestAtSlo(M, H, W, A, c, gd, slo); if (p && (!best || p.perGpu > best.perGpu)) best = p; }
      decCache.set(gd, best);
    }
    const dec = decCache.get(gd);
    if (dec) {
      const budget = Math.max(0.05, W.ttftMs / 1e3 - dec.tpot - 0.010);
      const pre = bestPrefill(M, H, W, A, gp, budget);
      if (pre) {
        const rd = dec.total / W.osl, goodput = Math.min(pre.reqPerS, rd);
        rows.push({ gp, gd, pre, dec, rp: pre.reqPerS, rd, goodput, tokPerGpu: goodput * W.osl / 72, limiting: pre.reqPerS < rd ? 'prefill' : 'decode' });
      }
    }
    yield (i + 1) / gps.length;
  }
  return rows.sort((a, b) => b.goodput - a.goodput);
}

// ------------------------------------------------------------------ auto-tune: highest throughput, stopped at the knee
// Throughput saturates with batch size, so the SLO-limited batch is often far past the knee: it buys almost no throughput and costs a lot
// of tokens/s/user. Auto-tune finds the maximum throughput T*, then returns the point with the HIGHEST tokens/s/user whose throughput is
// still within `eps` of T* (default 2%).
export const KNEE_EPS = 0.02;

// Smallest batch (integer, per attention rank) on a layout whose per-GPU throughput reaches `target`, searching up to `hiBa` (throughput is non-decreasing in batch).
function kneePoint(M, H, W, A, cfg, gpus, target, hiBa) {
  const P = normalize(M, gpus, cfg), at = (ba) => evalDecode(M, H, W, A, P, ba);
  let lo = 1, hi = hiBa, rh = at(hi);
  if (rh.perGpu < target) return null;
  let rb = rh;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1, rm = at(mid); if (rm.perGpu >= target) { hi = mid; rb = rm; } else lo = mid; }
  const r1 = at(lo); if (r1.perGpu >= target) return point(lo, r1, cfg, P);
  return point(hi, rb, cfg, P);
}

// KV placement for auto-tune: tune once with KV in HBM only and once with spill-to-LPDDR5X allowed, keep spill only if it raises the best
// feasible throughput by more than eps (it costs host-link traffic and complexity), and only if the chosen batch actually overflows HBM.
function* runScaled(gen, lo, hi) { let r; while (!(r = gen.next()).done) yield lo + r.value * (hi - lo); return r.value; }
function choosePlacement(none, spill, score, eps) {
  if (!spill || (none && score(spill) <= (1 + eps) * score(none))) return none && { res: none, kv: 'none' };
  return { res: spill, kv: 'spill' };
}
const overflowsHbm = (M, H, W, A, p) => p.ba > memoryPerGpu(M, H, { ...W, kvOffload: 'none' }, A, p.P, p.P.D).perRankMax;

// Decode pool. Returns { point, maxPerGpu, atSlo, kvOffload, kvGain } (point = knee choice; atSlo = the max-throughput point it was reduced from;
// point.kv = the KV placement to apply) or null if the SLO is unreachable.
export function* autoTuneDecode(M, H, W, A, gpus, opts = {}) {
  const eps = opts.eps ?? KNEE_EPS;
  const none = yield* runScaled(autoTuneDecodeAt(M, H, { ...W, kvOffload: 'none' }, A, gpus, opts), 0, 0.5);
  const spill = yield* runScaled(autoTuneDecodeAt(M, H, { ...W, kvOffload: 'spill' }, A, gpus, opts), 0.5, 1);
  const c = choosePlacement(none, spill, (r) => r.maxPerGpu, eps); if (!c) return null;
  const kv = c.kv === 'spill' && overflowsHbm(M, H, { ...W, kvOffload: 'spill' }, A, c.res.point) ? 'spill' : 'none';
  return { ...c.res, point: { ...c.res.point, kv }, kvOffload: kv, kvGain: none && spill ? spill.maxPerGpu / none.maxPerGpu : null };
}
function* autoTuneDecodeAt(M, H, W, A, gpus, { slo, eplb = true, eps = KNEE_EPS } = {}) {
  const cfgs = enumerateConfigs(M, gpus, [false, true], [eplb]), best = [];
  for (let i = 0; i < cfgs.length; i++) {
    const p = bestAtSlo(M, H, W, A, cfgs[i], gpus, slo); if (p) best.push(p);
    if (i % 100 === 99) yield (i + 1) / cfgs.length / 2;
  }
  if (!best.length) return null;
  const top = best.reduce((a, b) => (b.perGpu > a.perGpu ? b : a)), tau = (1 - eps) * top.perGpu;
  let chosen = null; const cand = best.filter((p) => p.perGpu >= tau);
  for (let i = 0; i < cand.length; i++) {
    const k = kneePoint(M, H, W, A, cand[i].cfg, gpus, tau, cand[i].ba);
    if (k && (!chosen || k.tokUser > chosen.tokUser + 1e-9 || (Math.abs(k.tokUser - chosen.tokUser) < 1e-9 && k.perGpu > chosen.perGpu))) chosen = k;
    if (i % 25 === 24) yield 0.5 + (i + 1) / cand.length / 2;
  }
  return { point: chosen || top, maxPerGpu: top.perGpu, atSlo: top };
}

// Prefill pool: max requests/s within the TTFT budget, then the lowest-latency layout within eps of it.
export function autoTunePrefill(M, H, W, A, gpus, budget, eps = KNEE_EPS) {
  const rows = topPrefill(M, H, W, A, gpus, budget, 1000);
  if (!rows.length) return null;
  const tau = (1 - eps) * rows[0].reqPerS, near = rows.filter((r) => r.reqPerS >= tau);
  const row = near.reduce((a, b) => (b.ttft < a.ttft ? b : a));
  return { row, maxReqPerS: rows[0].reqPerS };
}

// Split rack: best rate-matched split, then within eps of its goodput the split whose decode side has the highest tokens/s/user (decode batch cut to the knee).
// The decode pool's KV placement is chosen the same way as in autoTuneDecode. row.dec.kv = the placement to apply.
export function* autoTuneSplit(M, H, W, A, opts = {}) {
  const eps = opts.eps ?? KNEE_EPS;
  const none = yield* runScaled(autoTuneSplitAt(M, H, { ...W, kvOffload: 'none' }, A, opts), 0, 0.5);
  const spill = yield* runScaled(autoTuneSplitAt(M, H, { ...W, kvOffload: 'spill' }, A, opts), 0.5, 1);
  const c = choosePlacement(none, spill, (r) => r.maxGoodput, eps); if (!c) return null;
  const kv = c.kv === 'spill' && overflowsHbm(M, H, { ...W, kvOffload: 'spill' }, A, c.res.row.dec) ? 'spill' : 'none';
  return { ...c.res, row: { ...c.res.row, dec: { ...c.res.row.dec, kv } }, kvOffload: kv };
}
function* autoTuneSplitAt(M, H, W, A, { eplb = true, eps = KNEE_EPS, sizes } = {}) {
  const rows = yield* optimizeSplit(M, H, W, A, { eplb, sizes });
  if (!rows.length) return null;
  const tau = (1 - eps) * rows[0].goodput; let chosen = null;
  for (const r of rows.filter((x) => x.goodput >= tau)) {
    const k = kneePoint(M, H, W, A, r.dec.cfg, r.gd, tau * W.osl / r.gd, r.dec.ba);
    const dec = k || r.dec;
    if (!chosen || dec.tokUser > chosen.dec.tokUser) chosen = { ...r, dec, goodput: r.goodput };
  }
  return { row: chosen, maxGoodput: rows[0].goodput };
}
