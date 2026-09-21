import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near, run } from './util.mjs';

test('optimizer picks attention-DP + wide EP for DeepSeek decode at moderate interactivity (PLAN M4 acceptance)', () => {
  const M = mk('deepseek_v3'), res = run(C.searchDecode(M, H, W0, A0, 72, { slo: 20 }));
  const best = C.pickTop(res.pts, { objective: 'slo', slo: 20 }).top[0];
  assert.ok(best.P.dpA >= 18 && best.P.ep >= 36 && best.P.tpA === 1, C.cfgLabel(best));
  assert.ok(best.tokUser >= 20);
});

test('best-at-SLO batch is maximal: one more request per rank violates the SLO (or hits capacity)', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 72, eplb: true, overlap: true }), slo = 40;
  const p = C.bestAtSlo(M, H, W0, A0, c, 72, slo), P = p.P, mem = C.memoryPerGpu(M, H, W0, A0, P, P.D);
  assert.ok(p.tokUser >= slo);
  if (p.ba < Math.min(mem.perRankMax, 4096)) assert.ok(C.evalDecode(M, H, W0, A0, P, p.ba + 1).tokUser < slo);
});

test('interactivity vs throughput trade-off: the frontier is monotone (higher tok/s/user ⇒ lower tok/s/GPU)', () => {
  const M = mk('deepseek_v3'), res = run(C.searchDecode(M, H, W0, A0, 72, {}));
  const f = res.front; assert.ok(f.length >= 5);
  for (let i = 1; i < f.length; i++) { assert.ok(f[i].tokUser >= f[i - 1].tokUser); assert.ok(f[i].perGpu <= f[i - 1].perGpu * 1.0001); }
});

test('optimizer runtime: full search on 72 GPUs is fast enough for the browser (< 10 s even in Node)', () => {
  const M = mk('deepseek_v4_pro'), t0 = Date.now(); run(C.searchDecode(M, H, W0, A0, 72, { slo: 20 }));
  assert.ok(Date.now() - t0 < 10000, (Date.now() - t0) + ' ms');
});

test('objectives: max-throughput ≥ SLO-constrained throughput; latency-optimal has the highest tok/s/user', () => {
  const M = mk('deepseek_v3'), res = run(C.searchDecode(M, H, W0, A0, 72, { slo: 50 }));
  const thr = C.pickTop(res.pts, { objective: 'thr' }).top[0], slo = C.pickTop(res.pts, { objective: 'slo', slo: 50 }).top[0], lat = C.pickTop(res.pts, { objective: 'lat', minGpu: 500 }).top[0];
  assert.ok(thr.total >= slo.total); assert.ok(lat.tokUser >= slo.tokUser * 0.999); assert.ok(lat.perGpu >= 500);
});

test('prefill/decode split: rate-matched, beats a naive 50/50 split, and both sides are feasible', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 4096, osl: 512 };
  const rows = run(C.optimizeSplit(M, H, W, A0, { sizes: [8, 12, 16, 24, 36] }));
  assert.ok(rows.length >= 3); const b = rows[0];
  assert.equal(b.gp + b.gd, 72); assert.ok(b.goodput > 0);
  const near50 = rows.find((r) => r.gp === 36); if (near50) assert.ok(b.goodput >= near50.goodput);
  assert.ok(Math.min(b.rp, b.rd) === b.goodput);
});

test('robustness returns a stability score', () => {
  const M = mk('deepseek_v3'), res = run(C.searchDecode(M, H, W0, A0, 72, { slo: 20 }));
  const { top, score } = C.pickTop(res.pts, { objective: 'slo', slo: 20 }), r = C.robustness(M, H, W0, A0, top, 72, { objective: 'slo', slo: 20 }, score);
  assert.equal(r.n, 8); assert.ok(r.wins >= 0 && r.wins <= 8);
});

test('auto-tune (decode): meets the SLO, stays within 2% of the max throughput, and picks the highest tok/s/user at that knee', () => {
  const M = mk('deepseek_v4_pro'), slo = 20, r = run(C.autoTuneDecode(M, H, W0, A0, 72, { slo }));
  assert.ok(r.point.tokUser >= slo);
  assert.ok(r.point.perGpu >= (1 - C.KNEE_EPS) * r.maxPerGpu - 1e-6, 'within eps of max throughput');
  assert.ok(r.point.tokUser >= r.atSlo.tokUser, 'at least as interactive as the SLO-limited max-throughput point');
  // one more request per rank must not be needed: the previous batch is below the threshold (knee is the smallest batch reaching it)
  if (r.point.ba > 1) { const P = r.point.P, prev = C.evalDecode(M, H, W0, A0, P, r.point.ba - 1); assert.ok(prev.perGpu < (1 - C.KNEE_EPS) * r.maxPerGpu + 1e-6 || prev.tokUser <= r.point.tokUser); }
});

test('auto-tune (decode): a tighter SLO gives a more interactive answer and never more throughput', () => {
  const M = mk('deepseek_v3'), a = run(C.autoTuneDecode(M, H, W0, A0, 72, { slo: 20 })), b = run(C.autoTuneDecode(M, H, W0, A0, 72, { slo: 80 }));
  assert.ok(b.point.tokUser >= 80 && b.point.tokUser > a.point.tokUser); assert.ok(b.maxPerGpu <= a.maxPerGpu + 1e-6);
  assert.equal(run(C.autoTuneDecode(M, H, W0, A0, 72, { slo: 1e5 })), null, 'impossible SLO → null');
});

test('auto-tune (prefill): within 2% of the best request rate with the lowest latency', () => {
  const M = mk('deepseek_v3'), r = C.autoTunePrefill(M, H, W0, A0, 72, 1);
  assert.ok(r.row.reqPerS >= 0.98 * r.maxReqPerS && r.row.ttft <= 1);
  assert.equal(C.autoTunePrefill(M, H, W0, A0, 72, 1e-6), null);
});

test('auto-tune (split): rate-matched and within 2% of the best goodput', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 4096, osl: 512 }, r = run(C.autoTuneSplit(M, H, W, A0, { sizes: [8, 12, 16, 24, 36] }));
  assert.ok(r.row.goodput >= 0.98 * r.maxGoodput && r.row.gp + r.row.gd === 72 && r.row.dec.tokUser >= W.minTokUser);
});
