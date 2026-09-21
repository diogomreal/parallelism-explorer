import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near } from './util.mjs';

const dec = (M, gpus, c, ba, W = W0, A = A0, HW = H) => { const P = C.normalize(M, gpus, c); return C.evalDecode(M, HW, W, A, P, ba); };

test('batch=1 dense decode is bound by weights / HBM bandwidth', () => {
  const M = mk('llama_405b'), D = C.derive(M);
  const r = dec(M, 8, cfg({ tpA: 8 }), 1, { ...W0, isl: 128, osl: 128 });
  const floor = D.total * D.bW / 8 / (H.bw * A0.hbmEff);
  assert.ok(r.tpot >= floor && r.tpot < 2.4 * floor, `tpot ${r.tpot} vs weight-read floor ${floor}`);
  assert.ok(r.ops.find((o) => o.name === 'Attn projections').bound === 'memory');
});

test('batch=1 MoE decode reads only the touched experts (cheaper than dense-equivalent)', () => {
  const M = mk('deepseek_v3'), P = C.normalize(M, 72, cfg({ ep: 72 }));
  assert.ok(C.moeLoad(M, P, 1).touched < 0.1 * (P.slots / P.ep), 'one token touches ~top-k of 288 slots: ' + C.moeLoad(M, P, 1).touched);
  assert.ok(C.moeLoad(M, P, 1e5).touched > 3.99 && C.moeLoad(M, P, 1e5).touched <= 4, 'nearly all local experts touched at large batch');
});

test('TPOT is non-decreasing in batch; throughput per GPU rises then saturates', () => {
  const M = mk('deepseek_v3'); let prevT = 0, prevP = 0;
  for (const ba of [1, 4, 16, 64, 256, 1024]) { const r = dec(M, 72, cfg({ ep: 72, overlap: true, eplb: true }), ba); assert.ok(r.tpot >= prevT * 0.999, 'ba ' + ba); assert.ok(r.perGpu >= prevP * 0.999); prevT = r.tpot; prevP = r.perGpu; }
});

test('monotonic in hardware: more HBM bandwidth / more overlap never slows a step', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 72, overlap: true, eplb: true });
  const base = dec(M, 72, c, 256).tpot;
  assert.ok(dec(M, 72, c, 256, W0, { ...A0, hbmEff: 0.95 }).tpot <= base);
  assert.ok(dec(M, 72, c, 256, W0, { ...A0, overlap: 1 }).tpot <= base);
  assert.ok(dec(M, 72, c, 256, W0, { ...A0, alphaUs: 5 }).tpot <= base);
  assert.ok(dec(M, 72, c, 256, W0, { ...A0, overlap: 0 }).tpot >= base);
});

test('pipeline formula: TPOT = max(m,p)·t_stage + p·t_hop + overhead; PP does not reduce per-user latency', () => {
  const M = mk('deepseek_v3');
  const r1 = dec(M, 72, cfg({ ep: 72 }), 64), r4 = dec(M, 72, cfg({ pp: 4, ep: 18, microbatches: 4 }), 64);
  near(r4.tpot, Math.max(4, 4) * r4.tStage + 4 * 0 + 0.3e-3, 0.2, 'p·t_hop small on NVLink');   // 4 stages, 4 microbatches
  assert.ok(r4.tpot > r1.tpot * 0.9, 'PP buys capacity, not latency');
  const rb = dec(M, 72, cfg({ pp: 4, ep: 18, microbatches: 1 }), 64);      // m < p leaves bubbles
  assert.ok(rb.segs.find((s) => s.key === 'pp').ms > 0);
});

test('dual-batch overlap hides communication but re-reads weights per microbatch', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 72, eplb: true });
  const noO = dec(M, 72, c, 512), dbo = dec(M, 72, { ...c, overlap: true }, 512);
  assert.ok(dbo.hidden > 0 && noO.hidden === 0);
  assert.ok(dbo.tpot < noO.tpot, 'large batch: DBO wins');
  const s1 = dec(M, 72, c, 2), s2 = dec(M, 72, { ...c, overlap: true }, 2);
  assert.ok(s2.tpot > s1.tpot * 0.95, 'tiny batch: DBO does not help (weights read twice)');
});

test('wide EP + attention DP beats TP for DeepSeek at high batch (PLAN M2 acceptance)', () => {
  const M = mk('deepseek_v3');
  const wide = dec(M, 72, cfg({ ep: 72, overlap: true, eplb: true }), 384), tp8 = dec(M, 72, cfg({ tpA: 8, replicas: 9, ep: 8, overlap: true, eplb: true }), 384);
  assert.ok(wide.perGpu > tp8.perGpu, `wide-EP ${wide.perGpu} vs TP8×9 ${tp8.perGpu}`);
});

test('MLA + TP wastes capacity: KV replicated, so TP does not raise max batch', () => {
  const M = mk('deepseek_v3'), a = dec(M, 72, cfg({ tpA: 1, ep: 72 }), 32), b = dec(M, 72, cfg({ tpA: 8, ep: 72 }), 32);
  assert.ok(b.mem.perRankMax <= a.mem.perRankMax * 1.3);
});

test('CP shards KV reads at long context', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 262144, osl: 1024 };
  const a = dec(M, 72, cfg({ tpA: 2, ep: 72 }), 2, W), c = dec(M, 72, cfg({ tpA: 2, cp: 4, ep: 72 }), 2, W);
  assert.ok(c.ops.find((o) => o.cat === 'kv').bytes < a.ops.find((o) => o.cat === 'kv').bytes / 3.5);
});

test('speculative decoding: E[tokens/step] = (1−α^(γ+1))/(1−α); big win at small batch, little at large batch', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 72, eplb: true, overlap: true });
  const W1 = { ...W0, specGamma: 2, specAlpha: 0.8 };
  const r = dec(M, 72, c, 8, W1); near(r.E, (1 - 0.8 ** 3) / 0.2, 1e-9);
  const small = dec(M, 72, c, 8).tokUser, smallS = dec(M, 72, c, 8, W1).tokUser;
  const big = dec(M, 72, c, 1024).tokUser, bigS = dec(M, 72, c, 1024, W1).tokUser;
  assert.ok(smallS / small > 1.3, 'speed-up at small batch ' + smallS / small);
  assert.ok(bigS / big < smallS / small, 'speed-up shrinks at large batch (' + bigS / big + ')');
});

test('exact-zero alpha or gamma=0 leaves the step unchanged', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 72 });
  near(dec(M, 72, c, 64, { ...W0, specGamma: 0, specAlpha: 0.9 }).tpot, dec(M, 72, c, 64).tpot, 1e-12);
});

test('collectives follow α–β: all-reduce volume 2(n−1)/n·S at large S; all-to-all = α + S/BW', () => {
  const bw = H.link * A0.linkEff, S = 1e9, n = 8;
  const ring = 2 * (n - 1) / n * S / bw + 2 * (n - 1) * 0.4 * A0.alphaUs * 1e-6;
  assert.ok(C.collective('ar', S, n, H, A0) <= ring + 1e-12, 'min over algorithms never exceeds the ring cost');
  assert.ok(C.collective('ar', S, n, H, A0) >= S / bw, 'no faster than moving S once');
  near(C.collective('ar', 1e4, 2, H, A0), 0.8 * A0.alphaUs * 1e-6, 0.05, 'small messages are latency-bound');
  near(C.collective('a2a', S, 72, H, A0), 1.5 * A0.alphaUs * 1e-6 + S / bw, 1e-9);
  assert.equal(C.collective('ar', S, 1, H, A0), 0);
  near(C.distinctDest(72, 8), 72 * (1 - (71 / 72) ** 8) * 71 / 72, 1e-12);
});

test('MoE load imbalance grows with EP, shrinks with tokens, and EPLB reduces it', () => {
  const M = mk('deepseek_v3'), P = (ep, eplb) => C.normalize(M, 72, cfg({ ep, eplb }));
  assert.ok(C.moeLoad(M, P(72, false), 100).imb > C.moeLoad(M, P(8, false), 100).imb);
  assert.ok(C.moeLoad(M, P(72, false), 1e5).imb < C.moeLoad(M, P(72, false), 100).imb);
  assert.ok(C.moeLoad(M, P(72, true), 1000).imb < C.moeLoad(M, P(72, false), 1000).imb);
  assert.equal(C.moeLoad(M, P(1, false), 1000).imb, 1);
});

test('hybrid attention (V4): KV bytes per step far below MLA at long context', () => {
  const V4 = mk('deepseek_v4_pro'), V3 = mk('deepseek_v3'), W = { ...W0, isl: 262144, osl: 1024 };
  const a = dec(V4, 72, cfg({ tpA: 2, cp: 4, ep: 72 }), 2, W), b = dec(V3, 72, cfg({ tpA: 2, cp: 4, ep: 72 }), 2, W);
  assert.ok(a.ops.find((o) => o.cat === 'kv').bytes < 0.25 * b.ops.find((o) => o.cat === 'kv').bytes);
});

test('all outputs are finite for every preset and pool mode sizes', () => {
  for (const k of Object.keys(C.MODELS)) { const M = mk(k); for (const g of [8, 24, 48, 72]) {
    const ep = M.moe ? g : 1, r = dec(M, g, cfg({ ep, tpA: M.moe ? 1 : Math.min(8, g), eplb: true, overlap: true }), 16);
    for (const key of ['tpot', 'tokUser', 'perGpu', 'total']) assert.ok(Number.isFinite(r[key]) && r[key] > 0, `${k} g=${g} ${key}=${r[key]}`);
    const pre = C.evalPrefill(M, H, W0, A0, C.normalize(M, g, cfg({ ep, tpA: M.moe ? 1 : Math.min(8, g) })), g);
    for (const key of ['thr', 'reqPerS', 'tPrefill', 'kvXfer']) assert.ok(Number.isFinite(pre[key]) && pre[key] > 0, `${k} g=${g} prefill ${key}`);
  } }
});
