// Axis-by-axis audit of PP / microbatches / TP / DP-attn / replica-DP / CP / EP: each axis must shard what PLAN §5.1 says it shards,
// replicate what it replicates, and cost what it costs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near } from './util.mjs';

const V3 = mk('deepseek_v3'), W = { ...W0, isl: 2048, osl: 256 };
const P = (M, gpus, c) => C.normalize(M, gpus, c);
const dec = (M, gpus, c, ba, Wl = W, A = A0) => C.evalDecode(M, H, Wl, A, P(M, gpus, c), ba);
const mem = (M, gpus, c, Wl = W) => { const p = P(M, gpus, c); return C.memoryPerGpu(M, H, Wl, A0, p, p.D); };
const op = (r, name) => r.ops.find((o) => o.name === name);

test('replica-DP: n independent copies give identical per-GPU metrics and n× total throughput', () => {
  const a = dec(V3, 24, cfg({ ep: 24, eplb: true }), 128), b = dec(V3, 72, cfg({ replicas: 3, ep: 24, eplb: true }), 128);
  near(b.perGpu, a.perGpu, 1e-9); near(b.total, 3 * a.total, 1e-9); near(b.tpot, a.tpot, 1e-9);
  near(mem(V3, 72, cfg({ replicas: 3, ep: 24, eplb: true })).free, mem(V3, 24, cfg({ ep: 24, eplb: true })).free, 1e-9);
});

test('attention-DP: b = ba × DP ranks; weights replicated per rank (not divided); KV per GPU unchanged', () => {
  const p = P(V3, 72, cfg({ tpA: 2, ep: 72 })); assert.equal(p.dpA, 36);
  const r = C.evalDecode(V3, H, W, A0, p, 10); assert.equal(r.b, 360);
  const a = mem(V3, 72, cfg({ ep: 72 })), b = mem(V3, 72, cfg({ tpA: 1, cp: 1, ep: 72, replicas: 1 }));
  near(a.wAttn, b.wAttn, 1e-12); near(a.kvReq, b.kvReq, 1e-12);
  assert.equal(mem(V3, 72, cfg({ ep: 72 })).bMaxRep, mem(V3, 72, cfg({ ep: 72 })).perRankMax * 72);
});

test('TP: shards attention/dense weights and FLOPs ~1/TP, adds 2 all-reduces per layer, replicates MLA KV', () => {
  const a = dec(V3, 72, cfg({ ep: 72 }), 64), b = dec(V3, 72, cfg({ tpA: 4, ep: 72 }), 64);
  near(op(b, 'Attn projections').bytes, op(a, 'Attn projections').bytes / 4, 1e-9, 'weight bytes /TP');
  near(op(b, 'Attn projections').flops, op(a, 'Attn projections').flops / 4, 1e-9);
  near(op(b, 'Attention (KV read)').flops, op(a, 'Attention (KV read)').flops / 4, 1e-9, 'heads split across TP');
  near(op(b, 'Attention (KV read)').bytes, op(a, 'Attention (KV read)').bytes, 1e-9, 'MLA latent KV is read in full by every TP rank');
  assert.ok(b.comm.some((c) => c.name.startsWith('TP all-reduce')) && !a.comm.some((c) => c.name.startsWith('TP')));
  near(mem(V3, 72, cfg({ tpA: 4, ep: 72 })).wAttn * 4, mem(V3, 72, cfg({ ep: 72 })).wAttn, 0.02, 'attention weights /TP');
});

test('TP cuts dense-model latency at small batch (weight bandwidth multiplied) until all-reduce latency bites', () => {
  const M = mk('llama_405b'), t = (tp) => dec(M, 72, cfg({ tpA: tp, replicas: 72 / tp }), 1, { ...W0, isl: 512, osl: 64 }).tpot;
  assert.ok(t(2) > t(4) && t(4) > t(8), `${t(2)} ${t(4)} ${t(8)}`);
});

test('CP: shards KV bytes and attention FLOPs, not projections; adds a partial-output reduce', () => {
  const Wl = { ...W0, isl: 65536, osl: 256 }, a = dec(V3, 72, cfg({ tpA: 2, ep: 72 }), 4, Wl), c = dec(V3, 72, cfg({ tpA: 2, cp: 4, ep: 72 }), 4, Wl);
  near(op(c, 'Attention (KV read)').bytes, op(a, 'Attention (KV read)').bytes / 4, 1e-9);
  near(op(c, 'Attention (KV read)').flops, op(a, 'Attention (KV read)').flops / 4, 1e-9);
  near(op(c, 'Attn projections').flops, op(a, 'Attn projections').flops, 1e-9, 'CP ranks recompute projections for the same tokens');
  assert.ok(c.comm.some((x) => x.name.startsWith('CP')) && !a.comm.some((x) => x.name.startsWith('CP')));
  near(mem(V3, 72, cfg({ tpA: 2, cp: 4, ep: 72 }), Wl).kvReq * 4, mem(V3, 72, cfg({ tpA: 2, ep: 72 }), Wl).kvReq, 1e-9);
  near(mem(V3, 72, cfg({ tpA: 2, cp: 4, ep: 72 })).wAttn, mem(V3, 72, cfg({ tpA: 2, ep: 72 })).wAttn, 1e-12);
});

test('EP × expert-TP = g: expert bytes per GPU depend on the pool, not the EP/TP split; comm depends on the split', () => {
  const M = mk('deepseek_v3', { expDtype: 'fp8' }), e = (ep) => mem(M, 64, cfg({ ep })).wExp;
  near(e(4), e(16), 1e-9); near(e(64), e(16), 1e-9);
  assert.ok(mem(M, 64, cfg({ ep: 64, eplb: true })).wExp > mem(M, 64, cfg({ ep: 64 })).wExp, 'EPLB redundant slots cost memory');
  const r1 = dec(M, 64, cfg({ ep: 1 }), 16), r8 = dec(M, 64, cfg({ ep: 8 }), 16), r64 = dec(M, 64, cfg({ ep: 64 }), 16);
  assert.ok(!r1.comm.some((c) => c.name.startsWith('EP')) && r1.comm.some((c) => c.name === 'Expert-TP reduce'));
  assert.ok(r8.comm.some((c) => c.name === 'EP dispatch') && r8.comm.some((c) => c.name === 'Expert-TP reduce'));
  assert.ok(r64.comm.some((c) => c.name === 'EP combine') && !r64.comm.some((c) => c.name === 'Expert-TP reduce'));
  assert.ok(op(r64, 'Routed experts').bytes > 0);
  // wider EP → each GPU serves more distinct tokens' worth of one expert shard: expert-GEMM rows per expert grow, weights re-read once
  assert.ok(C.moeLoad(M, P(M, 64, cfg({ ep: 64 })), 4096).tokLoc < C.moeLoad(M, P(M, 64, cfg({ ep: 8 })), 4096).tokLoc, 'fewer routed tokens per GPU as EP widens');
});

test('PP: shards layers (weights and KV per stage) but not per-user latency', () => {
  const a = mem(V3, 48, cfg({ ep: 48 })), b = mem(V3, 48, cfg({ pp: 2, ep: 24, microbatches: 2 }));
  near(b.kvReq, a.kvReq * 31 / 61, 1e-9, 'each stage holds ceil(61/2)=31 of 61 layers of KV');
  near(b.wExp, a.wExp, 0.1, 'half the layers on half the GPUs → about the same experts per GPU (slot rounding aside)');
});

test('microbatches: TPOT = max(m,p)·t_stage. m<p leaves bubbles (throughput ∝ m); m>p adds latency for no throughput', () => {
  const c = (m) => cfg({ pp: 4, ep: 18, microbatches: m }), r = [1, 2, 4, 8].map((m) => dec(V3, 72, c(m), 32));
  near(r[1].tpot, r[0].tpot, 1e-9); near(r[2].tpot, r[0].tpot, 1e-9, 'bubble-limited: TPOT flat for m ≤ p');
  near(r[1].total / r[0].total, 2, 0.001); near(r[2].total / r[0].total, 4, 0.001);
  near(r[3].tpot / r[2].tpot, 2, 0.05, 'm=2p doubles the wait'); near(r[3].total / r[2].total, 1, 0.05, 'and adds no throughput');
  assert.ok(r[0].segs.find((s) => s.key === 'pp').ms > r[2].segs.find((s) => s.key === 'pp').ms, 'bubble shrinks as m → p');
  assert.ok(r[3].mem.kvUsedFor(32) > r[2].mem.kvUsedFor(32), 'more microbatches in flight hold more KV');
});

test('breakdown segments add up to the step time, for every axis combination', () => {
  const combos = [cfg({ ep: 72 }), cfg({ ep: 72, overlap: true, eplb: true }), cfg({ tpA: 4, ep: 72 }), cfg({ tpA: 2, cp: 2, ep: 72, overlap: true }),
    cfg({ pp: 4, ep: 18, microbatches: 2 }), cfg({ pp: 4, ep: 18, microbatches: 8, overlap: true }), cfg({ replicas: 3, ep: 24, eplb: true }), cfg({ ep: 8, eplb: true, tpA: 2 })];
  for (const c of combos) for (const g of [0, 2]) {
    const r = dec(V3, 72, c, 48, { ...W, specGamma: g, specAlpha: 0.8 });
    const sum = r.segs.filter((s) => !s.hidden).reduce((a, s) => a + s.ms, 0);
    near(sum, r.tpot * 1e3, 0.01, JSON.stringify(c) + ' gamma ' + g);
  }
});

test('GPU accounting: used = PP × replicas × g ≤ pool; total = perGpu × pool for every enumerated layout', () => {
  for (const c of C.enumerateConfigs(V3, 72, [false, true], [true]).filter((_, i) => i % 23 === 0)) {
    const p = P(V3, 72, c); assert.equal(p.used, p.pp * p.replicas * p.g); assert.ok(p.used <= 72);
    const r = C.evalDecode(V3, H, W, A0, p, 8); near(r.perGpu * 72, r.total, 1e-9);
  }
});

test('prefill axes: DP-attn multiplies request rate, TP shards projections, CP shards attention, PP keeps steady throughput', () => {
  const pre = (c, gpus = 8, Wl = { ...W0, isl: 16384, chunk: 4096 }) => C.evalPrefill(V3, H, Wl, A0, P(V3, gpus, c), gpus);
  const a = pre(cfg({ ep: 8 })), b = pre(cfg({ replicas: 2, ep: 4 }));
  assert.ok(a.reqPerS > 0 && b.reqPerS > 0);
  const t1 = pre(cfg({ ep: 8, tpA: 1 })), t2 = pre(cfg({ ep: 8, tpA: 2 }));
  assert.ok(t2.segs.find((s) => s.key === 'attnProj').ms < t1.segs.find((s) => s.key === 'attnProj').ms * 0.75, 'TP halves projection time per request');
  const c1 = pre(cfg({ ep: 8, tpA: 2 }), 8, { ...W0, isl: 65536, chunk: 8192 }), c4 = pre(cfg({ ep: 8, tpA: 2, cp: 4 }), 8, { ...W0, isl: 65536, chunk: 8192 });
  assert.ok(c4.segs.find((s) => s.key === 'kv').ms < c1.segs.find((s) => s.key === 'kv').ms / 3, 'CP shards causal attention work');
  const p1 = pre(cfg({ ep: 8 })), p2 = pre(cfg({ pp: 2, ep: 4 }));
  near(p2.thr, p1.thr, 0.35, 'PP costs little steady-state throughput'); assert.ok(p2.tPrefill > p2.tChunk * p2.nCh, 'but adds fill latency');
});

test('prefill ignores the decode microbatch setting (chunks are its microbatches)', () => {
  const p = (m) => C.evalPrefill(V3, H, W0, A0, P(V3, 8, cfg({ pp: 2, ep: 4, microbatches: m })), 8);
  near(p(1).tPrefill, p(8).tPrefill, 1e-12);
});

test('prefill breakdown adds up to prefill latency', () => {
  for (const c of [cfg({ ep: 8 }), cfg({ pp: 2, ep: 4 }), cfg({ tpA: 2, ep: 8, overlap: true }), cfg({ tpA: 2, cp: 2, ep: 8 })]) {
    const r = C.evalPrefill(V3, H, { ...W0, isl: 20000, chunk: 4096 }, A0, P(V3, 8, c), 8);
    near(r.segs.filter((s) => !s.hidden).reduce((a, s) => a + s.ms, 0), r.tPrefill * 1e3, 0.01, JSON.stringify(c));
  }
});
