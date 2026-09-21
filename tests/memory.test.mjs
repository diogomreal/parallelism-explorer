import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near } from './util.mjs';

const mem = (M, gpus, c, W = W0, A = A0) => { const P = C.normalize(M, gpus, c); return { P, m: C.memoryPerGpu(M, H, W, A, P, P.D) }; };

test('memory parts sum to HBM (conservation)', () => {
  const M = mk('deepseek_v3'); const { m } = mem(M, 72, cfg({ ep: 72, eplb: true, overlap: true }));
  near(m.wAttn + m.wExp + m.wEmb + m.reserved + m.act + m.free, m.hbm, 1e-9);
});

test('expert weights = (slots/EP)·params per GPU (hand calc, FP8)', () => {
  const M = mk('deepseek_v3', { expDtype: 'fp8' }), D = C.derive(M);
  const { m, P } = mem(M, 64, cfg({ ep: 64 }));
  near(m.wExp, D.moeLayers * (256 / 64) * D.expertP, 1e-9, '4 experts per GPU × 58 layers');
  assert.equal(P.slots, 256);
});

test('MLA KV is replicated under TP (not sharded); GQA KV is sharded until TP > kv heads', () => {
  const V3 = mk('deepseek_v3'), a = mem(V3, 72, cfg({ tpA: 1, ep: 72 })).m, b = mem(V3, 72, cfg({ tpA: 4, ep: 72 })).m;
  near(a.kvReq, b.kvReq, 1e-9, 'MLA per-request KV/GPU independent of TP');
  const Q = mk('qwen3_235b'), g1 = mem(Q, 72, cfg({ tpA: 1, ep: 8 })).m, g4 = mem(Q, 72, cfg({ tpA: 4, ep: 8 })).m, g8 = mem(Q, 72, cfg({ tpA: 8, ep: 8 })).m;
  near(g4.kvReq, g1.kvReq / 4, 1e-9, 'GQA-4: TP4 shards the 4 KV heads'); near(g8.kvReq, g4.kvReq, 1e-9, 'TP8 > 4 KV heads: replicated 2×, no further saving');
});

test('CP shards KV per request; more DP-attn ranks scale concurrent requests', () => {
  const M = mk('deepseek_v3');
  const a = mem(M, 72, cfg({ ep: 72 })).m, c = mem(M, 72, cfg({ cp: 4, tpA: 1, ep: 72 })).m;
  near(c.kvReq, a.kvReq / 4, 1e-9);
  assert.ok(a.bMaxRep > 0 && a.bMaxRep === a.perRankMax * 72);
});

test('more GPUs never lower the max feasible batch (same layout family)', () => {
  const M = mk('deepseek_v3'); let prev = -1;
  for (const g of [24, 36, 48, 72]) { const { m } = mem(M, g, cfg({ ep: g, eplb: true })); assert.ok(m.perRankMax >= prev - 1, `g=${g}`); prev = m.perRankMax; }
});

test('V3 on GB200: ≈ 1k+ requests of 2K context per GPU by capacity (LMSYS reports 1408 at 2K, 768 at 4K)', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 2000, osl: 100 };
  const a = mem(M, 48, cfg({ ep: 48, eplb: true }), W).m, b = mem(M, 48, cfg({ ep: 48, eplb: true }), { ...W, isl: 4000 }).m;
  assert.ok(a.perRankMax > 1400 && a.perRankMax < 2400, 'capacity at 2K ' + a.perRankMax);
  assert.ok(b.perRankMax < a.perRankMax * 0.6 && b.perRankMax > 700, 'capacity at 4K ' + b.perRankMax);
});

test('10T MoE at 1M context is capacity-starved but feasible with CP + FP4 experts', () => {
  const M = mk('hypo_10t'), W = { ...W0, isl: 1e6, osl: 1024 };
  const { m } = mem(M, 72, cfg({ tpA: 2, cp: 4, ep: 72, eplb: false }), W);
  assert.ok(m.free > 0, 'weights fit'); assert.ok(m.bMaxRep >= 1 && m.bMaxRep < 200, 'concurrent requests ' + m.bMaxRep);
});

test('weights that do not fit leave no KV budget', () => {
  const M = mk('hypo_10t', { expDtype: 'fp8' }); const { m } = mem(M, 72, cfg({ ep: 72 }));
  assert.equal(m.free, 0);
});
