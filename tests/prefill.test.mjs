import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near } from './util.mjs';

const pre = (M, gpus, c, W = W0, A = A0) => C.evalPrefill(M, H, W, A, C.normalize(M, gpus, c), gpus);

test('large-batch prefill ≈ FLOPs / (peak · efficiency)', () => {
  const M = mk('deepseek_v3'), D = C.derive(M), W = { ...W0, isl: 8192, chunk: 8192 };
  const r = pre(M, 8, cfg({ ep: 8, eplb: false }), W);
  const flops = 2 * D.active * 8192 * 8, ideal = flops / (8 * H.flops.fp8);      // 8 DP ranks each prefill an 8K prompt; if everything ran at FP8 peak
  assert.ok(r.tPrefill > ideal && r.tPrefill < 6 * ideal, `prefill ${r.tPrefill} vs FP8-peak ideal ${ideal}`);
});

test('prefill time grows super-linearly with prompt length (quadratic attention)', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 8, tpA: 1 });
  const t = (isl) => pre(M, 8, c, { ...W0, isl, chunk: 8192 }).tPrefill;
  assert.ok(t(131072) / t(16384) > 8, 'linear would be 8×: ' + t(131072) / t(16384));
});

test('chunking: same tokens processed; small chunks lower GEMM efficiency; PP adds fill latency but not steady throughput cost', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 8 });
  const big = pre(M, 8, c, { ...W0, isl: 32768, chunk: 16384 }), small = pre(M, 8, c, { ...W0, isl: 32768, chunk: 1024 });
  assert.ok(small.nCh === 32 && big.nCh === 2); assert.ok(small.thr < big.thr * 1.02);
  const p1 = pre(M, 8, cfg({ ep: 8 }), { ...W0, isl: 16384, chunk: 2048 }), p4 = pre(M, 8, cfg({ pp: 4, ep: 2 }), { ...W0, isl: 16384, chunk: 2048 });
  assert.ok(p4.tPrefill > p4.tChunk * p4.nCh, 'fill/drain adds latency'); assert.ok(p4.bubble > 0 && p1.bubble === 0);
});

test('KV hand-off inside the NVL72 domain is negligible vs a NIC', () => {
  const M = mk('deepseek_v3'), r = pre(M, 8, cfg({ ep: 8 }));
  assert.ok(r.kvXfer < r.pack * r.kvBytes / 50e9 / 5, 'NVLink ≥5× faster than a 400 Gb/s NIC: ' + r.kvXfer);
});

test('hybrid attention prefill: linear-ish in length (compressed pools) versus MLA quadratic', () => {
  const V4 = mk('deepseek_v4_flash'), V3 = mk('deepseek_v3'), c = cfg({ ep: 8 });
  const ratio = (M) => { const t = (isl) => pre(M, 8, c, { ...W0, isl, chunk: 8192 }).segs.find((s) => s.key === 'kv').ms; return t(131072) / t(16384); };
  assert.ok(ratio(V4) < ratio(V3), `V4 ${ratio(V4)} vs V3 ${ratio(V3)}`);
});

test('short prompts are packed into one chunk: per-GPU prefill throughput does not collapse below the chunk size', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 8 }), p = (isl) => pre(M, 8, c, { ...W0, isl, chunk: 8192 });
  assert.equal(p(256).pack, 32); assert.equal(p(3000).pack, 2); assert.equal(p(16384).pack, 1);
  assert.ok(p(256).thr > 0.9 * p(2048).thr, `ISL 256 ${p(256).thr} vs ISL 2048 ${p(2048).thr}`);
  assert.ok(p(256).ttft > pre(M, 8, c, { ...W0, isl: 256, chunk: 256 }).ttft, 'packing trades a little TTFT for throughput');
});

test('prefill attention uses its own efficiency knob, not the decode-fitted one', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 8 }), W = { ...W0, isl: 65536, chunk: 8192 }, P = C.normalize(M, 8, c);
  const base = C.evalPrefill(M, H, W, A0, P, 8).tAttn;
  near(C.evalPrefill(M, H, W, { ...A0, attnEff: A0.attnEff / 2 }, P, 8).tAttn, base, 1e-12, 'decode knob does not move prefill');
  assert.ok(C.evalPrefill(M, H, W, { ...A0, attnEffPrefill: A0.attnEffPrefill / 2 }, P, 8).tAttn > 1.2 * base);
});

test('glue kernels: memory-bound activation traffic that grows with the number of passes', () => {
  const M = mk('deepseek_v3'), c = cfg({ ep: 8 }), W = { ...W0, isl: 4096, chunk: 8192 }, P = C.normalize(M, 8, c);
  const g = (glueHid) => C.evalPrefill(M, H, W, { ...A0, glueHid }, P, 8).segs.find((s) => s.key === 'glue').ms;
  assert.ok(g(16) > g(8) && g(8) > g(0) && g(0) > 0, 'expert combine passes remain at glueHid = 0');
});
