// Published anchors (PLAN §12.2). Acceptance: within ±30% of measured absolute numbers using only the global efficiency knobs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, run } from './util.mjs';

const ratio = (model, measured) => model / measured;

test('LMSYS GB200 NVL72 Part II — DeepSeek-V3 decode, 48-GPU EP, 2K ISL, batch 1408: 13,386 output tok/s/GPU', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 2000, osl: 100 };
  const P = C.normalize(M, 48, cfg({ ep: 48, eplb: true, overlap: true })), r = C.evalDecode(M, H, W, A0, P, 1408);
  const x = ratio(r.perGpu, 13386); console.log('  decode model/measured =', x.toFixed(2));
  assert.ok(x > 0.7 && x < 1.3, 'ratio ' + x);
  assert.ok(!r.oom, 'batch 1408 fits in memory (LMSYS chose it to fill the KV cache)');
});

test('LMSYS GB200 NVL72 Part II — DeepSeek-V3 prefill, 2 GPUs per instance, 2K ISL: 26,156 input tok/s/GPU', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 2000, osl: 100, chunk: 8192 };
  const best = Math.max(...[cfg({ tpA: 2, ep: 2 }), cfg({ ep: 2 }), cfg({ tpA: 2, ep: 1 })].map((c) => C.evalPrefill(M, H, W, A0, C.normalize(M, 2, c), 2).thr));
  const x = ratio(best, 26156); console.log('  prefill model/measured =', x.toFixed(2));
  assert.ok(x > 0.7 && x < 1.3, 'ratio ' + x);
});

test('DeepSeek production layout: optimizer prefers wide EP + attention-DP for decode, small/medium EP for prefill', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 4096, osl: 512 };
  const best = C.pickTop(run(C.searchDecode(M, H, W, A0, 72, { slo: 20 })).pts, { objective: 'slo', slo: 20 }).top[0];
  assert.ok(best.P.ep >= 36 && best.P.tpA === 1, 'decode: ' + C.cfgLabel(best));
  const pre = C.bestPrefill(M, H, W, A0, 24, 2);
  assert.ok(pre.P.ep <= 24 && pre.P.dpA >= 1);
});

test('TRT-LLM wide-EP trend: per-GPU throughput rises with EP at a fixed high-batch operating point', () => {
  const M = mk('deepseek_v3'), W = { ...W0, isl: 8192, osl: 1024 };
  const thr = (ep) => { const P = C.normalize(M, ep, cfg({ ep, eplb: true, overlap: true })); return C.evalDecode(M, H, W, A0, P, Math.min(256, C.memoryPerGpu(M, H, W, A0, P, P.D).perRankMax)).perGpu; };
  const t = [8, 16, 36, 72].map(thr); console.log('  EP8/16/36/72 tok/s/GPU:', t.map((x) => x.toFixed(0)).join(' / '));
  assert.ok(t[3] > t[0]);
});

test('NVIDIA GB200 NVL72 DeepSeek-V4-Pro: the interactivity axis reaches ≥150 tok/s/user (marketing anchor; loose)', () => {
  const M = mk('deepseek_v4_pro'), W = { ...W0, specGamma: 1, specAlpha: 0.85 };
  const res = C.searchDecode(M, H, W, A0, 72, {}); let s; while (!(s = res.next()).done);
  const maxU = Math.max(...s.value.front.map((p) => p.tokUser)); console.log('  V4-Pro max tok/s/user (MTP1):', maxU.toFixed(0));
  assert.ok(maxU >= 150, 'max interactivity ' + maxU);
});
