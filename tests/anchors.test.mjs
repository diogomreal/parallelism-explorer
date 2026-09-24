// Published anchors (PLAN §12.2). Acceptance: within ±30% of measured absolute numbers using only the global efficiency knobs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, run } from './util.mjs';

import { ANCHORS } from './anchors.data.mjs';

// Absolute anchors (tests/anchors.data.mjs). Calibration and validation anchors must sit within ±30% (or their own `tol`);
// known gaps are reported as TODO so the miss stays visible without failing the suite.
for (const a of ANCHORS) {
  const name = `${a.id}: ${a.measured} ${a.unit} [${a.status}]`;
  if (!a.run) { test(name, { skip: a.why }, () => {}); continue; }
  test(name, a.status === 'gap' ? { todo: a.why } : {}, () => {
    const r = a.run(A0), x = r.value / a.measured, tol = a.tol ?? 0.3;
    console.log(`  ${a.id}: model/measured = ${x.toFixed(2)} (${r.detail})`);
    assert.ok(x > 1 - tol && x < 1 + tol, 'ratio ' + x.toFixed(3));
  });
}

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

test('defaults are the current calibration: re-run `npm run calibrate` after changing the model or the anchors', async () => {
  const { fit } = await import('../scripts/calibrate.mjs');
  const A = fit(ANCHORS.filter((a) => a.run && a.status === 'calibration'));
  for (const k of ['gemmEff', 'moeEff', 'attnEff']) assert.ok(Math.abs(A[k] / A0[k] - 1) < 0.05, `${k}: default ${A0[k]}, fit ${A[k].toFixed(3)}`);
});
