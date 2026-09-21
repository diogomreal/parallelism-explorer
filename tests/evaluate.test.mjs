import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near } from './util.mjs';

const state = (poolMode, over = {}) => ({ poolMode, split: 24, opt: { dbo: true, eplb: true }, par: { decode: cfg({ ep: 72 }), prefill: cfg({ replicas: 3, ep: 24 }) }, ...over });

test('defaults: cost is on ($3.5 / GPU-hour) and finite', () => {
  assert.equal(W0.gpuHr, 3.5);
  const ev = C.evalAll(mk('deepseek_v4_pro'), H, W0, A0, state('decode'));
  assert.ok(ev.costPerM > 0 && Number.isFinite(ev.costPerM)); assert.equal(ev.costUnit, 'output');
  near(ev.costPerM, (3.5 * 72 / 3600) / (ev.dec.perGpu * 72) * 1e6, 1e-9, '$/Mtok out = rack $/s ÷ tokens/s');
});

test('clearing $/GPU-hour hides cost', () => {
  assert.equal(C.evalAll(mk('deepseek_v3'), H, { ...W0, gpuHr: 0 }, A0, state('decode')).costPerM, null);
});

test('decode-only rack: throughput is decode tok/s/GPU, no TTFT', () => {
  const ev = C.evalAll(mk('deepseek_v3'), H, W0, A0, state('decode'));
  assert.equal(ev.ttft, null); assert.equal(ev.outPerGpu, ev.dec.perGpu); assert.equal(ev.dP.gpus, 72);
});

test('prefill-only rack: prices input tokens, TTFT = prefill + KV hand-off', () => {
  const ev = C.evalAll(mk('deepseek_v3'), H, W0, A0, state('prefill'));
  assert.equal(ev.costUnit, 'input'); assert.equal(ev.pP.gpus, 72); assert.equal(ev.outPerGpu, null);
  near(ev.ttft, ev.pre.ttft, 1e-12);
  near(ev.costPerM, (3.5 * 72 / 3600) / (ev.pre.thr * 72) * 1e6, 1e-9);
});

test('split rack: the slower pool sets end-to-end throughput; TTFT adds the first decode step', () => {
  const ev = C.evalAll(mk('deepseek_v3'), H, W0, A0, state('split'));
  near(ev.outPerGpu, Math.min(ev.rp, ev.rd) * W0.osl / 72, 1e-12);
  assert.ok(ev.ttft > ev.pre.ttft); assert.equal(ev.pP.gpus + ev.dP.gpus, 72);
});

test('top prefill layouts respect the latency budget and are sorted by request rate', () => {
  const M = mk('deepseek_v3'), rows = C.topPrefill(M, H, W0, A0, 24, 1);
  assert.ok(rows.length > 0); rows.forEach((r) => assert.ok(r.ttft <= 1));
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i].reqPerS <= rows[i - 1].reqPerS);
  assert.equal(C.topPrefill(M, H, { ...W0, isl: 131072 }, A0, 4, 0.001).length, 0, 'impossible budget → no layout');
});

test('hardware: GB200 NVL72 is the only platform; numbers are dense (not sparse) and per-GPU', () => {
  assert.deepEqual(Object.keys(C.HARDWARE), ['gb200']);
  assert.equal(H.gpus, 72); assert.equal(H.flops.fp8, 5e15); assert.equal(H.flops.fp4, 10e15); assert.equal(H.bw, 8e12); assert.equal(H.link, 900e9);
  near(H.hbmGB * 72 * 1e9, 13.4e12, 0.01, 'rack HBM 13.4 TB');
});
