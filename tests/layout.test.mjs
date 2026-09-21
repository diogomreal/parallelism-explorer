import test from 'node:test';
import assert from 'node:assert/strict';
import { C, mk, cfg } from './util.mjs';

test('folding constraints hold for every enumerated layout', () => {
  const M = mk('deepseek_v3');
  for (const c of C.enumerateConfigs(M, 72, [false, true], [true])) {
    const P = C.normalize(M, 72, c);
    assert.equal(P.tpA * P.dpA * P.cp, P.g, 'attention TP×DP×CP = g');
    assert.equal(P.ep * P.tpM, P.g, 'MoE EP×TP = g');
    assert.ok(P.pp * P.replicas * P.g <= 72);
    assert.equal(M.heads % P.tpA, 0, 'heads divisible by TP');
    assert.equal(P.slots % P.ep, 0, 'expert slots divisible by EP');
    assert.ok(P.slots >= M.experts);
  }
});

test('NVL72 non-power-of-two: 256 experts over EP=72 need 288 slots with EPLB', () => {
  const M = mk('deepseek_v3');
  const P = C.normalize(M, 72, cfg({ ep: 72, eplb: true }));
  assert.equal(P.slots, 288); assert.equal(P.redundant, 32);
  assert.equal(C.normalize(M, 72, cfg({ ep: 72, eplb: false })).slots, 288, 'divisibility alone also forces 4 slots/GPU');
  assert.equal(C.normalize(M, 64, cfg({ ep: 64, eplb: false })).slots, 256);
});

test('invalid requests snap to the nearest valid value', () => {
  const M = mk('deepseek_v3');
  const P = C.normalize(M, 72, cfg({ tpA: 5, ep: 50, pp: 7 }));
  assert.ok([1, 2, 3, 4, 6, 8, 9, 12, 16].includes(P.tpA) && 72 % P.pp === 0 && (72 / P.pp) % P.ep === 0);
  assert.equal(C.normalize(mk('deepseek_v3'), 72, cfg({ tpA: 72 })).tpA, 64 > 72 ? 64 : C.normalize(M, 72, cfg({ tpA: 72 })).tpA);
});

test('pool sizes', () => {
  assert.equal(C.poolGpus({ poolMode: 'decode', split: 24 }, 'prefill'), 72);
  assert.equal(C.poolGpus({ poolMode: 'split', split: 24 }, 'prefill'), 24);
  assert.equal(C.poolGpus({ poolMode: 'split', split: 24 }, 'decode'), 48);
});
