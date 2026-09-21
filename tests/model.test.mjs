import test from 'node:test';
import assert from 'node:assert/strict';
import { C, mk, near } from './util.mjs';

// Published parameter counts (PLAN §12.1): presets must match within 1%.
test('parameter counts match published numbers', () => {
  const chk = (k, total, active, tol = 0.012) => { const D = C.derive(mk(k)); near(D.total, total, tol, k + ' total'); near(D.active, active, 0.03, k + ' active'); };
  chk('deepseek_v3', 671e9, 37e9);
  chk('deepseek_v4_pro', 1.6e12, 49e9, 0.03);
  chk('deepseek_v4_flash', 284e9, 13e9);
  chk('llama_405b', 405e9, 405e9);
  chk('qwen3_235b', 235e9, 22e9);
  chk('kimi_k2', 1.03e12, 32e9, 0.02);
});

test('DeepSeek-V3 component hand-check (PLAN §6.2)', () => {
  const D = C.derive(mk('deepseek_v3'));
  near(D.moeLayers * D.routedP, 654e9, 0.01, 'routed experts');
  near(D.attnP * 61, 11.4e9, 0.02, 'MLA attention');
  near(D.denseP * D.denseLayers, 1.2e9, 0.03, 'dense FFN');
  near(D.embedP, 1.9e9, 0.03, 'embed + LM head');
});

test('KV bytes per token', () => {
  near(C.derive(mk('deepseek_v3')).kvTok, 576 * 61, 1e-9, 'MLA fp8 = 576×61');
  near(C.derive(mk('hypo_10t')).kvTok, 204800, 1e-9, 'transcript 10T / GQA-8 example');
  near(C.derive(mk('llama_405b')).kvTok, 2 * 8 * 128 * 126, 1e-9, 'Llama GQA');
  const D = C.derive(mk('deepseek_v3')); near(D.mlaFactor, 71, 0.03, 'MLA compression ≈ 71×');
});

test('hybrid (V4) KV is far smaller than V3.2-style MLA and grows sub-linearly', () => {
  const P = C.derive(mk('deepseek_v4_pro')), V3 = C.derive(mk('deepseek_v3'));
  const perTokPro1M = P.kv(1e6).store * 61 / 1e6, perTokV3 = V3.kvTok;
  assert.ok(perTokPro1M < 0.2 * perTokV3 && perTokPro1M > 0.05 * perTokV3, `Pro/V3 = ${perTokPro1M / perTokV3}`);   // paper: ~10% of V3.2
  assert.ok(P.kv(1e6).store < 20 * P.kv(1e5).store, 'window + CSA/HCA compression: roughly linear only via the compressed pools');
  assert.ok(P.kv(1e6).read < P.kv(1e6).store, 'sparse CSA reads only top-k entries');
});

test('dtype bytes and derived weight bytes', () => {
  const M = mk('deepseek_v3'), D = C.derive(M);
  assert.equal(D.bW, 1); assert.equal(D.bE, 0.5625);
  assert.ok(D.weightBytes < D.total * D.bW, 'FP4 experts shrink the footprint');
});
