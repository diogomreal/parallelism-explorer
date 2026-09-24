// KV offload to Grace memory: prefix cache tier, dense spill, HiSparse-style sparse offload (docs/validation.md "KV offload").
import test from 'node:test';
import assert from 'node:assert/strict';
import { C, H, A0, W0, mk, cfg, near } from './util.mjs';

const A = { ...A0, memUtil: 0.72 };
const V3 = mk('deepseek_v3'), FLASH = mk('deepseek_v4_flash'), PRO = mk('deepseek_v4_pro');
const setup = (M, gpus, W) => { const P = C.normalize(M, gpus, cfg({ ep: gpus, eplb: gpus > 1 })); return { P, mem: C.memoryPerGpu(M, H, W, A, P, P.D) }; };
const dec = (M, gpus, W, ba) => C.evalDecode(M, H, W, A, setup(M, gpus, W).P, ba);
const W128 = { ...W0, isl: 128000, osl: 8000 };

test('host bandwidth per GPU: C2C and LPDDR5X are shared by the two GPUs of a Grace', () => {
  near(C.hostBw(H, { hostBwEff: 1 }), Math.min(450e9, 512e9) / 2, 1e-12);
  assert.ok(C.hostBw(H, A0) < H.bw / 30, 'the host link is well over an order of magnitude slower than HBM');
});

test('offload is off by default and leaves every result unchanged', () => {
  assert.equal(W0.kvOffload, 'none'); assert.equal(W0.prefixHit, 0);
  const a = dec(V3, 8, W128, 10), b = dec(V3, 8, { ...W128, kvOffload: 'none', prefixHit: 0 }, 10);
  near(a.tpot, b.tpot, 1e-12); assert.equal(a.offload, null);
});

test('spill: nothing moves until HBM is full; beyond it capacity grows and every spilled byte streams over C2C each step', () => {
  const W = { ...W128, kvOffload: 'spill' }, { mem } = setup(V3, 8, W), base = setup(V3, 8, W128).mem;
  assert.equal(mem.hbmMax, base.perRankMax);
  assert.equal(mem.perRankMax, Math.floor((mem.free + A.hostGB * 1e9) / mem.kvReq), 'HBM + Grace memory');
  near(dec(V3, 8, W, mem.hbmMax).tpot, dec(V3, 8, W128, mem.hbmMax).tpot, 1e-12, 'within HBM: identical');
  const over = dec(V3, 8, W, mem.hbmMax * 2), host = over.ops.find((o) => o.cat === 'host');
  assert.ok(host && host.bytes > 0 && over.segs.some((s) => s.key === 'host'));
  assert.ok(over.offload.hostFrac > 0.4 && over.offload.hostFrac < 0.6, 'about half the KV is in host at 2× the HBM batch');
});

test('spill does not pay for dense attention at long context: more requests, less throughput (why production does not do it)', () => {
  const W = { ...W128, kvOffload: 'spill' }, cap = setup(V3, 8, W128).mem.perRankMax;
  assert.ok(dec(V3, 8, W, 2 * cap).perGpu < dec(V3, 8, W128, cap).perGpu);
});

test('spill picks its mechanism from the attention: top-k swap-in for sparse (CSA) models, per-step streaming for dense ones', () => {
  assert.equal(setup(V3, 8, { ...W128, kvOffload: 'spill' }).mem.mode, 'spill');
  assert.equal(setup(PRO, 16, { ...W128, kvOffload: 'spill' }).mem.mode, 'sparse');
  assert.equal(setup(PRO, 16, { ...W128, kvOffload: 'sparse' }).mem.mode, 'sparse', 'old share links with "sparse" still work');
});

test('sparse-attention spill: engages only past HBM; then CSA entries move to host, HBM keeps the hot buffer; swap-in cost is serial and grows with misses', () => {
  const W = { ...W0, isl: 1040000, osl: 8000, kvOffload: 'spill' }, { mem } = setup(PRO, 16, W), base = setup(PRO, 16, { ...W, kvOffload: 'none' }).mem;
  assert.equal(mem.hbmMax, base.perRankMax);
  near(dec(PRO, 16, W, mem.hbmMax).tpot, dec(PRO, 16, { ...W, kvOffload: 'none' }, mem.hbmMax).tpot, 1e-12, 'within HBM: identical');
  assert.ok(mem.kvHost > 0.5 * mem.kvReq, 'at 1M context most of the KV is compressed CSA entries');
  assert.ok(mem.kvDev < 0.5 * base.kvDev && mem.perRankMax > 2 * base.perRankMax);
  const ba = mem.hbmMax * 2, t = (sparseMiss) => C.evalDecode(PRO, H, W, { ...A, sparseMiss }, setup(PRO, 16, W).P, ba).tpot;
  assert.ok(t(0.3) > t(0.1) && t(0.1) > t(0));
  const d0 = C.evalDecode(PRO, H, W, { ...A, sparseMiss: 0 }, setup(PRO, 16, W).P, ba);
  assert.equal(d0.segs.find((s) => s.key === 'host')?.ms ?? 0, 0, 'no misses: no host traffic');
});

test('sparse-attention spill is minimal: only the overflow requests move to LPDDR5X, so HBM stays full and swap-in scales with them', () => {
  const W = { ...W0, isl: 623616, kvOffload: 'spill' }, P = C.normalize(PRO, 72, cfg({ ep: 72, eplb: true })), mem = C.memoryPerGpu(PRO, H, W, A0, P, P.D);
  const save = mem.kvReq - mem.kvDev, hostReqs = Math.floor(mem.hostCap / mem.kvHost);
  assert.equal(mem.perRankMax, Math.floor(Math.min(mem.free / mem.kvDev, (mem.free + hostReqs * save) / mem.kvReq)));
  assert.ok(mem.perRankMax > hostReqs, 'more requests than LPDDR5X alone could hold: the rest stay fully in HBM');
  assert.equal(mem.offloadedFor(mem.hbmMax), 0);
  assert.equal(mem.offloadedFor(mem.hbmMax + 1), Math.ceil(((mem.hbmMax + 1) * mem.kvReq - mem.free) / save));
  assert.ok(mem.kvUsedFor(mem.perRankMax) > 0.97 * mem.free && mem.kvUsedFor(mem.perRankMax) <= mem.free + 1, 'HBM is used, not left idle');
  assert.ok(!C.evalDecode(PRO, H, W, A0, P, mem.perRankMax).oom && C.evalDecode(PRO, H, W, A0, P, mem.perRankMax + 1).oom);
  const t = (ba) => C.evalDecode(PRO, H, W, A0, P, ba).tpot, n = C.evalDecode(PRO, H, { ...W, kvOffload: 'none' }, A0, P, mem.hbmMax).tpot;
  assert.ok(t(mem.hbmMax + 1) / n < 1.05, 'no jump when spilling starts: one request offloaded, one request swapping in');
});

test('HiSparse trend (LMSYS DeepSeek-V4 day 0: "up to 3x" throughput, V4-Flash 200K/20K on 2 GPUs): 1.5–5× at the capacity-limited batch', () => {
  const W = { ...W0, isl: 200000, osl: 20000 }, thr = (mode) => { const w = { ...W, kvOffload: mode }; return dec(FLASH, 2, w, setup(FLASH, 2, w).mem.perRankMax).perGpu; };
  const x = thr('spill') / thr('none'); console.log('  V4-Flash 200K/20K spill (sparse) throughput gain:', x.toFixed(2));
  assert.ok(x > 1.5 && x < 5, 'gain ' + x);
});

test('auto-tune chooses the KV placement: spill for a capacity-bound sparse-attention model, HBM only for dense attention', () => {
  const run = (g) => { let r; while (!(r = g.next()).done); return r.value; };
  const flash = run(C.autoTuneDecode(FLASH, H, { ...W0, isl: 200000, osl: 20000, minTokUser: 10 }, A, 8, { slo: 10 }));
  assert.equal(flash.kvOffload, 'spill'); assert.equal(flash.point.kv, 'spill'); assert.ok(flash.kvGain > 1.02);
  const v3 = run(C.autoTuneDecode(V3, H, { ...W128, minTokUser: 10, kvOffload: 'spill' }, A, 8, { slo: 10 }));
  assert.equal(v3.kvOffload, 'none', 'dense spill never beats HBM only here, even when the user had spill selected');
  const split = run(C.autoTuneSplit(FLASH, H, { ...W0, isl: 200000, osl: 20000, minTokUser: 10, ttftMs: 60000 }, A, { sizes: [16, 24] }));
  assert.ok(split && ['none', 'spill'].includes(split.row.dec.kv));
});

test('prefix cache: cached tokens skip compute, prefill throughput counts the whole prompt, load hides behind compute from Grace memory', () => {
  const P = C.normalize(V3, 4, cfg({ pp: 4 })), W = { ...W128, chunk: 16384 };
  const pre = (w) => C.evalPrefill(V3, H, { ...W, ...w }, A0, P, 4);
  const p0 = pre({}), p5 = pre({ prefixHit: 0.5 }), p9 = pre({ prefixHit: 0.9 });
  near(pre({ prefixHit: 0 }).ttft, p0.ttft, 1e-12);
  assert.ok(p9.ttft < p5.ttft && p5.ttft < p0.ttft && p9.thr > p5.thr && p5.thr > p0.thr);
  assert.ok(p9.prefix.tLoad < 0.1 * p9.tPrefill, 'Grace memory load is hidden');
  const slow = pre({ prefixHit: 0.9, prefixTier: 'storage' }), crawl = C.evalPrefill(V3, H, { ...W, prefixHit: 0.9, prefixTier: 'storage' }, { ...A0, storageGBs: 0.2 }, P, 4);
  assert.ok(slow.prefix.tLoad > p9.prefix.tLoad && crawl.ttft > p9.ttft, 'a slow storage tier can make the load, not the compute, the bottleneck');
  assert.ok(crawl.segs.find((s) => s.key === 'host').ms > 1);
});
