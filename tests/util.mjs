import * as C from '../site/js/core/index.js';
export { C };
export const H = C.HARDWARE.gb200;
export const A0 = { ...C.ASSUMP_DEFAULT };
export const mk = (k, over = {}) => ({ qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128, ...C.MODELS[k], ...over });
export const W0 = { ...C.WORKLOAD_DEFAULT };
export const cfg = (o = {}) => ({ pp: 1, replicas: 1, tpA: 1, cp: 1, ep: 1, microbatches: 1, overlap: false, eplb: false, ...o });
export const near = (a, b, rel, msg) => { if (!(Math.abs(a - b) <= Math.abs(b) * rel)) throw new Error(`${msg || ''} ${a} not within ${rel * 100}% of ${b}`); };
export const run = (gen) => { let s; while (!(s = gen.next()).done); return s.value; };
