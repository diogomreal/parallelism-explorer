// End-to-end evaluation of the current serving plan (one decode pool, one prefill pool, or both on a split rack).
import { normalize, poolGpus } from './layout.js';
import { evalDecode } from './decode.js';
import { evalPrefill } from './prefill.js';
import { derive } from './model.js';

export function evalAll(M, H, W, A, state) {
  const D = derive(M);
  const dGpus = poolGpus(state, 'decode'), pGpus = poolGpus(state, 'prefill');
  const dP = normalize(M, dGpus, { ...state.par.decode, overlap: state.opt.dbo, eplb: state.opt.eplb }, D);
  const pP = normalize(M, pGpus, { ...state.par.prefill, overlap: state.opt.dbo, eplb: state.opt.eplb }, D);
  const dec = evalDecode(M, H, W, A, dP, Math.max(1, Math.round(Math.min(W.batchPerRank, 4096))));
  const pre = evalPrefill(M, H, W, A, pP, pGpus, dP.tpA * dP.cp);
  const pm = state.poolMode;
  const tpGpuSec = W.isl / pre.thr;                                    // prefill GPU-seconds per request
  const tdGpuSec = W.osl / Math.max(dec.perGpu, 1e-9);                // decode GPU-seconds per request
  let outPerGpu = null, rp = null, rd = null;
  if (pm === 'split') { rp = pre.reqPerS; rd = dec.total / W.osl; outPerGpu = Math.min(rp, rd) * W.osl / 72; }   // steady state, rate-matched: the slower pool sets the request rate
  else if (pm === 'decode') outPerGpu = dec.perGpu;                    // whole rack decodes: decode-only tokens/s/GPU
  const ttft = pm === 'decode' ? null : pm === 'prefill' ? pre.ttft : pre.ttft + dec.stepMs / 1e3 + 0.010;   // + first decode step + 10 ms queueing headroom
  // $ per million tokens: output tokens for decode / split, input tokens for a prefill rack. Only when $/GPU-hr is set.
  const rackPerSec = (W.gpuHr || 0) * 72 / 3600, tokPerSec = pm === 'prefill' ? pre.thr * 72 : outPerGpu * 72;
  const costPerM = W.gpuHr > 0 ? rackPerSec / Math.max(tokPerSec, 1e-9) * 1e6 : null;
  return { dec, pre, dP, pP, outPerGpu, rp, rd, tpGpuSec, tdGpuSec, ttft, costPerM, costUnit: pm === 'prefill' ? 'input' : 'output', ok: !dec.oom, meetsSlo: dec.tokUser >= W.minTokUser && (ttft == null || ttft * 1e3 <= W.ttftMs) };
}
