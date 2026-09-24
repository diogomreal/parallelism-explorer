// Fit the efficiency knobs to the published anchors (tests/anchors.data.mjs).
//   node scripts/calibrate.mjs          fit, then print every anchor and the leave-one-out error
// Objective (log space): Σ over calibration anchors (ln model/measured)² / σ_m²  +  Σ over knobs (ln k/prior)² / σ_k².
// The priors come from public kernel benchmarks, so the anchors only move a knob as far as they actually constrain it.
import { ANCHORS } from '../tests/anchors.data.mjs';
import { A0 } from '../tests/util.mjs';

const SIGMA_M = 0.1;                          // per-anchor uncertainty: measurement noise + unstated setup details
export const PRIORS = {
  gemmEff: { prior: 0.6, sigma: 0.25, lo: 0.3, hi: 0.9, why: 'FP8 blockwise GEMM on Blackwell (DeepGEMM / cuBLAS class), minus sustained-clock losses' },
  moeEff: { prior: 0.55, sigma: 0.25, lo: 0.25, hi: 0.9, why: 'NVFP4 grouped GEMM (CUTLASS / CuTe DSL); no public roofline fraction' },
  attnEff: { prior: 0.3, sigma: 0.5, lo: 0.05, hi: 0.9, why: 'FP8-math MLA decode: no public number; FlashMLA BF16-math reaches ≈ 0.31 of BF16 peak on B200' },
};
const KNOBS = Object.keys(PRIORS);

const ratio = (a, A) => a.run(A).value / a.measured;
function loss(A, cal) {
  let s = 0;
  for (const a of cal) s += (Math.log(ratio(a, A)) / SIGMA_M) ** 2;
  for (const k of KNOBS) s += (Math.log(A[k] / PRIORS[k].prior) / PRIORS[k].sigma) ** 2;
  return s;
}
// Coordinate descent with a golden-section line search per knob (log space). Few knobs, smooth objective: converges in a handful of sweeps.
export function fit(cal, A = { ...A0 }, sweeps = 6) {
  A = { ...A };
  const g = (Math.sqrt(5) - 1) / 2;
  for (let s = 0; s < sweeps; s++) for (const k of KNOBS) {
    let a = Math.log(PRIORS[k].lo), b = Math.log(PRIORS[k].hi);
    const f = (x) => loss({ ...A, [k]: Math.exp(x) }, cal);
    let c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
    for (let i = 0; i < 24; i++) { if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); } }
    A[k] = Math.exp((a + b) / 2);
  }
  return A;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const usable = ANCHORS.filter((a) => a.run), cal = usable.filter((a) => a.status === 'calibration');
  const A = fit(cal);
  console.log('Fitted knobs (prior → fit):');
  for (const k of KNOBS) console.log(`  ${k.padEnd(9)} ${PRIORS[k].prior.toFixed(2)} → ${A[k].toFixed(3)}   (${PRIORS[k].why})`);
  console.log('\nAnchors (model / measured):');
  for (const a of usable) { const r = a.run(A); console.log(`  ${a.status.padEnd(11)} ${a.id.padEnd(22)} ${(r.value / a.measured).toFixed(2)}   ${r.detail}`); }
  console.log('\nLeave-one-out (refit without the anchor, then predict it):');
  for (const a of cal) { const Al = fit(cal.filter((x) => x !== a)); console.log(`  ${a.id.padEnd(22)} ${ratio(a, Al).toFixed(2)}   (${KNOBS.map((k) => `${k} ${Al[k].toFixed(2)}`).join(', ')})`); }
}
