// α–β collective cost model (PLAN §6.5). SI units. bw = per-GPU one-direction link bandwidth × efficiency.
// Latency multipliers per collective are relative to the base α in the assumptions panel.
export function collective(kind, bytes, n, H, A) {
  if (n <= 1 && kind !== 'p2p') return 0;
  const alpha = A.alphaUs * 1e-6, bw = H.link * A.linkEff;
  switch (kind) {
    case 'ar': {           // all-reduce: min(ring, one-shot / NVLS-style)
      const ring = 2 * (n - 1) * 0.4 * alpha + (2 * (n - 1) / n) * bytes / bw;
      const oneShot = 0.8 * alpha + bytes / bw;
      return Math.min(ring, oneShot);
    }
    case 'rsag': return (n - 1) * 0.4 * alpha + ((n - 1) / n) * bytes / bw;   // reduce-scatter or all-gather
    case 'a2a': return 1.5 * alpha + bytes / bw;                              // bytes = data each GPU sends to others
    case 'p2p': return 0.5 * alpha + bytes / bw;
    default: throw new Error('unknown collective ' + kind);
  }
}

// Expected number of distinct remote EP ranks a token reaches when routed to k experts spread over ep ranks (dedup: one send per destination).
export const distinctDest = (ep, k) => (ep <= 1 ? 0 : ep * (1 - Math.pow(1 - 1 / ep, k)) * (ep - 1) / ep);
