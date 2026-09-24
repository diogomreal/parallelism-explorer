// Model derivations: parameter counts, active parameters, KV-cache bytes. Pure functions, SI units (bytes).
import { DT_BYTES } from './data.js';

export function derive(M) {
  const d = M.hidden, bW = DT_BYTES[M.wDtype], bE = DT_BYTES[M.expDtype], bKV = DT_BYTES[M.kvDtype];
  const dispatchB = DT_BYTES[M.dispatchDtype || 'fp8'], combineB = DT_BYTES[M.combineDtype || 'bf16'];   // EP all-to-all payloads (DeepEP: FP8 dispatch, BF16 combine)
  const L = M.layers, hyb = M.attn === 'hybrid';
  // hybrid attention (DeepSeek-V4): layer counts from config.json when given, else 2 HCA bootstrap layers then alternating CSA / HCA
  const swaL = hyb ? M.swaLayers || 0 : 0;
  const csaL = hyb ? M.csaLayers ?? Math.ceil((L - 2 - swaL) / 2) : 0, hcaL = hyb ? M.hcaLayers ?? L - swaL - csaL : 0;
  const entryB = hyb ? (M.headDim - M.ropeDim) * bKV + M.ropeDim * 2 : 0;     // one compressed KV entry: quantized nope dims + BF16 rope dims
  const idxB = hyb ? M.idxDim * bKV : 0;                                       // one indexer key
  const win = (n) => Math.min(n, M.window);

  // ---- weights per layer (parameters)
  const attnP = hyb
    ? d * M.qLora + M.qLora * M.heads * M.headDim + 2 * d * M.headDim + M.heads * M.headDim * M.oLora + M.oGroups * M.oLora * d
      + (csaL / L) * (M.qLora * M.idxHeads * M.idxDim + d * M.idxDim)          // + KV compressor, + lightning indexer on CSA layers
    : M.attn === 'mla'
    ? d * M.qLora + M.qLora * M.heads * (M.nope + M.rope) + d * (M.kvLora + M.rope) + M.kvLora * M.heads * (M.nope + M.vDim) + M.heads * M.vDim * d
    : d * M.heads * M.headDim + 2 * d * M.kvHeads * M.headDim + M.heads * M.headDim * d;
  const denseP = 3 * d * M.denseInter;                                          // gated FFN (gate, up, down)
  const expertP = M.moe ? 3 * d * M.expertInter : 0;
  const moeLayers = M.moe ? L - M.denseLayers : 0;
  const denseLayers = L - moeLayers;
  const sharedP = expertP * (M.shared || 0);
  const routedP = expertP * (M.experts || 0);
  const routerP = M.moe ? d * M.experts : 0;
  const embedP = 2 * M.vocab * d;                                               // untied embedding + LM head
  const total = L * attnP + denseLayers * denseP + moeLayers * (routedP + sharedP + routerP) + embedP;
  const active = L * attnP + denseLayers * denseP + moeLayers * (M.topK * expertP + sharedP + routerP) + M.vocab * d;

  // ---- KV cache. kv(n, tpA): per layer (averaged over layer types), one request at context n, before CP sharding.
  //   store = bytes kept in HBM; read = bytes touched per decode step; ent = attended entries; idxFl = indexer FLOPs per query token.
  const kvTokLayer = hyb ? 0 : M.attn === 'mla' ? (M.kvLora + M.rope) * bKV : 2 * M.kvHeads * M.headDim * bKV;
  const mhaEquiv = M.attn === 'mla' ? ((M.nope + M.rope + M.vDim) * M.heads) : 2 * M.heads * M.headDim;
  const kv = (n, tpA = 1) => {
    if (hyb) {
      const csaE = win(n) + Math.min(M.idxTopk, n / M.csaRatio), hcaE = win(n) + n / M.hcaRatio;
      const store = (csaL * ((win(n) + n / M.csaRatio) * entryB + (n / M.csaRatio) * idxB) + hcaL * (win(n) + n / M.hcaRatio) * entryB + swaL * win(n) * entryB) / L;
      const ent = (csaL * csaE + hcaL * hcaE + swaL * win(n)) / L;
      const read = (csaL * (csaE * entryB + (n / M.csaRatio) * idxB) + hcaL * hcaE * entryB + swaL * win(n) * entryB) / L;
      const idxFl = (csaL / L) * 2 * M.idxHeads * M.idxDim * (n / M.csaRatio);
      return { store, read, ent, idxFl };
    }
    const per = M.attn === 'mla' ? (M.kvLora + M.rope) * bKV : 2 * Math.max(1, M.kvHeads / tpA) * M.headDim * bKV;   // MLA latent is not sharded by TP
    return { store: per * n, read: per * n, ent: n, idxFl: 0 };
  };
  return {
    d, bW, bE, bKV, dispatchB, combineB, attnP, denseP, expertP, moeLayers, denseLayers, sharedP, routedP, routerP, embedP, total, active,
    kvTokLayer, kv, kvTok: hyb ? (kv(1e6).store * L) / 1e6 : kvTokLayer * L, hyb, csaL, hcaL, swaL,
    mlaFactor: M.attn === 'mla' ? mhaEquiv / (M.kvLora + M.rope) : hyb ? (mhaEquiv * bKV * 1e6) / kv(1e6).store : null,
    sparsity: M.moe ? M.topK / Math.max(M.experts, 1) : 1,
    weightBytes: (total - moeLayers * routedP) * bW + moeLayers * routedP * bE,
  };
}
