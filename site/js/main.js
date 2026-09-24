import { HARDWARE, MODELS, HYBRID_DEFAULT, WORKLOAD_DEFAULT, ASSUMP_DEFAULT, derive, normalize, poolGpus, evalAll, searchDecode, pickTop, robustness, optimizeSplit, topPrefill, autoTuneDecode, autoTunePrefill, autoTuneSplit, KNEE_EPS, family, cfgLabel } from './core/index.js';
import { fmt, esc } from './charts.js';
import { decodeC, prefillC, paretoC, sweepsC, optimizerC, computeRegime, mathFor } from './panes.js';

// ------------------------------------------------------------------ state
const fullModel = (k) => ({ qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128, dispatchDtype: 'fp8', combineDtype: 'bf16', ...MODELS[k] });
const DEFAULT_PAR = () => ({
  decode: { pp: 1, replicas: 1, tpA: 1, cp: 1, ep: 72, microbatches: 1 },
  prefill: { pp: 1, replicas: 3, tpA: 1, cp: 1, ep: 24, microbatches: 1 },
});
const S = {
  theme: 'auto', modelKey: 'deepseek_v4_pro', model: fullModel('deepseek_v4_pro'), hwKey: 'gb200',   // GB200 NVL72 is the only hardware for now
  W: { ...WORKLOAD_DEFAULT }, A: { ...ASSUMP_DEFAULT },
  poolMode: 'decode', open: { model: false, hw: false, wl: false }, csub: { decode: 'overview' },
  split: 24, target: 'decode',      // poolMode: 'decode' | 'prefill' (whole rack serves one phase) | 'split' (shared rack); split = prefill GPUs
  par: DEFAULT_PAR(), opt: { dbo: true, eplb: true },
  tab: 'decode',
  search: { key: '', status: 'idle', progress: 0, pts: [], front: [] }, regime: null,
  optim: { objective: 'slo', minGpu: 800, running: false, fake: 0, results: null },
};
let R = null;   // computed results bundle

const ICON = {
  logo: '<svg viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="var(--accent)"/><path d="M8 21l5-10 4 7 3-5 4 8" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  info: '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/></svg>',
  warn: '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v4M12 17.5h.01"/></svg>',
  sun: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  link: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1"/></svg>',
  check: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>',
  moon: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 13A9 9 0 1111 3a7 7 0 0010 10z"/></svg>',
  auto: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 3v18" /><path d="M12 3a9 9 0 010 18z" fill="currentColor"/></svg>',
};

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
const gpusOf = (pool) => poolGpus(S, pool);
const cfgOf = (pool) => ({ ...S.par[pool], overlap: S.opt.dbo, eplb: S.opt.eplb });
const pick = (P) => ({ pp: P.pp, replicas: P.replicas, tpA: P.tpA, cp: P.cp, ep: P.ep, microbatches: P.microbatches });
const MB_LIST = [1, 2, 3, 4, 6, 8, 12, 16];
function normTarget() { return normalize(S.model, gpusOf(S.target), cfgOf(S.target)); }
function renormPars() {
  ['decode', 'prefill'].forEach((p) => { S.par[p] = pick(normalize(S.model, gpusOf(p), cfgOf(p))); });
}
function compute() { const M = S.model, H = HARDWARE[S.hwKey]; R = { M, H, W: S.W, A: S.A, D: derive(M), ev: evalAll(M, H, S.W, S.A, S) }; }

// ------------------------------------------------------------------ persistence (URL hash + theme)
function saveHash() {
  clearTimeout(saveHash.t);
  saveHash.t = setTimeout(() => { try { const o = { k: S.modelKey, m: S.model, h: S.hwKey, w: S.W, a: S.A, sp: S.split, par: S.par, opt: S.opt, tab: S.tab, pm: S.poolMode, cs: S.csub }; history.replaceState(null, '', '#' + btoa(unescape(encodeURIComponent(JSON.stringify(o))))); } catch (e) { /* ignore */ } }, 300);
}
function loadHash() {
  try {
    if (!location.hash.slice(1)) return;
    const o = JSON.parse(decodeURIComponent(escape(atob(location.hash.slice(1)))));
    S.modelKey = o.k; S.model = { ...fullModel(o.k), ...o.m }; S.W = { ...S.W, ...o.w, osl: WORKLOAD_DEFAULT.osl }; delete S.W.prefixHit;   // output length and prefix hit are fixed; ignore them in old links
     S.A = { ...S.A, ...o.a }; S.split = o.sp; if (o.pm) S.poolMode = o.pm; if (o.cs) S.csub = { ...S.csub, ...o.cs }; S.par = { ...S.par, ...o.par }; S.opt = { ...S.opt, ...o.opt }; S.tab = o.tab || 'decode';
  } catch (e) { /* ignore bad hash */ }
}
function setTheme(t) { S.theme = t; if (t === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', t); try { localStorage.setItem('pe-theme', t); } catch (e) { /* ignore */ } renderBrand(); }

// ------------------------------------------------------------------ brand strip (top of the sidebar; there is no top bar)
function renderBrand() {
  $('#brand').innerHTML = `
    <div class="b1">${ICON.logo}<span class="bname">Parallelism Explorer</span><span class="spacer"></span>
      <button class="btn iconbtn sm" data-act="share" data-tip="Copy a link to this exact setup" aria-label="Copy share link">${ICON.link}</button>
      <button class="btn iconbtn sm" data-act="theme" data-tip="Theme: ${S.theme}" aria-label="Toggle theme">${S.theme === 'dark' ? ICON.moon : S.theme === 'light' ? ICON.sun : ICON.auto}</button></div>`;
}

// ------------------------------------------------------------------ sidebar
const numF = (key, label, step) => `<div class="f"><label>${label}</label><input type="number" data-m="${key}" value="${S.model[key]}" step="${step || 1}" min="${key === 'layers' || key === 'hidden' || key === 'heads' ? 1 : 0}"></div>`;
const selF = (key, label, opts, extra = '') => `<div class="f"><label>${label}</label><select data-ms="${key}" ${extra}>${opts.map(([v, l]) => `<option value="${v}" ${S.model[key] === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>`;
const DT = [['bf16', 'BF16'], ['fp8', 'FP8'], ['fp4', 'NVFP4']];
function sliderF(obj, key, label, min, max, step, fmtv, log, tip) {
  const v = obj[key];
  return `<div class="f sl"${tip ? ` data-tip="${esc(tip)}"` : ''}><label>${label}</label><span class="v" id="v-${obj === S.W ? 'w' : 'a'}-${key}">${fmtv(v)}</span><input type="range" data-${obj === S.W ? 'ws' : 'as'}="${key}" data-log="${log ? 1 : 0}" min="${min}" max="${max}" step="${step}" value="${log ? Math.log2(v) : v}"></div>`;
}
const sec = (id, title, body) => `<details class="sec" data-sec="${id}" ${S.open[id] ? 'open' : ''}><summary><span class="stt">${title}</span><span class="num" id="sum-${id}"></span></summary><div class="bd">${body}</div></details>`;
function updateSideSums() {
  const D = derive(S.model), W = S.W, set = (id, t) => { const e = $('#sum-' + id); if (e) { e.textContent = t; e.title = t; } };
  set('model', `${S.model.name} · ${fmt.n(D.total, D.total >= 1e12 ? 1 : 0)}${S.model.moe ? ' / ' + fmt.n(D.active, 0) + ' active' : ''}`);
  set('hw', HARDWARE[S.hwKey].name);
  set('wl', `≥${W.minTokUser} tok/s/user · TTFT ≤${W.ttftMs} ms`);
}
function renderSide() {
  const M = S.model, H = HARDWARE[S.hwKey];
  $('#sidesecs').innerHTML =
  sec('model', 'Model', `
    <div class="f wide"><select class="sel" data-preset="1" style="width:100%">${Object.entries(MODELS).map(([k, m]) => `<option value="${k}" ${k === S.modelKey ? 'selected' : ''}>${m.name}</option>`).join('')}</select></div>
    <div class="sub-hd">Backbone</div>${numF('layers', 'Layers')}${numF('hidden', 'Hidden size')}${numF('vocab', 'Vocabulary')}
    <div class="sub-hd">Attention</div>${selF('attn', 'Kind', [['mla', 'MLA (latent KV)'], ['gqa', 'GQA / MHA'], ['hybrid', 'Hybrid CSA+HCA (V4)']])}${numF('heads', 'Query heads')}
    ${M.attn === 'gqa' ? numF('kvHeads', 'KV heads') + numF('headDim', 'Head dim')
      : M.attn === 'hybrid' ? numF('headDim', 'Head dim (shared K=V)') + numF('ropeDim', 'RoPE dim') + numF('qLora', 'Q latent rank') + numF('oGroups', 'Output groups') + numF('oLora', 'Output rank / group') + numF('csaRatio', 'CSA: tokens per KV entry') + numF('hcaRatio', 'HCA: tokens per KV entry') + numF('idxTopk', 'CSA indexer top-k') + numF('window', 'Sliding window')
      : numF('kvLora', 'KV latent rank') + numF('rope', 'RoPE dim') + numF('qLora', 'Q latent rank')}
    <div class="sub-hd">Feed-forward</div>${selF('moe', 'Type', [[true, 'Mixture of experts'], [false, 'Dense']], 'data-bool="1"')}
    ${numF('denseInter', 'Dense FFN size')}${M.moe ? numF('denseLayers', 'First-k dense layers') + numF('experts', 'Routed experts') + numF('topK', 'Active per token (top-k)') + numF('expertInter', 'Expert FFN size') + numF('shared', 'Shared experts') : ''}
    <div class="sub-hd">Precision</div>${selF('wDtype', 'Attention / dense', DT)}${M.moe ? selF('expDtype', 'Experts', DT) : ''}${selF('kvDtype', 'KV cache', DT)}${M.moe ? selF('dispatchDtype', 'EP dispatch', DT) : ''}
    <div class="derived" id="derived"></div>${M.notes ? `<div class="hint" style="margin-top:8px">${esc(M.notes)}</div>` : ''}`) +

  sec('hw', 'Hardware', `
    <div class="derived"><span>HBM / GPU</span><span>${H.hbmGB} GB</span><span>HBM bandwidth</span><span>${H.bw / 1e12} TB/s</span><span>FP8 dense</span><span>${H.flops.fp8 / 1e15} PF/s</span><span>FP4 dense</span><span>${H.flops.fp4 / 1e15} PF/s</span><span>NVLink / GPU</span><span>${H.link / 1e9} GB/s ea. way</span><span>GPUs</span><span>${H.gpus} (18 trays × 4)</span><span>Rack HBM</span><span>${fmt.bytes(H.hbmGB * 72 * 1e9)}</span></div>
    <details style="margin-top:10px"><summary class="sub-hd" style="cursor:pointer;margin:6px 0">Assumptions <span class="prov">assumed</span></summary>
      ${sliderF(S.A, 'gemmEff', 'GEMM efficiency (peak)', 0.3, 0.95, 0.01, (v) => v.toFixed(2))}
      ${sliderF(S.A, 'moeEff', 'Expert GEMM efficiency', 0.2, 0.95, 0.01, (v) => v.toFixed(2))}
      ${sliderF(S.A, 'hbmEff', 'HBM efficiency', 0.5, 0.98, 0.01, (v) => v.toFixed(2))}
      ${sliderF(S.A, 'attnEff', 'Attention math eff. (decode)', 0.05, 0.8, 0.01, (v) => v.toFixed(2), false, 'Tensor-core efficiency of the decode attention kernel, as a fraction of the peak for its math dtype (FP8 when the KV cache is FP8). Decode attention takes the slowest of this, KV streaming at the KV-bandwidth efficiency, and the softmax exp rate. Fitted to the LMSYS anchors: no public FP8-math MLA decode benchmark exists (FlashMLA’s BF16-math kernel reaches ≈ 0.31 of the BF16 peak on B200).')}
      ${sliderF(S.A, 'attnEffPrefill', 'Attention math eff. (prefill)', 0.3, 1, 0.01, (v) => v.toFixed(2), false, 'Tensor-core efficiency of the prefill (FlashAttention-style) kernel. Prefill attention also pays one softmax exp per score on the SFUs, which on GB200 takes longer than the FP8 matmuls for MLA; together they reproduce LMSYS’s 128K FMHA timings on GB200 and GB300.')}
      ${sliderF(S.A, 'kvBwEff', 'KV streaming bandwidth eff.', 0.4, 1, 0.01, (v) => v.toFixed(2), false, 'Fraction of HBM bandwidth an attention kernel reaches while streaming the KV cache (≈ 0.78 from FlashInfer’s trtllm-gen MLA decode latency vs context). Sets decode speed at long context when attention is memory-bound.')}
      ${sliderF(S.A, 'glueHid', 'Glue passes / token / layer', 0, 30, 1, (v) => v.toFixed(0), false, 'Memory-bound elementwise traffic: how many times each token’s BF16 hidden vector is read or written per layer by norms, residual adds, activation quantization and RoPE (≈ 8 with fused add+RMSNorm). Routed experts add 2 more passes per (token, expert) for the FC2 output and combine.')}
      ${sliderF(S.A, 'overlap', 'Comm/compute overlap', 0, 1, 0.05, (v) => v.toFixed(2))}
      ${sliderF(S.A, 'alphaUs', 'Collective latency α', 2, 40, 1, (v) => v + ' µs')}
      ${sliderF(S.A, 'overheadMs', 'Per-step framework overhead', 0, 2, 0.05, (v) => v.toFixed(2) + ' ms')}
      ${sliderF(S.A, 'floorUs', 'Per-layer kernel floor', 0, 100, 1, (v) => v + ' µs')}
      ${sliderF(S.A, 'linkEff', 'NVLink efficiency', 0.3, 1, 0.01, (v) => v.toFixed(2))}
    </details>`) +

  sec('wl', 'Workload &amp; SLOs', `
    <div class="sub-hd">Targets</div>
    <div class="f"><label>Min tokens/s/user</label><input type="number" data-w="minTokUser" value="${S.W.minTokUser}" min="1"></div>
    <div class="f"><label>Max TTFT (ms)</label><input type="number" data-w="ttftMs" value="${S.W.ttftMs}" step="100" min="1"></div>
    <div class="sub-hd">Cost</div>
    <div class="f"><label>$ / GPU-hour</label><input type="number" data-w="gpuHr" value="${S.W.gpuHr > 0 ? S.W.gpuHr : ''}" placeholder="not set" step="0.25" min="0"></div>
    <div class="hint">Clear the field to hide all cost figures.</div>
    <div class="sub-hd">Speculative decoding</div>
    <div class="f"><label>Draft tokens per step (γ)</label><input type="number" data-w="specGamma" value="${S.W.specGamma}" step="1" min="0" max="6"></div>
    ${sliderF(S.W, 'specAlpha', 'Acceptance rate (α)', 0.3, 0.99, 0.01, (v) => v.toFixed(2))}
    <div class="hint">γ = 0 turns it off. Uses the model’s MTP layer (one full layer, run γ times) to draft; the target verifies γ+1 tokens per step.</div>`);
  updateDerived(); updateSideSums();
}
function updateDerived() {
  const D = derive(S.model), M = S.model, el = $('#derived'); if (!el) return;
  el.innerHTML = `<span>Total params</span><span>${fmt.n(D.total, D.total >= 1e12 ? 1 : 0)}</span><span>Active / token</span><span>${fmt.n(D.active, 0)}</span>${M.moe ? `<span>Sparsity (k / E)</span><span>${(D.sparsity * 100).toFixed(1)}%</span>` : ''}<span>Weights</span><span>${fmt.bytes((D.total - D.moeLayers * D.routedP) * D.bW + D.moeLayers * D.routedP * D.bE)}</span><span>KV / token${D.hyb ? ' (at 1M ctx)' : ''}</span><span class="hl">${fmt.bytes(D.kvTok)}</span>${D.mlaFactor ? `<span>KV compression vs MHA</span><span class="hl">${D.mlaFactor.toFixed(0)}×</span>` : ''}<span>1M-token request</span><span>${fmt.bytes(D.kv(1e6).store * M.layers)}</span>`;
}

// ------------------------------------------------------------------ parallelism card
const SLIDERS = {
  pp: { label: 'Pipeline', short: 'PP', list: (o) => o.pp, hint: 'layers' },
  replicas: { label: 'Replicas', short: 'DP', list: (o) => o.rep, hint: 'model copies' },
  microbatches: { label: 'Microbatches', short: 'm', list: () => MB_LIST, hint: 'in flight' },
  tpA: { label: 'Tensor', short: 'TP', list: (o) => o.tp, hint: 'attention heads' },
  cp: { label: 'Context', short: 'CP', list: (o) => o.cp, hint: 'sequence' },
  ep: { label: 'Expert', short: 'EP', list: (o) => o.ep, hint: 'experts' },
};
function slHTML(key, P) {
  if (key === 'microbatches' && S.target === 'prefill') return `<div class="crow off" data-tip="Prefill pipelines the chunks of a prompt, so it has no microbatch setting (chunk size is in Workload)."><span class="nm">Microbatches<small>m</small></span><input type="range" disabled><span class="val">–</span></div>`;
  const cfg = SLIDERS[key], list = cfg.list(P.opts), cur = key === 'microbatches' ? P.microbatches : P[key], idx = Math.max(0, list.indexOf(cur));
  const dis = list.length < 2;
  return `<div class="crow" data-tip="${esc(`${cfg.label} (${cfg.hint}). Valid values: ${list.join(', ')}`)}"><span class="nm">${cfg.label}<small>${cfg.short}</small></span><input type="range" data-par="${key}" min="0" max="${Math.max(list.length - 1, 0)}" step="1" value="${idx}" ${dis ? 'disabled' : ''} aria-label="${cfg.label} parallelism"><span class="val">${cur}</span></div>`;
}
function notesHTML(P) {
  const oom = R.ev.dec.oom && S.target === 'decode';   // in split mode, only when the decode pool is being edited
  const items = [...P.notes.filter((n) => !(n.key === 'mb' && S.target === 'prefill')).map((n) => `<span class="pill ${n.lvl === 'warn' ? 'warn' : ''}">${n.lvl === 'warn' ? '⚠ ' : ''}${esc(n.text)}</span>`)];
  if (P.used < P.gpus) items.push(`<span class="pill warn">⚠ ${P.gpus - P.used} of ${P.gpus} GPUs idle</span>`);
  if (oom) items.push('<span class="pill bad">✕ KV cache does not fit at this batch</span>');
  if (!items.length) items.push('<span class="pill good">✓ valid layout</span>');
  return items.join('');
}
const poolText = (p) => `${p === 'prefill' ? 'Prefill' : 'Decode'} · ${gpusOf(p)} GPUs`;
const POOLMODES = [['decode', 'Decode only', 'The whole rack serves decode (token generation). Prefill runs elsewhere.'], ['prefill', 'Prefill only', 'The whole rack serves prefill (prompt processing). Decode runs elsewhere.'], ['split', 'Split rack', 'Prefill and decode share the rack: one part of the GPUs prefills, the rest decodes.']];
// Log-scale slider + number box for a workload value (context length, decode batch). Slider and box edit the same S.W field.
const LG = { isl: { lo: 7, hi: 20, min: 1, max: 1048576 }, batchPerRank: { lo: 0, hi: 12, min: 1, max: 4096 } };
const lgRound = (v) => (v >= 8192 ? Math.round(v / 1024) * 1024 : v >= 1024 ? Math.round(v / 128) * 128 : v >= 128 ? Math.round(v / 16) * 16 : Math.round(v));
const lgRow = (key, label, unit, tip) => { const c = LG[key], v = Math.round(S.W[key]); return `<div class="crow inp" data-tip="${esc(tip)}"><span class="nm">${label}<small>${unit}</small></span><input type="range" data-lg="${key}" data-role="slider" min="${c.lo}" max="${c.hi}" step="0.25" value="${Math.log2(Math.max(c.min, v))}" aria-label="${label}"><input type="number" data-lg="${key}" data-role="text" value="${v}" min="${c.min}" max="${c.max}" step="1" aria-label="${label}"></div>`; };
function renderParShell() {
  const M = S.model, split = S.poolMode === 'split';
  const modeSeg = `<div class="seg2 full" role="tablist">${POOLMODES.map(([v, l, tip]) => `<button data-act="poolmode" data-v="${v}" data-tip="${esc(tip)}" class="${S.poolMode === v ? 'on' : ''}">${l}</button>`).join('')}</div>`;
  const pools = split ? `<div class="seg2 full" style="margin-top:8px"><button id="pool-prefill" data-act="target" data-v="prefill" class="${S.target === 'prefill' ? 'on' : ''}">${poolText('prefill')}</button><button id="pool-decode" data-act="target" data-v="decode" class="${S.target === 'decode' ? 'on' : ''}">${poolText('decode')}</button></div>
    <div class="splitbox"><div class="psl"><div class="row"><span class="nm">GPU split<small>prefill : decode</small></span><span class="val" id="splitval">${S.split} : ${72 - S.split}</span></div>
     <input type="range" data-split="1" min="4" max="68" step="2" value="${S.split}" aria-label="Prefill GPUs"><div class="splitbar"><i id="sb-p" style="width:${(S.split / 72) * 100}%;background:var(--c-kv)"></i><i id="sb-d" style="width:${((72 - S.split) / 72) * 100}%;background:var(--c-attn)"></i></div></div></div>` : '';
  const toggles = `<div class="togs">
        <label class="tog" data-tip="Two microbatches ping-pong so one's all-to-all hides behind the other's compute."><input type="checkbox" data-opt="dbo" ${S.opt.dbo ? 'checked' : ''}><i></i>Dual-batch overlap</label>
        ${M.moe ? `<label class="tog" data-tip="Expert-parallel load balancer with redundant hot experts."><input type="checkbox" data-opt="eplb" ${S.opt.eplb ? 'checked' : ''}><i></i>EPLB</label>` : ''}</div>`;
  $('#parcard').innerHTML = `
    <div class="hd"><h3>Parallelism</h3><span class="muted" style="font-size:12px;margin-left:auto">GPUs / stage / replica <b class="mono" id="gper"></b></span></div>
    <div class="bd cpar">
      ${modeSeg}${pools}
      ${lgRow('isl', 'Context length', 'tokens', 'Prompt length in tokens. Sets prefill work and the KV each request holds while decoding.')}
      ${S.poolMode === 'prefill' ? '' : lgRow('batchPerRank', 'Batches', '/ rank', 'Decode batch per attention rank: how many requests each attention rank steps together. Total in flight = this × attention-DP ranks × microbatches × replicas.')}
      <div class="autorow"><button class="btn sm" id="auto-btn" data-act="auto" data-tip="${esc('Search every valid layout for the highest total throughput that meets your SLOs. If throughput has flattened (the knee), stop at the most interactive point that still keeps ' + Math.round((1 - KNEE_EPS) * 100) + '% of it.')}">Auto-tune</button></div>
      <div class="grp">Stage <em>how the ${split ? 'pool' : 'rack'} is carved up</em></div><div id="sl-pp"></div><div id="sl-replicas"></div><div id="sl-microbatches"></div>
      <div class="grp">Attention <em>TP × DP × CP = GPUs / stage</em></div><div id="sl-tpA"></div><div id="sl-cp"></div>
      <div class="grp">${M.moe ? 'MoE <em>EP × TP<sub>moe</sub> = GPUs / stage</em>' : 'FFN <em>dense: follows attention TP</em>'}</div>${M.moe ? '<div id="sl-ep"></div>' : ''}
      ${toggles}<div class="notes" id="par-notes"></div>
    </div>`;
  updateParPieces();
}
function updateParPieces(active) {
  const P = normTarget(), o = P.opts;
  ['pp', 'replicas', 'microbatches', 'tpA', 'cp', 'ep'].forEach((k) => {
    const el = $(`#sl-${k}`); if (!el) return;
    if (k === active) { const list = SLIDERS[k].list(o), cur = k === 'microbatches' ? P.microbatches : P[k]; el.querySelector('.val').textContent = cur; el.querySelector('.crow').dataset.tip = `${SLIDERS[k].label} (${SLIDERS[k].hint}). Valid values: ${list.join(', ')}`; }
    else el.innerHTML = slHTML(k, P);
  });
  const g = $('#gper'); if (g) g.textContent = `${P.g}`;
  const ab = $('#auto-btn'); if (ab) { ab.disabled = !!S.autoRunning; ab.innerHTML = S.autoRunning ? '<span class="spinner"></span>Auto-tune' : 'Auto-tune'; ab.classList.toggle('miss', !!S.autoMiss); }
  const n = $('#par-notes'); if (n) n.innerHTML = notesHTML(P);
  const sv = $('#splitval'); if (sv) sv.textContent = `${S.split} : ${72 - S.split}`;
  const sp = $('#sb-p'), sd = $('#sb-d'); if (sp) { sp.style.width = (S.split / 72) * 100 + '%'; sd.style.width = ((72 - S.split) / 72) * 100 + '%'; }
  const bp = $('#pool-prefill'), bd = $('#pool-decode'); if (bp) { bp.textContent = poolText('prefill'); bd.textContent = poolText('decode'); }
}
// ------------------------------------------------------------------ KPIs, banner, tabs
const bad = (t) => `<span style="color:var(--bad)">${t}</span>`;
const kpi = (key, label, big, unit, sm, cls = '') => `<button class="card kpi ${cls}" data-act="math" data-k="${key}"><div class="lb">${label}</div><div class="big">${big}<small>${unit}</small></div><div class="sm">${sm}</div><span class="fx">ƒ</span></button>`;
const secOrMs = (s) => (s >= 1 ? [s.toFixed(2), 's'] : [(s * 1e3).toFixed(0), 'ms']);
function renderKpis() {
  const { W, ev } = R, d = ev.dec, pre = ev.pre, pm = S.poolMode;
  const eff = (() => { const Ls = S.model.layers / ev.dP.pp; const fl = d.ops.reduce((s, o) => s + o.flops, 0) * Ls, by = d.ops.reduce((s, o) => s + o.bytes, 0) * Ls; return [fl / d.tStage / R.H.flops[S.model.wDtype], by / d.tStage / R.H.bw]; })();
  const cost = ev.costPerM != null ? kpi('cost', 'Cost', fmt.usd(ev.costPerM), `/ Mtok ${ev.costUnit === 'input' ? 'in' : 'out'}`, `at $${W.gpuHr}/GPU-hr`) : '';
  const inter = kpi('tokUser', 'Interactivity', d.tokUser.toFixed(0), 'tok/s/user', `TPOT ${fmt.ms(d.tpot)}`);
  const conc = kpi('conc', 'Concurrency', fmt.n(d.concurrency, 0), 'requests', `max ${fmt.n(d.mem.bMaxRep * ev.dP.replicas, 0)} · ${d.oom ? bad('over capacity') : fmt.pct(Math.min(1, d.mem.kvUsedFor(d.ba) / Math.max(d.mem.free, 1))) + ' of KV'}`, d.oom ? 'bad' : '');
  const mfu = kpi('mfu', 'Decode utilization', fmt.pct(Math.min(eff[0], 1)), 'MFU', `MBU ${fmt.pct(Math.min(eff[1], 1))}`);
  let html, banner;
  if (pm === 'prefill') {
    const [tv, tu] = secOrMs(pre.ttft);
    html = kpi('ttft', 'Prefill latency', tv, tu, `${pre.nCh} chunk${pre.nCh > 1 ? 's' : ''}`) +
      kpi('inGpu', 'Input throughput', fmt.n(pre.thr, 0), 'tok/s/GPU', `prefill-only · ${fmt.n(pre.thr * 72, 0)} tok/s per rack`) +
      kpi('reqs', 'Prefill rate', fmt.n(pre.reqPerS, 1), 'req/s', `prompts of ${fmt.n(pre.isl, 0)} tokens`) +
      kpi('bubble', 'Pipeline bubble', fmt.pct(pre.bubble), '', ev.pP.pp > 1 ? `PP${ev.pP.pp} · ${pre.nCh} chunks` : 'PP = 1: none') + cost;
    const top = [...pre.segs].filter((x) => !x.hidden).sort((a, b) => b.ms - a.ms)[0], tot = pre.segs.filter((x) => !x.hidden).reduce((a, x) => a + x.ms, 0), pct = Math.round((top.ms / tot) * 100);
    const why = { attnProj: 'Linear projections dominate: prefill is compute-bound, so throughput tracks GEMM efficiency. FP8 attention weights or more replicas raise it.', moe: 'Routed experts dominate: compute-bound expert GEMMs. FP4 experts, a wider EP or less padding raise throughput.', kv: 'Attention over the prompt dominates: quadratic in prompt length. More CP, prefix caching or a sparser attention pattern help.', glue: 'Memory-bound glue kernels (norms, quantization, residual adds, MoE combine) dominate: fusion is the lever.', comm: 'Exposed communication dominates. Try dual-batch overlap or a narrower EP / TP.', pp: 'Pipeline fill and drain dominate: use more chunks per prompt or less PP.' }[top.key];
    banner = { kind: top.key, text: `${top.label} takes ${pct}% of prefill time. ${why}` , ttl: 'Prefill bottleneck' };
  } else {
    const [tv, tu] = ev.ttft != null ? secOrMs(ev.ttft) : [];
    html = inter + kpi('perGpu', 'Throughput', fmt.n(ev.outPerGpu, 0), 'tok/s/GPU', pm === 'split' ? `end-to-end · decode-only ${fmt.n(d.perGpu, 0)}` : 'decode-only') +
      (pm === 'split' ? kpi('ttft', 'Time to first token', tv, tu, `prefill ${fmt.ms(ev.pre.ttft)}`) : '') + cost + conc + mfu;
    banner = { ...d.bottleneck, ttl: 'Decode bottleneck' };
  }
  $('#kpis').innerHTML = html;
  $('#banner').className = 'banner' + (banner.kind === 'capacity' ? ' bad' : '');
  $('#banner').innerHTML = `${banner.kind === 'capacity' ? ICON.warn : ICON.info}<div><b class="ttl">${banner.ttl}</b>${esc(banner.text)}</div>`;
}
const TABS = [['decode', 'Decode'], ['prefill', 'Prefill'], ['pareto', 'Pareto frontier'], ['sweeps', 'Sweeps & regimes'], ['optimizer', 'Optimizer']];
const TABSETS = { decode: ['decode', 'pareto', 'sweeps', 'optimizer'], prefill: ['prefill', 'optimizer'], split: ['decode', 'prefill', 'pareto', 'sweeps', 'optimizer'] };   // Pareto / sweeps / optimizer search the decode pool
const tabsFor = () => TABS.filter(([k]) => TABSETS[S.poolMode].includes(k));
const ensureTab = () => { if (!TABSETS[S.poolMode].includes(S.tab)) S.tab = TABSETS[S.poolMode][0]; };
const SUBS = { decode: [['overview', 'Overview'], ['detail', 'Layer detail']] };
function renderTabs() {
  const sub = SUBS[S.tab] ? `<span class="spacer"></span><div class="seg2 sm">${SUBS[S.tab].map(([v, l]) => `<button data-act="sub" data-v="${v}" class="${S.csub[S.tab] === v ? 'on' : ''}">${l}</button>`).join('')}</div>` : '';
  $('#tabs').innerHTML = tabsFor().map(([k, l]) => `<button data-act="tab" data-v="${k}" class="${S.tab === k ? 'on' : ''}">${l}${k === 'pareto' && S.search.status === 'running' ? '<span class="badge">…</span>' : ''}</button>`).join('') + sub;
}
function renderPane() {
  const fn = { decode: decodeC, prefill: prefillC, pareto: paretoC, sweeps: sweepsC, optimizer: optimizerC }[S.tab];
  const y = window.scrollY, py = $('#pane').scrollTop; $('#pane').innerHTML = fn(S, R); window.scrollTo(0, y); $('#pane').scrollTop = py;
  if (S.tab === 'sweeps' && (!S.regime || S.regime.key !== regimeKey())) startRegime();
}

function renderShell() {
  $('#side').innerHTML = '<div class="brand" id="brand"></div><div id="sidesecs"></div><section class="card par" id="parcard"></section>';
  $('#main').innerHTML = '<div class="kpis" id="kpis"></div><div class="banner" id="banner"></div><div class="tabs" id="tabs"></div><div id="pane"></div>';
  renderBrand(); renderSide();
}

function updateAll(active) {
  compute(); updateSideSums(); updateParPieces(active); renderKpis(); renderTabs(); renderPane(); saveHash();
}

// ------------------------------------------------------------------ background search (Pareto / optimizer)
let searchToken = 0, searchTimer = null;
const QS = new URLSearchParams(location.search), SYNC = QS.has('sync');   // ?sync: run searches synchronously (screenshots/tests)
const tick = () => (SYNC ? Promise.resolve() : new Promise((r) => setTimeout(r, 0)));
function searchKey() { const { batchPerRank, chunk, ...w } = S.W; return JSON.stringify([S.model, S.hwKey, w, S.A, S.poolMode, S.split, S.opt.eplb]); }
function scheduleSearch() { if (S.poolMode === 'prefill') return; clearTimeout(searchTimer);   // prefill-only racks have no decode search
  S.search = { ...S.search, status: 'running', progress: 0 }; renderTabs(); if (S.tab === 'pareto') renderPane(); searchTimer = setTimeout(runSearch, 250); }
async function runSearch() {
  const key = searchKey(), tok = ++searchToken, M = S.model, H = HARDWARE[S.hwKey], gpus = gpusOf('decode');
  S.search = { key, status: 'running', progress: 0, pts: [], front: [] };
  const g = searchDecode(M, H, S.W, S.A, gpus, { eplb: S.opt.eplb, slo: S.W.minTokUser });
  let r;
  for (;;) {
    r = g.next(); if (r.done) break;
    if (tok !== searchToken) return;
    S.search.progress = r.value; if (S.tab === 'pareto') renderPane();
    await tick();
  }
  if (tok !== searchToken) return;
  S.search = { key, status: 'done', progress: 1, ...r.value };
  renderTabs(); if (S.tab === 'pareto' || S.tab === 'optimizer') renderPane();
}
const regimeKey = () => searchKey() + S.par.decode.pp;
async function startRegime() {
  const key = regimeKey(); S.regime = { key, status: 'running' };
  await tick(); await tick();
  if (S.regime?.key !== key) return;
  S.regime = { ...computeRegime(S, R), key }; if (S.tab === 'sweeps') renderPane();
}

// ------------------------------------------------------------------ optimizer
const poolBudget = (dec) => Math.max(0.05, S.W.ttftMs / 1e3 - (dec ? dec.stepMs / 1e3 : 0.03) - 0.010);   // prefill latency allowed by the TTFT SLO
async function runOptimizer() {
  const O = S.optim; if (O.running) return; O.running = true; O.fake = 0; renderPane();
  const { M, H, W, A, ev } = R, slo = W.minTokUser;
  if (S.poolMode === 'split') {
    const g = optimizeSplit(M, H, W, A, { eplb: S.opt.eplb }); let r;
    for (;;) { r = g.next(); if (r.done) break; O.fake = r.value; if (S.tab === 'optimizer') renderPane(); await tick(); }
    const rows = r.value.slice(0, 8), best = rows[0];
    const cost = W.gpuHr > 0 ? (row) => (W.gpuHr * 72 / 3600) / Math.max(row.tokPerGpu * 72, 1e-9) * 1e6 : null;
    const why = best ? `Split <b>${best.gp} : ${best.gd}</b> rate-matches the two pools: prefill sustains ${fmt.n(best.rp, 1)} req/s and decode ${fmt.n(best.rd, 1)} req/s, so the ${best.limiting} pool is the limit at ${fmt.n(best.tokPerGpu, 0)} output tok/s per rack GPU. Decode uses ${family(best.dec)} (PP${best.dec.P.pp} · TP${best.dec.P.tpA} · DP-attn ${best.dec.P.dpA} · EP${best.dec.P.ep}); prefill uses PP${best.pre.P.pp} · TP${best.pre.P.tpA} · DP-attn ${best.pre.P.dpA} · EP${best.pre.P.ep} with ${fmt.n(best.pre.chunk, 0)}-token chunks.` : 'No split satisfies both the interactivity SLO and the TTFT budget. Relax one of them.';
    O.results = { mode: 'split', rows, cost, why }; O.running = false; renderPane(); return;
  }
  if (S.poolMode === 'prefill') {
    await tick();
    const rows = topPrefill(M, H, W, A, 72, poolBudget(null)), best = rows[0];
    const cost = W.gpuHr > 0 ? (row) => (W.gpuHr * 72 / 3600) / Math.max(row.thr * 72, 1e-9) * 1e6 : null;
    const why = best ? `Best prefill layout: PP${best.P.pp} · TP${best.P.tpA} · CP${best.P.cp} · DP-attn ${best.P.dpA} · EP${best.P.ep} × ${best.P.replicas} replica${best.P.replicas > 1 ? 's' : ''}, ${fmt.n(best.chunk, 0)}-token chunks. Prefill is compute-bound, so the winners avoid wide EP and pipeline bubbles: small instances keep GEMMs large and communication local.` : 'No layout meets the TTFT budget. Relax the TTFT target or shorten the prompt.';
    O.results = { mode: 'prefill', rows, cost, why }; O.running = false; renderPane(); return;
  }
  const t0 = performance.now();
  while (!SYNC && (S.search.status !== 'done' || performance.now() - t0 < 900)) { O.fake = Math.min(0.95, (performance.now() - t0) / 1000); if (S.tab === 'optimizer') renderPane(); await new Promise((r) => setTimeout(r, 60)); if (S.search.status === 'idle') break; }
  const pts = S.search.pts, gpus = gpusOf('decode');
  const { top, feasible, score } = pickTop(pts, { objective: O.objective, slo, minGpu: O.minGpu });
  const cost = W.gpuHr > 0 ? (p) => (W.gpuHr * 72 / 3600) / Math.max(p.perGpu * 72, 1e-9) * 1e6 : null;   // only when $/GPU-hr is set
  const best = top[0];
  const robust = robustness(M, H, W, A, top, gpus, { objective: O.objective, slo }, score);
  const close = best ? top.filter((p) => score(p) >= score(best) * 0.95).length - 1 : 0;
  let why = 'No configuration satisfies the constraints. Relax the SLO or the throughput floor.';
  if (best) {
    const P = best.P, cur = ev.dP, bits = [];
    bits.push(`<b>${family(best)}</b>: PP${P.pp} · TP${P.tpA} · CP${P.cp} · DP-attn ${P.dpA} · EP${P.ep}, ${best.ba} requests per attention rank.`);
    if (P.tpA === 1 && M.attn !== 'gqa') bits.push(`No TP: ${M.attn === 'mla' ? 'MLA’s single latent KV head' : 'the single shared KV head'} would be replicated by every TP rank, so attention runs data-parallel and each GPU owns whole requests.`);
    if (P.ep >= P.g / 2 && M.moe) bits.push('Experts are spread as wide as the layout allows: each expert sees tokens from every attention rank, which lifts expert GEMMs toward the compute-bound regime.');
    if (P.pp === 1) bits.push('Pipeline depth 1: PP only adds latency at these interactivity levels.');
    if (P.cp > 1) bits.push(`CP=${P.cp} shards the long-context KV so more requests fit per GPU.`);
    if (cur.tpA !== P.tpA || cur.ep !== P.ep || cur.pp !== P.pp) bits.push(`Compared with your sliders (TP${cur.tpA}, EP${cur.ep}, PP${cur.pp}) it trades ${(ev.dec.tokUser).toFixed(0)} → ${best.tokUser.toFixed(0)} tok/s/user for ${fmt.n(ev.dec.perGpu, 0)} → ${fmt.n(best.perGpu, 0)} tok/s/GPU.`);
    why = bits.join(' ');
  }
  O.results = { mode: 'decode', top, cost, considered: pts.length, feasible, why, robust, close };
  O.running = false; renderPane();
}
function applyPoint(p, keepTab) {
  S.par.decode = pick(p.P); S.opt.dbo = p.cfg.overlap; S.opt.eplb = p.cfg.eplb; S.W.batchPerRank = p.ba; S.target = 'decode';
  renderSide(); renderParShell(); if (!keepTab) S.tab = 'decode'; updateAll();
}
function applySplitRow(row, keepTab) {
  S.split = row.gp; S.par.prefill = pick(row.pre.P); S.par.decode = pick(row.dec.P); S.W.chunk = row.pre.chunk; S.W.batchPerRank = row.dec.ba;
  S.opt.dbo = row.dec.cfg.overlap; S.opt.eplb = row.dec.cfg.eplb; S.target = 'decode';
  renderSide(); renderParShell(); scheduleSearch(); if (!keepTab) S.tab = 'decode'; updateAll();
}
function applyPrefillRow(row, keepTab) {
  S.par.prefill = pick(row.P); S.W.chunk = row.chunk; S.opt.dbo = row.cfg.overlap; S.target = 'prefill';
  renderSide(); renderParShell(); if (!keepTab) S.tab = 'prefill'; updateAll();
}
// ------------------------------------------------------------------ auto-tune button
async function autoTune() {
  if (S.autoRunning) return;
  S.autoRunning = true; S.autoMiss = false; updateParPieces(); await tick();
  const M = S.model, H = HARDWARE[S.hwKey], W = S.W, A = S.A, pm = S.poolMode;
  try {
    let res;
    if (pm === 'decode') {
      const g = autoTuneDecode(M, H, W, A, 72, { slo: W.minTokUser, eplb: S.opt.eplb }); let r;
      for (;;) { r = g.next(); if (r.done) break; await tick(); }
      res = r.value; if (res) applyPoint(res.point, true);
    } else if (pm === 'prefill') {
      await tick(); res = autoTunePrefill(M, H, W, A, 72, poolBudget(null)); if (res) applyPrefillRow(res.row, true);
    } else {
      const g = autoTuneSplit(M, H, W, A, { eplb: S.opt.eplb }); let r;
      for (;;) { r = g.next(); if (r.done) break; await tick(); }
      res = r.value; if (res) applySplitRow(res.row, true);
    }
    S.autoMiss = !res;   // nothing meets the targets: the layout is left as it was and the button briefly flashes
    if (!res) setTimeout(() => { S.autoMiss = false; updateParPieces(); }, 1800);
  } catch (e) { S.autoMiss = true; console.error('Auto-tune failed', e); }
  S.autoRunning = false; updateParPieces();
}

function setModel(key) {
  S.modelKey = key; S.model = fullModel(key);
  const m = S.model;
  if (!m.moe) S.par.decode = { pp: 1, replicas: 9, tpA: 8, cp: 1, ep: 1, microbatches: 1 };
  else S.par.decode = { pp: 1, replicas: 1, tpA: 1, cp: 1, ep: 72, microbatches: 1 };
  S.par.prefill = { pp: 1, replicas: 3, tpA: 1, cp: 1, ep: 24, microbatches: 1 };
  if (key === 'hypo_10t') { S.W.isl = 1000000; S.W.batchPerRank = 1; S.par.decode = { pp: 1, replicas: 1, tpA: 2, cp: 4, ep: 72, microbatches: 1 }; } else if (S.W.isl >= 1000000) { S.W.isl = 2048; S.W.batchPerRank = 96; }
  renderParShell(); renderSide(); scheduleSearch(); updateAll(); if (S.optim) S.optim.results = null;
}

// switching attention kind: fill the fields the new kind needs
function ensureAttn(M) {
  if (M.attn === 'hybrid') Object.assign(M, HYBRID_DEFAULT);
  else if (M.attn === 'mla') { for (const [k, v] of Object.entries({ qLora: 1536, kvLora: 512, rope: 64, nope: 128, vDim: 128 })) if (!(M[k] > 0) || k === 'qLora') M[k] = v; }
  else { if (!(M.kvHeads > 1)) M.kvHeads = Math.min(8, M.heads); if (!(M.headDim > 0) || M.headDim > 256) M.headDim = 128; }
}

// ------------------------------------------------------------------ math drawer
function openDrawer(key) {
  const m = mathFor(key, S, R);
  $('#drawer').innerHTML = `<div class="dh"><h3>${m.title}</h3><span class="pill">show the math</span><span class="spacer"></span><button class="btn iconbtn" data-act="close" aria-label="Close">✕</button></div><div class="db">${m.html}<p class="hint">Constants (efficiencies, latencies, overheads) come from the Hardware → Assumptions panel. They are all <em>assumed</em> defaults, fitted only to two published GB200 anchors; see docs/validation.md in the repository.</p></div>`;
  $('#drawer').classList.add('on'); $('#scrim').classList.add('on'); $('#drawer').setAttribute('aria-hidden', 'false');
}
function closeDrawer() { $('#drawer').classList.remove('on'); $('#scrim').classList.remove('on'); $('#drawer').setAttribute('aria-hidden', 'true'); }

// ------------------------------------------------------------------ events
const num = (el) => (el.value === '' ? 0 : +el.value);
document.addEventListener('input', (e) => {
  const t = e.target;
  if (t.dataset.par) {
    const key = t.dataset.par, P = normTarget(), list = SLIDERS[key].list(P.opts);
    S.autoMiss = false;
    S.par[S.target] = { ...S.par[S.target], [key]: list[+t.value] };   // PP and microbatches are independent: raising PP no longer touches your microbatches setting
    S.par[S.target] = pick(normTarget()); updateAll(key); return;
  }
  if (t.dataset.lg) {
    const k = t.dataset.lg, c = LG[k], slider = t.dataset.role === 'slider';
    if (!slider && t.value === '') return;
    const v = slider ? Math.max(c.min, Math.min(c.max, lgRound(2 ** +t.value))) : Math.max(c.min, Math.min(c.max, Math.round(num(t))));
    S.W[k] = v; S.autoMiss = false;
    const o = t.closest('.crow').querySelector(`[data-role=${slider ? 'text' : 'slider'}]`);
    if (slider) o.value = v; else o.value = Math.log2(v);
    updateAll(); if (k === 'isl') scheduleSearch(); return;
  }
  if (t.dataset.split) { S.split = +t.value; renormPars(); updateAll(); scheduleSearch(); return; }
  if (t.dataset.m) { S.model[t.dataset.m] = num(t); S.modelKey = S.modelKey; renormPars(); updateDerived(); updateAll(); scheduleSearch(); return; }
  if (t.dataset.w) { if (t.closest('#pane')) return; S.W[t.dataset.w] = num(t); updateAll(); scheduleSearch(); return; }
  if (t.dataset.ws) { const k = t.dataset.ws, v = t.dataset.log === '1' ? Math.pow(2, +t.value) : +t.value; S.W[k] = v; const lab = $('#v-w-' + k); if (lab) lab.textContent = k === 'specAlpha' ? v.toFixed(2) : Math.round(v); updateAll(); scheduleSearch(); return; }
  if (t.dataset.as) { const k = t.dataset.as; S.A[k] = +t.value; const lab = $('#v-a-' + k); if (lab) lab.textContent = k === 'alphaUs' ? t.value + ' µs' : k === 'overheadMs' ? (+t.value).toFixed(2) + ' ms' : (+t.value).toFixed(2); updateAll(); scheduleSearch(); return; }
  if (t.dataset.o) { if (t.type === 'number') return; const k = t.dataset.o; S.optim[k] = t.type === 'checkbox' ? t.checked : t.tagName === 'SELECT' ? t.value : num(t); if (t.tagName === 'SELECT') renderPane(); return; }
});
document.addEventListener('change', (e) => {
  const t = e.target;
  if (t.dataset.preset) return setModel(t.value);
  if (t.dataset.ms) { const k = t.dataset.ms; S.model[k] = t.dataset.bool ? t.value === 'true' : t.value;
    if (k === 'attn') ensureAttn(S.model);
    if (k === 'moe' && S.model.moe && !(S.model.experts > 0)) Object.assign(S.model, { experts: 64, topK: 4, expertInter: 2048, shared: 0, denseLayers: Math.min(S.model.denseLayers || 1, S.model.layers - 1) }); renormPars(); renderSide(); renderParShell(); scheduleSearch(); updateAll(); return; }
  if (t.dataset.opt) { S.opt[t.dataset.opt] = t.checked; scheduleSearch(); updateAll(); return; }
  if (t.dataset.w && t.closest('#pane')) { S.W[t.dataset.w] = num(t); updateAll(); scheduleSearch(); return; }
  if (t.dataset.o && t.type === 'number') { S.optim[t.dataset.o] = num(t); return; }
  if (t.dataset.m) { renderSide(); }
  if (t.dataset.lg && t.dataset.role === 'text') t.value = Math.round(S.W[t.dataset.lg]);   // snap the typed value into range once editing ends
});
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]'); if (!b) return;
  const a = b.dataset.act, v = b.dataset.v;
  if (a === 'tab') { S.tab = v; renderTabs(); renderPane(); saveHash(); }
  else if (a === 'poolmode') { S.poolMode = v; S.target = v === 'split' ? 'decode' : v; ensureTab(); S.optim.results = null; renormPars(); renderParShell(); scheduleSearch(); updateAll(); }
  else if (a === 'sub') { S.csub[S.tab] = v; renderTabs(); renderPane(); saveHash(); }
  else if (a === 'apply-split-row') applySplitRow(S.optim.results.rows[+b.dataset.i]);
  else if (a === 'apply-prefill-row') applyPrefillRow(S.optim.results.rows[+b.dataset.i]);
  else if (a === 'target') { S.target = v; renderParShell(); }
  else if (a === 'theme') setTheme({ auto: 'light', light: 'dark', dark: 'auto' }[S.theme]);
  else if (a === 'share') { try { navigator.clipboard.writeText(location.href); b.innerHTML = ICON.check; b.dataset.tip = 'Link copied'; setTimeout(() => { b.innerHTML = ICON.link; b.dataset.tip = 'Copy a link to this exact setup'; }, 1500); } catch (err) { b.dataset.tip = 'Copy the link from the address bar'; } }
  else if (a === 'math') openDrawer(b.dataset.k);
  else if (a === 'close') closeDrawer();
  else if (a === 'run-opt') runOptimizer();
  else if (a === 'auto') autoTune();
  else if (a === 'apply-opt') applyPoint(S.optim.results.top[+b.dataset.i]);
  else if (a === 'apply-frontier') applyPoint(S.search.front[+b.dataset.i]);
});
$('#scrim').addEventListener('click', closeDrawer);
document.addEventListener('toggle', (e) => { const d = e.target; if (d.dataset && d.dataset.sec) S.open[d.dataset.sec] = d.open; }, true);   // remember which sidebar sections are open across re-renders
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });

// tooltip
const tip = $('#tip');
document.addEventListener('mouseover', (e) => { const t = e.target.closest('[data-tip]'); if (t) { tip.innerHTML = t.getAttribute('data-tip'); tip.classList.add('on'); } else tip.classList.remove('on'); });
document.addEventListener('mousemove', (e) => { if (!tip.classList.contains('on')) return; const w = tip.offsetWidth, h = tip.offsetHeight; let x = e.clientX + 14, y = e.clientY + 14; if (x + w > innerWidth - 8) x = e.clientX - w - 14; if (y + h > innerHeight - 8) y = e.clientY - h - 14; tip.style.left = x + 'px'; tip.style.top = y + 'px'; });

// ------------------------------------------------------------------ boot
try { const t = QS.get('theme') || localStorage.getItem('pe-theme'); if (t) setThemeInit(t); } catch (e) { /* ignore */ }
function setThemeInit(t) { S.theme = t; if (t !== 'auto') document.documentElement.setAttribute('data-theme', t); }
loadHash();
if (QS.get('pool')) S.poolMode = QS.get('pool');   // ?pool=decode|prefill|split
S.hwKey = 'gb200';
if (!TABSETS[S.poolMode]) S.poolMode = 'decode';
if (QS.get('tab')) S.tab = QS.get('tab');   // ?tab=pareto and ?sub=detail: test hooks
if (QS.get('sub')) S.csub.decode = QS.get('sub');
if (QS.get('open')) QS.get('open').split(',').forEach((k) => (S.open[k] = true));   // ?open=model,hw,wl expands sidebar sections (screenshots)
ensureTab();
compute(); renormPars();
renderShell(); renderParShell(); updateAll();
(S.poolMode === 'prefill' ? Promise.resolve() : runSearch()).then(() => { if (QS.get('run') === 'opt') runOptimizer(); if (QS.get('auto')) autoTune(); });
