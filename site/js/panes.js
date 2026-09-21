// Result panes (one screen, no scrolling at ~1440×900) + the "show the math" drawer content.
// Pure functions: (S = UI state, R = computed results) -> HTML string. Each pane root is a `.cpane` grid.
import { xyChart, roofline, stackBar, legendRows, stackedColumns, heatmap, gantt, laneTimeline, fmt, esc } from './charts.js';
import { evalDecode, evalPrefill, memoryPerGpu, sweepConfig, cfgLabel, family, poolGpus, enumerateConfigs } from './core/index.js';

export const COLORS = { spec: '#2e9fd6', attnProj: 'var(--c-attn)', kv: 'var(--c-kv)', dense: 'var(--c-dense)', moe: 'var(--c-moe)', comm: 'var(--c-comm)', hidden: 'var(--c-comm)', pp: 'var(--c-pp)', over: 'var(--c-over)' };
export const FAM_COLORS = { 'DP-attn + wide EP': '#4f7cf0', 'Small EP groups × replicas': '#17a89a', 'TP + EP': '#f08c2b', 'TP + CP': '#d0629f', 'Deep PP': '#c9a21a' };
export const FAM_SHORT = { 'DP-attn + wide EP': 'DP·wide-EP', 'Small EP groups × replicas': 'EP×repl', 'TP + EP': 'TP·EP', 'TP + CP': 'TP·CP', 'Deep PP': 'PP' };

export const tagFor = (b) => ({ memory: '<span class="tag bw">HBM-bound</span>', compute: '<span class="tag cmp">compute-bound</span>', latency: '<span class="tag lat">latency-bound</span>', bandwidth: '<span class="tag bw">link-BW-bound</span>' }[b] || '');

export const CTX = [1024, 4096, 16384, 65536, 262144, 1048576];
export const ctxLabel = (v) => (v >= 1048576 ? '1M' : v >= 1024 ? v / 1024 + 'K' : v);
export function computeRegime(S, R) {
  const { M, H, W, A } = R, gpus = poolGpus(S, 'decode');
  let cfgs = enumerateConfigs(M, gpus, [true], [S.opt.eplb]);
  const step = Math.max(1, Math.floor(cfgs.length / 140)); cfgs = cfgs.filter((_, i) => i % step === 0);
  const ys = [10, 20, 40, 80, 120, 200], xs = CTX;
  const cells = []; let min = Infinity, max = 0;
  xs.forEach((ctx, i) => {
    const pts = []; const W2 = { ...W, isl: ctx };
    cfgs.forEach((c) => sweepConfig(M, H, W2, A, c, gpus, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]).forEach((p) => pts.push(p)));
    ys.forEach((y, j) => {
      const ok = pts.filter((p) => p.tokUser >= y).sort((a, b) => b.perGpu - a.perGpu)[0];
      if (ok) { min = Math.min(min, ok.perGpu); max = Math.max(max, ok.perGpu); }
      cells.push({ i, j, best: ok || null, fam: ok ? family(ok) : null });
    });
  });
  return { status: 'done', xs, ys, cells, min: min === Infinity ? 1 : min, max: Math.max(max, min * 1.01) };
}

// ================================================================= MATH DRAWER
const mb = (title, formula, rows, note) => `<div class="math"><h5>${title}</h5><div class="fm">${esc(formula)}</div>${rows ? `<div class="kv" style="margin-top:8px">${rows.map(([a, b]) => `<span>${a}</span><span>${b}</span>`).join('')}</div>` : ''}${note ? `<p class="sub">${note}</p>` : ''}</div>`;
export function mathFor(key, S, R) {
  const { M, H, W, A, ev } = R, dec = ev.dec, P = ev.dP;
  const Ls = M.layers / P.pp, hop = P.pp > 1 ? (dec.stepMs / 1e3 - A.overheadMs * 1e-3 - Math.max(P.microbatches, P.pp) * dec.tStage) / P.pp : 0;
  if (key === 'tokUser' || key === 'tpot') return { title: 'Interactivity & TPOT', html:
    mb('Per-user token rate', 'tokens/s/user = 1 / TPOT', [['TPOT', fmt.ms(dec.tpot)], ['tokens/s/user', dec.tokUser.toFixed(1)]]) +
    mb('Pipeline-aware step time', 'TPOT = max(m, p) · t_stage + p · t_hop + t_overhead', [['m (microbatches)', P.microbatches], ['p (PP stages)', P.pp], ['t_stage', fmt.ms(dec.tStage)], ['t_hop', fmt.ms(Math.max(hop, 0))], ['t_overhead', A.overheadMs + ' ms']], 'PP buys capacity, not latency: with m < p the pipeline has bubbles; with m ≥ p each user waits m stage-times.') +
    mb('Stage time', 't_stage = layers_per_stage · t_layer + t_lm_head / p', [['layers / stage', Ls.toFixed(1)], ['t_layer', fmt.ms(dec.tLayer)]]) +
    mb('One layer', 't_layer = Σ compute + exposed comm + kernel floors', [...dec.ops.map((o) => [o.name, fmt.ms(o.t)]), ['comm total', fmt.ms(dec.commT)], ['comm hidden by overlap', fmt.ms(dec.hidden)], ['comm exposed', fmt.ms(dec.exposed)]], 'Each op time = max(FLOPs / (peak · η_c(M)), bytes / (HBM BW · η_m)).') };
  if (key === 'perGpu') return { title: 'Tokens / s / GPU', html:
    mb('Decode pool', 'decode tok/s = m · b / TPOT × replicas', [['b (tokens / microbatch)', dec.b], ['m', P.microbatches], ['replicas', P.replicas], ['decode tok/s (pool)', fmt.n(dec.total, 0)]]) +
    (S.poolMode === 'split' ? mb('End to end (per GPU on the rack)', 'out tok/s/GPU = min(prefill req/s, decode req/s) · OSL / 72', [['prefill pool req/s', fmt.n(ev.rp, 2)], ['decode pool req/s', fmt.n(ev.rd, 2)], ['result', fmt.n(ev.outPerGpu, 0) + ' tok/s/GPU']], 'The slower pool sets the rate; the optimizer balances the split.') : mb('Per GPU', 'tok/s/GPU = decode tok/s (pool) / GPUs in the pool', [['GPUs dedicated to decode', P.gpus], ['result', fmt.n(dec.perGpu, 0) + ' tok/s/GPU']], 'The whole rack decodes, so this is decode-only throughput; prefill happens elsewhere.')) };
  if (key === 'inGpu') return { title: 'Input tokens / s / GPU', html: mb('Prefill throughput', 'input tok/s/GPU = (ISL / t_prefill) · replicas / GPUs', [['tokens to prefill', fmt.n(ev.pre.isl, 0)], ['t_prefill', fmt.ms(ev.pre.tPrefill)], ['replicas', ev.pP.replicas], ['GPUs', ev.pP.gpus], ['result', fmt.n(ev.pre.thr, 0) + ' tok/s/GPU']]) };
  if (key === 'reqs') return { title: 'Prefill requests / s', html: mb('Rack request rate', 'req/s = replicas / t_prefill', [['replicas', ev.pP.replicas], ['t_prefill', fmt.ms(ev.pre.tPrefill)], ['result', fmt.n(ev.pre.reqPerS, 2) + ' req/s']], 'This is how many prompts of the configured length the rack can absorb per second.') };
  if (key === 'bubble') return { title: 'Pipeline bubble', html: mb('PP bubble', 'bubble = (p − 1) / (chunks + p − 1)', [['p (PP stages)', ev.pP.pp], ['chunks', ev.pre.nCh], ['result', fmt.pct(ev.pre.bubble)]], 'Chunked prefill keeps the pipeline full: more chunks per prompt amortize fill and drain.') };
  if (key === 'ttft' && S.poolMode === 'prefill') return { title: 'Prefill latency', html: mb('Latency to first token, prefill side', 'latency = t_prefill + KV hand-off', [['prefill', fmt.ms(ev.pre.tPrefill)], ['chunks', ev.pre.nCh + ' × ' + fmt.n(ev.pre.C, 0)], ['KV transfer', fmt.ms(ev.pre.kvXfer)]], 'Excludes queueing and the first decode step, which happen on the decode rack.') };
  if (key === 'ttft') return { title: 'Time to first token', html: mb('TTFT', 'TTFT = queueing + prefill + KV hand-off + first decode step', [['prefill', fmt.ms(ev.pre.tPrefill)], ['chunks', ev.pre.nCh + ' × ' + fmt.n(ev.pre.C, 0)], ['KV transfer', fmt.ms(ev.pre.kvXfer)], ['first decode step', fmt.ms(dec.tpot)], ['queueing (assumed)', '10 ms']], 'Mean-only in v1; a discrete-event simulator would add queueing and tails.') };
  if (key === 'cost') return { title: 'Cost per million tokens', html: mb(`$ / Mtok ${ev.costUnit}`, `$/Mtok = ($/GPU-hr · 72 / 3600) / (rack ${ev.costUnit} tok/s) · 10⁶`, [['$/GPU-hr', fmt.usd(W.gpuHr)], [`rack ${ev.costUnit} tok/s`, fmt.n((ev.costUnit === 'input' ? ev.pre.thr : ev.outPerGpu) * 72, 0)], ['result', fmt.usd(ev.costPerM)]]) };
  if (key === 'conc') return { title: 'Concurrency & capacity', html: mb('Max requests per attention rank', 'floor( free HBM / (KV per request per GPU · microbatches) )', [['free HBM', fmt.bytes(dec.mem.free)], ['KV / request / GPU', fmt.bytes(dec.mem.kvReq)], ['microbatches', P.microbatches], ['max / rank', fmt.n(dec.mem.perRankMax, 0)], ['× DP-attn ranks', P.dpA], ['max concurrent (replica)', fmt.n(dec.mem.bMaxRep, 0)]], 'KV per request per GPU = KV bytes at this context · layers per stage, divided by CP (and by TP for GQA). Hybrid CSA/HCA models store one compressed entry per 4 or 128 tokens per layer, plus a 128-token window.') };
  if (key === 'mfu') { const fl = dec.ops.reduce((s, o) => s + o.flops, 0) * Ls, by = dec.ops.reduce((s, o) => s + o.bytes, 0) * Ls; return { title: 'Utilization', html: mb('MFU / MBU', 'MFU = FLOPs per stage / (t_stage · peak);  MBU = bytes / (t_stage · HBM BW)', [['FLOPs / stage / GPU', fmt.n(fl)], ['HBM bytes / stage / GPU', fmt.bytes(by)], ['t_stage', fmt.ms(dec.tStage)], ['MFU', fmt.pct(fl / dec.tStage / H.flops[M.wDtype])], ['MBU', fmt.pct(by / dec.tStage / H.bw)]]) }; }
  return { title: 'Details', html: '<p class="muted">Coming soon.</p>' };
}

const CW = 500, CH = 270;   // default chart viewBox: roughly one half-width card at 1440×900
const cc = (title, body, { info, right, cls = '' } = {}) => `<section class="card cc ${cls}"><div class="hd"><h3>${title}</h3>${info ? `<span class="i" data-tip="${esc(info)}">i</span>` : ''}${right ? `<span class="hr">${right}</span>` : ''}</div><div class="bd">${body}</div></section>`;
// compact layout label: omit factors equal to 1 (they are implied); DP-attn and EP always shown
const lay = (P, fam) => `<div class="cfgchips">${fam ? `<span class="fam" style="--c:${FAM_COLORS[fam]}">${FAM_SHORT[fam]}</span>` : ''}${[P.pp > 1 && `PP${P.pp}`, P.tpA > 1 && `TP${P.tpA}`, P.cp > 1 && `CP${P.cp}`].filter(Boolean).map((t) => `<span>${t}</span>`).join('')}<span class="hi">DP-attn ${P.dpA}</span><span class="hi">EP${P.ep}</span>${P.replicas > 1 ? `<span>×${P.replicas} repl.</span>` : ''}</div>`;
const bigv = (v, unit = '') => `<span class="hv">${v}${unit ? `<small>${unit}</small>` : ''}</span>`;

// ================================================================= DECODE
export function decodeC(S, R) {
  return S.csub.decode === 'detail' ? decodeDetail(S, R) : decodeOverview(S, R);
}
function decodeOverview(S, R) {
  const { M, H, W, A, ev } = R, dec = ev.dec, P = ev.dP, mem = dec.mem;
  const f = (v) => fmt.ms(v / 1e3);
  const vis = dec.segs.filter((s) => !s.hidden), hid = dec.segs.find((s) => s.hidden);
  const segs = vis.map((s) => ({ v: s.ms, label: s.label, color: COLORS[s.key] })), tot = segs.reduce((a, s) => a + s.v, 0);
  const breakdown = cc('Step time', `${stackBar(segs, { total: tot, fmtv: f, height: 26 })}${legendRows(segs, tot, f)}
    ${hid ? `<div class="lrow" style="margin-top:5px"><i class="sw hatch" style="--c:var(--c-comm)"></i><span class="ln muted">Comm hidden by overlap</span><span class="lv">${f(hid.ms)}</span><span class="lp faint">off path</span></div>` : ''}`,
    { info: 'Critical path of one decode step (TPOT) for one user. Hover a segment for details.', right: `${bigv(fmt.ms(dec.tpot))}<span class="muted">TPOT · batch ${dec.ba}/rank${dec.E > 1 ? ` · ${dec.E.toFixed(2)} tok/step` : ''}</span>` });

  const peak = H.flops[M.wDtype];
  const pts = dec.ops.map((o) => ({ name: o.name, ai: o.ai, perf: o.flops / o.t, color: COLORS[o.cat], tip: `<b>${o.name}</b><br>AI ${o.ai.toFixed(1)} FLOP/B<br>${fmt.flops(o.flops / o.t)} achieved<br>${o.bound}-bound` }));
  const roof = cc('Roofline', roofline({ peak, bw: H.bw * A.hbmEff, peakLabel: `${M.wDtype.toUpperCase()} ${fmt.flops(peak)}`, points: pts, w: CW, h: CH }), { info: 'Per GPU, one decode layer. Each dot is an op. Left of the ridge = memory-bound; right = compute-bound.' });

  const grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
  const sw = grid.map((ba) => ({ ba, r: evalDecode(M, H, W, A, P, ba) }));
  const flip = sw.find((s) => s.r.ops.find((o) => o.cat === 'moe')?.bound === 'compute');
  const cap = Math.max(mem.perRankMax, 1);
  const sweep = cc('Batch sweep', xyChart({
    w: CW, h: CH, rightAxis: true, xLog: true, xMin: 1, xMax: 4096, xLabel: 'decode batch per attention rank', yLabel: 'TPOT (ms)', yLabelR: 'tok/s/GPU', yColor: 'var(--c-comm)', yColorR: 'var(--accent)', xFmt: (v) => fmt.n(v, 0), yFmtR: (v) => fmt.n(v, 0),
    series: [
      { name: 'TPOT (ms)', color: 'var(--c-comm)', pts: sw.map((s) => [s.ba, s.r.tpot * 1e3]), dots: true, r: 2.5, tips: sw.map((s) => `batch ${s.ba}/rank<br>TPOT ${fmt.ms(s.r.tpot)}<br>${s.r.tokUser.toFixed(0)} tok/s/user`) },
      { name: 'tokens/s/GPU', color: 'var(--accent)', axis: 'r', pts: sw.map((s) => [s.ba, s.r.perGpu]), dots: true, r: 2.5, tips: sw.map((s) => `batch ${s.ba}/rank<br>${fmt.n(s.r.perGpu, 0)} tok/s/GPU`) },
    ],
    regions: cap < 4096 ? [{ x0: cap, x1: 4096, color: 'var(--bad)', label: 'KV does not fit', op: 0.09 }] : [],
    vlines: flip ? [{ x: flip.ba, color: 'var(--c-moe)', label: 'experts compute-bound', dy: 15 }] : [],
    markers: [{ x: dec.ba, y: dec.tpot * 1e3, color: 'var(--c-comm)', r: 6, label: 'you' }, { x: dec.ba, y: dec.perGpu, axis: 'r', color: 'var(--accent)', r: 6 }],
  }), { info: 'Bigger batches raise throughput until something saturates: latency grows, and eventually the KV cache no longer fits.' });

  const kvUsed = Math.min(mem.kvUsedFor(dec.ba), Math.max(0, mem.hbm - mem.wAttn - mem.wExp - mem.wEmb - mem.reserved - mem.act));
  const free = Math.max(0, mem.hbm - mem.wAttn - mem.wExp - mem.wEmb - kvUsed - mem.act - mem.reserved);
  const ms = [
    { v: mem.wAttn, label: 'Attention / dense weights', color: 'var(--c-attn)' },
    { v: mem.wExp, label: 'Expert weights (+ redundant)', color: 'var(--c-moe)' },
    { v: mem.wEmb, label: 'Embedding / LM head', color: 'var(--c-dense)' },
    { v: kvUsed, label: 'KV cache in use', color: 'var(--c-kv)' },
    { v: mem.act, label: 'Activations & workspace', color: 'var(--c-pp)' },
    { v: free, label: 'Free', color: 'var(--c-free)' },
  ];
  const usable = mem.hbm - mem.reserved;   // HBM the model may use: the fixed reserve (utilization cap, CUDA graphs, comm buffers) is left out of the bar
  const memory = cc('Memory per GPU', `${stackBar(ms, { total: usable, fmtv: fmt.bytes, height: 22, showLabels: false })}${legendRows(ms, usable, fmt.bytes)}
    <div class="kv tight"><span>Max requests / attention rank</span><span>${fmt.n(mem.perRankMax, 0)}</span><span>Max concurrent (replica)</span><span>${fmt.n(mem.bMaxRep, 0)}</span></div>`,
    { info: 'Weights left after sharding: attention replicates across DP-attn ranks; experts shard across EP. Bar shows usable HBM: 10% headroom plus ~3 GB of CUDA-graph and communication buffers are already set aside.', right: `${bigv(fmt.bytes(usable - free))}<span class="muted">of ${fmt.bytes(usable)} usable</span>${dec.oom ? '<span class="pill bad">OOM</span>' : ''}` });
  return `<div class="cpane g22">${breakdown}${roof}${sweep}${memory}</div>`;
}
function decodeDetail(S, R) {
  const { M, H, A, ev } = R, dec = ev.dec, P = ev.dP;
  const dot = (c) => `<i style="display:inline-block;width:9px;height:9px;border-radius:3px;background:${c};margin-right:7px"></i>`;
  const ops = dec.ops.map((o) => `<tr><td>${dot(COLORS[o.cat])}${o.name}</td><td class="n">${fmt.n(o.flops)}</td><td class="n">${fmt.bytes(o.bytes)}</td><td class="n">${fmt.ms(o.t)}</td><td>${tagFor(o.bound)}</td></tr>`).join('');
  const comm = dec.comm.map((c) => { const reg = c.bytes / (H.link * 0.7) < A.alphaUs * 1e-6 * 1.5 ? 'latency' : 'bandwidth'; return `<tr><td>${dot('var(--c-comm)')}${c.name} <span class="faint">(n=${c.n})</span></td><td class="n">–</td><td class="n">${fmt.bytes(c.bytes)}</td><td class="n">${fmt.ms(c.t)}</td><td>${tagFor(reg)}</td></tr>`; }).join('');
  const table = cc('Per-layer ops & collectives', `<div class="tblwrap"><table class="t dense"><thead><tr><th>Component</th><th class="n">FLOPs</th><th class="n">Bytes</th><th class="n">Time</th><th>Regime</th></tr></thead><tbody>${ops}${comm || '<tr><td colspan="5" class="muted">No collectives with this layout.</td></tr>'}</tbody></table></div>
    <div class="kv tight" style="margin-top:8px"><span>Experts read / GPU / step</span><span>${M.moe ? dec.touched.toFixed(1) + ' of ' + (P.slots / P.ep).toFixed(1) : '–'}</span><span>Expert load imbalance (max / mean)</span><span>${M.moe ? '×' + dec.imb.toFixed(2) : '–'}</span></div>`,
    { info: 'Per layer, per GPU, at the current batch: the "show the math" view in table form.' });
  const compute = dec.ops.map((o) => ({ name: o.name, short: o.name.split(' ')[0], t: o.t, color: COLORS[o.cat] }));
  const commLane = dec.comm.map((c) => ({ name: c.name, short: c.name.split(' ')[0] + ' ' + (c.name.split(' ')[1] || ''), t: c.t }));
  const tl = laneTimeline({ compute, comm: commLane, overlap: P.overlap, w: CW }) + (P.pp > 1 ? `<div class="sub-hd" style="margin-top:8px">Pipeline schedule · ${P.microbatches} microbatches</div>${gantt({ stages: P.pp, micro: Math.min(P.microbatches, 12), kind: 'decode', w: CW })}` : '');
  const anat = cc('Anatomy of one layer', tl, { info: 'With dual-batch overlap on, one microbatch’s all-to-all runs while the other computes.' });
  return `<div class="cpane g12 top">${table}${anat}</div>`;
}

// ================================================================= PREFILL
export function prefillC(S, R) {
  const { M, H, W, A, ev } = R, pre = ev.pre, P = ev.pP;
  const f = (v) => fmt.ms(v / 1e3);
  const segs = pre.segs.filter((s) => !s.hidden).map((s) => ({ v: s.ms, label: s.label, color: COLORS[s.key] })), hid = pre.segs.find((s) => s.hidden);
  const tot = segs.reduce((a, s) => a + s.v, 0);
  const stat = (lb, v) => `<div class="stat"><div class="lb">${lb}</div><div class="v">${v}</div></div>`;
  const stats = `<div class="stats s4">${stat(`Prefill time (ISL ${fmt.n(pre.isl, 0)})`, fmt.ms(pre.tPrefill))}${stat('Input tok/s/GPU', fmt.n(pre.thr, 0))}${stat('Chunks × size', `${pre.nCh} × ${fmt.n(pre.C, 0)}`)}${stat('PP bubble', fmt.pct(pre.bubble))}</div>`;
  const bd = cc('Prefill time breakdown', `${stackBar(segs, { total: tot, fmtv: f, height: 26 })}${legendRows(segs, tot, f)}${hid ? `<div class="lrow" style="margin-top:5px"><i class="sw hatch" style="--c:var(--c-comm)"></i><span class="ln muted">Comm hidden behind compute</span><span class="lv">${f(hid.ms)}</span><span class="lp faint">off path</span></div>` : ''}
    <div class="kv tight"><span>KV per request</span><span>${fmt.bytes(pre.kvBytes)}</span><span>Hand-off over NVLink</span><span>${fmt.ms(pre.kvXfer)}</span><span>Same over a 400 Gb/s NIC</span><span>${fmt.ms(pre.kvBytes / 50e9)}</span></div>`,
    { info: 'Prefill is compute-bound: watch the attention share grow with prompt length.' });

  const lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
  const rows = lens.map((n) => ({ n, r: evalPrefill(M, H, { ...W, isl: n }, A, P, P.gpus) })), base = rows[0].r.tPrefill / 512;
  const ttft = cc('Prefill time vs prompt length', xyChart({
    w: CW, h: CH, xLog: true, yLog: true, xMin: 512, xMax: 131072, xLabel: 'prompt length (tokens)', yLabel: 'prefill time', margin: { l: 66 }, xFmt: (v) => fmt.n(v, 0), yFmt: (v) => (v >= 1 ? v.toFixed(0) + ' s' : v >= 0.01 ? (v * 1e3).toFixed(0) + ' ms' : (v * 1e3).toFixed(1) + ' ms'),
    series: [{ name: 'prefill time', color: 'var(--accent)', pts: rows.map((x) => [x.n, x.r.tPrefill]), dots: true, tips: rows.map((x) => `${fmt.n(x.n, 0)} tokens<br>${fmt.ms(x.r.tPrefill)}`) }, { name: 'if it scaled linearly', color: 'var(--faint)', dash: '5 4', pts: rows.map((x) => [x.n, base * x.n]) }],
    markers: [{ x: Math.max(W.isl, 512), y: pre.tPrefill, color: 'var(--c-comm)', label: 'you' }],
  }), { info: 'The gap between the curves is the quadratic attention term.' });

  const chunks = [1024, 2048, 4096, 8192, 16384, 32768], Wc = { ...W, isl: Math.max(W.isl, 32768) };
  const cr = chunks.map((c) => ({ c, r: evalPrefill(M, H, { ...Wc, chunk: c }, A, P, P.gpus) }));
  const chunk = cc('Chunk size trade-off', xyChart({
    w: CW, h: CH, rightAxis: true, xLog: true, xMin: 1024, xMax: 32768, xTicks: chunks, xLabel: 'prefill chunk size (tokens)', yLabel: 'input tok/s/GPU', yLabelR: 'time (ms)', xFmt: (v) => fmt.n(v, 0), yFmtR: (v) => fmt.n(v * 1e3, 0),
    series: [{ name: 'throughput', color: 'var(--accent)', pts: cr.map((x) => [x.c, x.r.thr]), dots: true, tips: cr.map((x) => `chunk ${x.c}<br>${fmt.n(x.r.thr, 0)} tok/s/GPU`) }, { name: 'time to prefill', color: 'var(--c-comm)', axis: 'r', pts: cr.map((x) => [x.c, x.r.tPrefill]), dots: true, tips: cr.map((x) => `chunk ${x.c}<br>${fmt.ms(x.r.tPrefill)}`) }],
  }), { info: 'Shown for a ≥32K prompt. Small chunks starve GEMMs and raise the bubble; large chunks delay short prompts queued behind them.' });
  const sched = cc('Chunked pipeline schedule', P.pp > 1 ? gantt({ stages: P.pp, micro: Math.min(pre.nCh, 12), kind: 'prefill', w: CW }) : '<div class="empty">PP = 1 for the prefill pool: no pipeline bubbles. Raise PP to see how chunks flow through stages.</div>', { info: 'Red cells are idle stages (fill and drain).' });
  const alone = S.poolMode === 'prefill';   // the KPI strip already shows these numbers
  return `<div class="cpane ${alone ? 'g22' : 'g22s'}">${alone ? '' : stats}${bd}${ttft}${chunk}${sched}</div>`;
}

// ================================================================= PARETO
export function paretoC(S, R) {
  const { M, H, W, A, ev } = R, sr = S.search;
  if (sr.status !== 'done') return `<div class="cpane">${cc('Pareto frontier', `<div class="empty"><span class="spinner"></span>&nbsp; Searching parallelism configurations… ${(sr.progress * 100).toFixed(0)}%<div class="progress" style="max-width:320px;margin:12px auto 0"><i style="width:${sr.progress * 100}%"></i></div></div>`)}</div>`;
  const gpus = poolGpus(S, 'decode');
  const user = sweepConfig(M, H, W, A, { ...S.par.decode, overlap: S.opt.dbo, eplb: S.opt.eplb }, gpus, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]);
  const cloud = sr.pts.filter((_, i) => i % Math.ceil(sr.pts.length / 1500) === 0), fr = sr.front, slo = W.minTokUser;
  const bestFr = fr.filter((p) => p.tokUser >= slo).sort((a, b) => b.perGpu - a.perGpu)[0], bestUs = user.filter((p) => p.tokUser >= slo).sort((a, b) => b.perGpu - a.perGpu)[0];
  const maxX = Math.max(...fr.map((p) => p.tokUser), slo * 1.5) * 1.05;
  const chart = xyChart({
    w: 640, h: 540, xMin: 0, xMax: maxX, xLabel: 'tokens / s / user  (interactivity →)', yLabel: 'tokens / s / GPU  (decode pool)', xFmt: (v) => v.toFixed(0), yFmt: (v) => fmt.n(v, 0),
    regions: [{ x0: 0, x1: slo, color: 'var(--bad)', label: 'below SLO', op: 0.07 }], vlines: [{ x: slo, label: `SLO ${slo}`, color: 'var(--bad)', dy: 16 }],
    series: [
      { name: `all evaluated (${fmt.n(sr.pts.length, 0)})`, color: 'var(--faint)', pts: cloud.map((p) => [p.tokUser, p.perGpu]), noLine: true, dots: true, r: 2, fill: 'color-mix(in srgb, var(--faint) 55%, transparent)' },
      { name: 'Pareto frontier', color: 'var(--accent)', width: 2.6, pts: fr.map((p) => [p.tokUser, p.perGpu]), dots: true, r: 3.5, tips: fr.map((p) => `<b>${family(p)}</b><br>${cfgLabel(p)}<br>batch ${p.ba}/rank<br>${p.tokUser.toFixed(0)} tok/s/user · ${fmt.n(p.perGpu, 0)} tok/s/GPU`) },
      { name: 'your parallelism, sweeping batch', color: 'var(--c-comm)', width: 2.4, pts: user.map((p) => [p.tokUser, p.perGpu]), dots: true, r: 3, tips: user.map((p) => `your config<br>batch ${p.ba}/rank<br>${p.tokUser.toFixed(0)} tok/s/user · ${fmt.n(p.perGpu, 0)} tok/s/GPU`) },
    ],
    markers: [...(bestFr ? [{ x: bestFr.tokUser, y: bestFr.perGpu, shape: 'star', color: 'var(--good)', tip: `<b>Best at SLO</b><br>${cfgLabel(bestFr)}<br>batch ${bestFr.ba}/rank` }] : []), { x: ev.dec.tokUser, y: ev.dec.perGpu, color: 'var(--c-comm)', r: 7, label: 'current', tip: 'Your current operating point' }],
  });
  const gap = bestFr && bestUs ? bestUs.perGpu / bestFr.perGpu : null;
  const stat = (lb, v, col) => `<div class="stat"><div class="lb">${lb}</div><div class="v"${col ? ` style="color:${col}"` : ''}>${v}</div></div>`;
  const stats = `<div class="stats s1">${stat(`Best possible @ ≥${slo} tok/s/user`, bestFr ? fmt.n(bestFr.perGpu, 0) + '<small class="muted"> tok/s/GPU</small>' : '–')}${stat('Your layout, same SLO', bestUs ? fmt.n(bestUs.perGpu, 0) : 'cannot meet')}${stat('Gap to optimal', gap == null ? '–' : (gap * 100).toFixed(0) + '% of optimal', gap == null ? 'var(--bad)' : gap > 0.9 ? 'var(--good)' : 'var(--warn)')}</div>`;
  const pick = fr.filter((_, i) => i % Math.max(1, Math.floor(fr.length / 5)) === 0).slice(0, 6);
  const tbl = `<div class="tblwrap"><table class="t dense"><thead><tr><th>Layout</th><th class="n">tok/s/user</th><th class="n">tok/s/GPU</th><th></th></tr></thead><tbody>${pick.map((p) => `<tr><td>${lay(p.P)}</td><td class="n">${p.tokUser.toFixed(0)}</td><td class="n">${fmt.n(p.perGpu, 0)}</td><td><button class="btn sm" data-act="apply-frontier" data-i="${fr.indexOf(p)}">Apply</button></td></tr>`).join('')}</tbody></table></div>`;
  return `<div class="cpane gpar">${cc('Throughput vs interactivity', chart, { info: 'Every dot is a (layout, batch size) pair. Move right for happier users, up to serve more per GPU.' })}<div class="stackcol">${cc('At your SLO', stats)}${cc('On the frontier', tbl, { info: 'Click Apply to load a layout into the sliders.' })}</div></div>`;
}

// ================================================================= SWEEPS
export function sweepsC(S, R) {
  const { M, H, W, A, ev } = R, P = ev.dP;
  const cols = CTX.map((c) => {
    const mem = memoryPerGpu(M, H, { ...W, isl: c }, A, P, P.D), ba = Math.min(W.batchPerRank, mem.perRankMax);
    if (ba < 1) return { label: ctxLabel(c), v: {}, oom: true, y: 0 };
    const r = evalDecode(M, H, { ...W, isl: c }, A, P, ba), v = {}; r.segs.filter((s) => !s.hidden).forEach((s) => (v[s.key] = s.ms));
    return { label: ctxLabel(c), v, oom: false, y: mem.bMaxRep, tip: `${ctxLabel(c)} context<br>${fmt.n(mem.bMaxRep, 0)} max concurrent requests<br>TPOT ${fmt.ms(r.tpot)} at batch ${ba}/rank` };
  });
  const keys = [['attnProj', 'Attn proj'], ['kv', 'Attn core (KV)'], ['dense', 'Dense / shared'], ['moe', 'Experts'], ['comm', 'Comm'], ['pp', 'PP'], ['over', 'Overheads']].map(([key, label]) => ({ key, label, color: COLORS[key] }));
  const ctx = cc('What limits decode as context grows?', stackedColumns({ cols, keys, w: CW, h: 420, xLabel: 'context length (tokens)', line: { label: 'max concurrent requests', pts: cols.map((c) => ({ y: c.y, tip: c.tip || 'does not fit' })) } }), { info: 'Your current layout, batch capped to fit. Attention / KV eats the step at long context while capacity collapses.' });

  const rg = S.regime; let body;
  if (!rg || rg.status !== 'done') body = '<div class="empty"><span class="spinner"></span>&nbsp; Computing the regime map…</div>';
  else {
    const maxLog = Math.log10(rg.max), minLog = Math.log10(rg.min);
    const cells = rg.cells.map((c) => {
      if (!c.best) return { i: c.i, j: c.j, fill: 'var(--panel2)', text: '–', textColor: 'var(--faint)', tip: `${ctxLabel(rg.xs[c.i])} ctx · ≥${rg.ys[c.j]} tok/s/user<br>no layout fits or meets the SLO` };
      const t = (Math.log10(c.best.perGpu) - minLog) / (maxLog - minLog || 1), col = FAM_COLORS[c.fam];
      return { i: c.i, j: c.j, fill: `color-mix(in srgb, ${col} ${Math.round(28 + t * 72)}%, var(--panel))`, text: FAM_SHORT[c.fam], textColor: t > 0.45 ? '#fff' : 'var(--ink)', tip: `<b>${c.fam}</b><br>${cfgLabel(c.best)} · batch ${c.best.ba}/rank<br>${fmt.n(c.best.perGpu, 0)} tok/s/GPU @ ${c.best.tokUser.toFixed(0)} tok/s/user` };
    });
    body = heatmap({ xs: rg.xs, ys: rg.ys, cells, w: CW, h: 400, xLabel: 'context length (tokens)', yLabel: 'min tokens / s / user', xFmt: ctxLabel, yFmt: (v) => v, legend: `<div class="legend">${Object.keys(FAM_COLORS).map((k) => `<span><i style="background:${FAM_COLORS[k]}"></i>${k}</span>`).join('')}<span class="faint">darker = more tok/s/GPU</span></div>` });
  }
  return `<div class="cpane g12">${ctx}${cc('Regime map: which parallelism wins?', body, { info: 'For each (context, interactivity floor), the best layout found by the search. Watch the winning family change as context grows.' })}</div>`;
}

// ================================================================= OPTIMIZER
export function optimizerC(S, R) {
  const { W, ev } = R, O = S.optim, sr = S.search, obj = O.objective, pm = S.poolMode;
  const fixedObj = { split: `Max goodput s.t. tokens/s/user ≥ ${W.minTokUser} and TTFT ≤ ${W.ttftMs} ms`, prefill: `Max input tokens/s/GPU s.t. prefill latency within the TTFT budget (${W.ttftMs} ms)` };
  const objField = pm === 'decode'
    ? `<div class="f"><label>Objective</label><select data-o="objective" style="width:280px"><option value="slo" ${obj === 'slo' ? 'selected' : ''}>Max throughput s.t. tok/s/user ≥ X</option><option value="thr" ${obj === 'thr' ? 'selected' : ''}>Max total throughput (no latency floor)</option><option value="lat" ${obj === 'lat' ? 'selected' : ''}>Max tok/s/user s.t. throughput ≥ Z</option></select></div>
       <div class="f"><label>Min tok/s/user (X)</label><input type="number" data-w="minTokUser" value="${W.minTokUser}" ${obj === 'thr' ? 'disabled' : ''}></div>
       <div class="f"><label>Min tok/s/GPU (Z)</label><input type="number" data-o="minGpu" value="${O.minGpu}" ${obj === 'lat' ? '' : 'disabled'}></div>`
    : `<div class="f"><label>Objective</label><div class="hint" style="max-width:420px;padding:6px 0">${fixedObj[pm]}</div></div>
       ${pm === 'split' ? `<div class="f"><label>Min tok/s/user (X)</label><input type="number" data-w="minTokUser" value="${W.minTokUser}"></div>` : ''}
       <div class="f"><label>Max TTFT (ms)</label><input type="number" data-w="ttftMs" value="${W.ttftMs}"></div>`;
  const form = `<div class="formrow">${objField}
    <button class="btn primary" data-act="run-opt" ${O.running ? 'disabled' : ''}>${O.running ? '<span class="spinner"></span>&nbsp;Searching…' : 'Find best configuration'}</button></div>
    ${O.running ? `<div class="progress" style="margin-top:10px"><i style="width:${Math.max(pm === 'decode' ? sr.progress : 0, O.fake) * 100}%"></i></div>` : ''}`;
  const head = cc('Optimizer', form, { info: pm === 'split' ? 'For each prefill/decode split of the rack, finds the best layout on each side, then rate-matches them: the slower pool sets the request rate.' : 'Enumerates every valid factorization of the pool, drops what doesn’t fit in memory, finds the largest batch that meets the SLO for each, and ranks the rest.', cls: 'auto' });
  const rs = O.results;
  if (!rs || rs.mode !== pm) return `<div class="cpane gopt">${head}<section class="card cc"><div class="bd"><div class="empty">Press <b>Find best configuration</b>.</div></div></section></div>`;
  const usd = (row) => (rs.cost ? `<td class="n">${fmt.usd(rs.cost(row))}</td>` : '');
  const costTh = rs.cost ? `<th class="n">$/Mtok ${pm === 'prefill' ? 'in' : 'out'}</th>` : '';
  const why = cc('Why this layout', `<p class="prose">${rs.why}</p>`);
  const btn = (act, i) => `<td><button class="btn sm ${i === 0 ? 'primary' : ''}" data-act="${act}" data-i="${i}">Apply</button></td>`;
  if (pm === 'split') {
    const rows = rs.rows.map((r, i) => `<tr class="${i === 0 ? 'me' : ''}"><td>${i + 1}</td><td class="n"><b>${r.gp} : ${r.gd}</b></td><td>${lay(r.pre.P)}<div class="hint">chunk ${fmt.n(r.pre.chunk, 0)} · ${fmt.n(r.rp, 1)} req/s</div></td><td>${lay(r.dec.P, family(r.dec))}<div class="hint">batch ${r.dec.ba}/rank · ${fmt.n(r.rd, 1)} req/s</div></td><td class="n">${r.dec.tokUser.toFixed(0)}</td><td class="n">${fmt.n(r.tokPerGpu, 0)}</td>${usd(r)}${btn('apply-split-row', i)}</tr>`).join('');
    const table = cc('Best splits of the rack', `<div class="tblwrap"><table class="t dense"><thead><tr><th>#</th><th class="n">prefill : decode</th><th>Prefill layout</th><th>Decode layout</th><th class="n">tok/s/user</th><th class="n">tok/s/GPU</th>${costTh}<th></th></tr></thead><tbody>${rows}</tbody></table></div>`, { info: 'tok/s/GPU is end-to-end output tokens per second per GPU of the whole rack.' });
    return `<div class="cpane gopt">${head}<div class="optbody wide">${table}<div class="stackcol">${why}</div></div></div>`;
  }
  if (pm === 'prefill') {
    const rows = rs.rows.map((r, i) => `<tr class="${i === 0 ? 'me' : ''}"><td>${i + 1}</td><td>${lay(r.P)}</td><td class="n">${fmt.n(r.chunk, 0)}</td><td class="n">${fmt.ms(r.ttft)}</td><td class="n">${fmt.n(r.reqPerS, 1)}</td><td class="n">${fmt.n(r.thr, 0)}</td>${usd(r)}${btn('apply-prefill-row', i)}</tr>`).join('');
    const table = cc('Top prefill layouts', `<div class="tblwrap"><table class="t dense"><thead><tr><th>#</th><th>Layout</th><th class="n">chunk</th><th class="n">latency</th><th class="n">req/s (rack)</th><th class="n">in tok/s/GPU</th>${costTh}<th></th></tr></thead><tbody>${rows}</tbody></table></div>`);
    return `<div class="cpane gopt">${head}<div class="optbody wide">${table}<div class="stackcol">${why}</div></div></div>`;
  }
  const best = rs.top[0];
  const rows = rs.top.map((p, i) => `<tr class="${i === 0 ? 'me' : ''}"><td>${i + 1}</td><td>${lay(p.P, family(p))}</td><td class="n">${p.ba}</td><td class="n">${p.tokUser.toFixed(0)}</td><td class="n">${fmt.n(p.perGpu, 0)}</td>${usd(p)}${btn('apply-opt', i)}</tr>`).join('');
  const table = cc('Top configurations', `<div class="tblwrap"><table class="t dense"><thead><tr><th>#</th><th>Layout</th><th class="n">batch/rank</th><th class="n">tok/s/user</th><th class="n">tok/s/GPU</th>${costTh}<th></th></tr></thead><tbody>${rows}</tbody></table></div>`, { info: `${fmt.n(rs.considered, 0)} points evaluated · ${fmt.n(rs.feasible, 0)} satisfy the constraints`, right: `<span class="muted">${fmt.n(rs.feasible, 0)} of ${fmt.n(rs.considered, 0)} feasible</span>` });
  const ok = rs.robust.wins / rs.robust.n > 0.7;
  const rob = cc('Robustness', `<div class="kv tight"><span>Winner holds in</span><span>${rs.robust.wins} / ${rs.robust.n} scenarios</span><span>Runner-ups within 5%</span><span>${rs.close} layouts</span></div><div class="progress" style="margin-top:8px"><i style="width:${(rs.robust.wins / rs.robust.n) * 100}%;background:${ok ? 'var(--good)' : 'var(--warn)'}"></i></div><div class="hint" style="margin-top:5px">${ok ? 'Robust to ±20% efficiency, 2× collective latency, 0/100% overlap.' : 'Fragile: small assumption changes reorder the top layouts. Calibrate first.'}</div>`);
  return `<div class="cpane gopt">${head}<div class="optbody">${table}<div class="stackcol">${why}${rob}</div></div></div>`;
}
