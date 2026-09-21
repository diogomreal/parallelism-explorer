// Tiny dependency-free SVG chart helpers. Everything returns an HTML/SVG string.
// Colors come from CSS variables so light/dark theming "just works".

export const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export const fmt = {
  n(x, d = 1) { if (!isFinite(x)) return '–'; const a = Math.abs(x); if (a >= 1e12) return (x / 1e12).toFixed(d) + 'T'; if (a >= 1e9) return (x / 1e9).toFixed(d) + 'B'; if (a >= 1e6) return (x / 1e6).toFixed(d) + 'M'; if (a >= 1e4) return (x / 1e3).toFixed(0) + 'k'; if (a >= 1e3) return (x / 1e3).toFixed(1) + 'k'; return a >= 100 ? x.toFixed(0) : a >= 10 ? x.toFixed(1) : x.toFixed(2); },
  bytes(x) { const a = Math.abs(x); if (a >= 1e12) return (x / 1e12).toFixed(2) + ' TB'; if (a >= 1e9) return (x / 1e9).toFixed(a >= 1e11 ? 0 : 1) + ' GB'; if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e8 ? 0 : 1) + ' MB'; if (a >= 1e3) return (x / 1e3).toFixed(1) + ' KB'; return x.toFixed(0) + ' B'; },
  ms(s) { const ms = s * 1e3; return ms >= 100 ? ms.toFixed(0) + ' ms' : ms >= 10 ? ms.toFixed(1) + ' ms' : ms >= 1 ? ms.toFixed(2) + ' ms' : (ms * 1e3).toFixed(0) + ' µs'; },
  flops(x) { return x >= 1e15 ? (x / 1e15).toFixed(1) + ' PF/s' : x >= 1e12 ? (x / 1e12).toFixed(0) + ' TF/s' : (x / 1e9).toFixed(0) + ' GF/s'; },
  pct(x) { return (x * 100).toFixed(x < 0.1 ? 1 : 0) + '%'; },
  usd(x) { return '$' + (x >= 10 ? x.toFixed(1) : x.toFixed(2)); },
};

function scale(d0, d1, r0, r1, log) {
  if (log) { const l0 = Math.log10(d0), l1 = Math.log10(d1); return (v) => r0 + ((Math.log10(Math.max(v, d0 * 1e-9)) - l0) / (l1 - l0)) * (r1 - r0); }
  return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
}
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function logTicks(lo, hi) { const out = []; for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) out.push(Math.pow(10, e)); return out; }

const tipAttr = (t) => (t ? ` data-tip="${esc(t)}"` : '');

// ------------------------------------------------------------------ generic XY chart
// series: [{name, color, pts:[[x,y]], dash, axis:'l'|'r', width, dots, area, fill}]
export function xyChart(o) {
  const W = o.w || 640, H = o.h || 300, m = { l: 52, r: o.rightAxis ? 52 : 14, t: 14, b: 40, ...(o.margin || {}) };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const all = o.series.flatMap((s) => s.pts);
  const xs = all.map((p) => p[0]), ys = (ax) => o.series.filter((s) => (s.axis || 'l') === ax).flatMap((s) => s.pts.map((p) => p[1]));
  const x0 = o.xMin ?? Math.min(...xs), x1 = o.xMax ?? Math.max(...xs);
  const rng = (ax, log) => { const v = ys(ax); if (!v.length) return [0, 1]; let lo = log ? Math.min(...v.filter((z) => z > 0)) : (o.yMin ?? 0), hi = Math.max(...v); if (log) { lo = Math.pow(10, Math.floor(Math.log10(lo))); hi = Math.pow(10, Math.ceil(Math.log10(hi))); } else hi = hi * 1.08; return [lo, hi]; };
  const [yl0, yl1] = rng('l', o.yLog), [yr0, yr1] = o.rightAxis ? rng('r', false) : [0, 1];
  const sx = scale(x0, x1, m.l, m.l + iw, o.xLog), syL = scale(yl0, yl1, m.t + ih, m.t, o.yLog), syR = scale(yr0, yr1, m.t + ih, m.t, false);
  let g = '';
  const xt = o.xTicks || (o.xLog ? logTicks(x0, x1) : niceTicks(x0, x1, 6));
  const yt = o.yLog ? logTicks(yl0, yl1) : niceTicks(yl0, yl1, 5);
  xt.forEach((t) => { const x = sx(t); g += `<line class="grid" x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ih}"/><text class="tick" x="${x}" y="${m.t + ih + 16}" text-anchor="middle">${esc((o.xFmt || fmt.n)(t))}</text>`; });
  yt.forEach((t) => { const y = syL(t); g += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}"/><text class="tick" x="${m.l - 8}" y="${y + 4}" text-anchor="end">${esc((o.yFmt || fmt.n)(t))}</text>`; });
  if (o.rightAxis) niceTicks(yr0, yr1, 5).forEach((t) => { const y = syR(t); g += `<text class="tick" x="${m.l + iw + 8}" y="${y + 4}" text-anchor="start">${esc((o.yFmtR || fmt.n)(t))}</text>`; });
  g += `<line class="axis" x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/>`;
  if (o.xLabel) g += `<text class="axlabel" x="${m.l + iw / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel)}</text>`;
  if (o.yLabel) g += `<text class="axlabel" ${o.yColor ? `fill="${o.yColor}"` : ''} transform="translate(13 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel)}</text>`;
  if (o.yLabelR) g += `<text class="axlabel" ${o.yColorR ? `fill="${o.yColorR}"` : ''} transform="translate(${W - 8} ${m.t + ih / 2}) rotate(90)" text-anchor="middle">${esc(o.yLabelR)}</text>`;
  (o.regions || []).forEach((r) => { const xa = sx(r.x0), xb = sx(r.x1); g += `<rect x="${Math.min(xa, xb)}" y="${m.t}" width="${Math.abs(xb - xa)}" height="${ih}" fill="${r.color}" opacity="${r.op ?? 0.08}"/>`; if (r.label) g += `<text class="note" x="${(xa + xb) / 2}" y="${m.t + 13}" text-anchor="middle">${esc(r.label)}</text>`; });
  (o.vlines || []).forEach((v) => { const x = sx(v.x); g += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ih}" stroke="${v.color || 'var(--bad)'}" stroke-dasharray="5 4" stroke-width="1.4"/>${v.label ? `<text class="note" x="${x + 5}" y="${m.t + 12 + (v.dy || 0)}" fill="${v.color || 'var(--bad)'}">${esc(v.label)}</text>` : ''}`; });
  o.series.forEach((s) => {
    if (!s.pts.length) return;
    const sy = (s.axis || 'l') === 'r' ? syR : syL;
    const pts = s.pts.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`);
    if (s.area) g += `<polygon points="${sx(s.pts[0][0])},${m.t + ih} ${pts.join(' ')} ${sx(s.pts[s.pts.length - 1][0])},${m.t + ih}" fill="${s.color}" opacity=".12"/>`;
    if (s.pts.length > 1 && !s.noLine) g += `<polyline fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} stroke-linejoin="round" points="${pts.join(' ')}"/>`;
    if (s.dots) s.pts.forEach((p, i) => { g += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="${s.r || 3}" fill="${s.fill || s.color}" stroke="var(--panel)" stroke-width="1"${tipAttr(s.tips && s.tips[i])}/>`; });
  });
  (o.markers || []).forEach((k) => {
    const x = sx(k.x), y = (k.axis === 'r' ? syR : syL)(k.y);
    if (k.shape === 'star') g += `<path d="${starPath(x, y, 9, 4)}" fill="${k.color}" stroke="var(--panel)" stroke-width="1.2"${tipAttr(k.tip)}/>`;
    else g += `<circle cx="${x}" cy="${y}" r="${k.r || 5}" fill="${k.color}" stroke="var(--panel)" stroke-width="1.5"${tipAttr(k.tip)}/>`;
    if (k.label) g += `<text class="note" x="${x + 8}" y="${y - 8}" fill="${k.color}" font-weight="600">${esc(k.label)}</text>`;
  });
  const legend = o.legend === false ? '' : `<div class="legend">${o.series.filter((s) => s.name && !s.hideLegend).map((s) => `<span><i style="background:${s.color};${s.dash ? 'background:none;border-top:2px dashed ' + s.color : ''}"></i>${esc(s.name)}</span>`).join('')}${(o.legendExtra || []).map((l) => `<span><i style="background:${l.color};border-radius:50%"></i>${esc(l.name)}</span>`).join('')}</div>`;
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || 'chart')}">${g}</svg>${legend}</div>`;
}
function starPath(cx, cy, R, r) { let p = ''; for (let i = 0; i < 10; i++) { const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? r : R; p += (i ? 'L' : 'M') + (cx + Math.cos(a) * rr).toFixed(1) + ' ' + (cy + Math.sin(a) * rr).toFixed(1); } return p + 'Z'; }

// ------------------------------------------------------------------ roofline
export function roofline({ peak, bw, peakLabel, points, w = 640, h = 340 }) {
  const m = { l: 66, r: 16, t: 14, b: 42 }, iw = w - m.l - m.r, ih = h - m.t - m.b;
  const ridge0 = peak / bw, lo = (k) => Math.pow(10, Math.floor(Math.log10(k))), hi = (k) => Math.pow(10, Math.ceil(Math.log10(k)));
  const ais = points.map((p) => p.ai).filter((v) => v > 0), perfs = points.map((p) => p.perf).filter((v) => v > 0);
  const x0 = Math.max(0.1, lo(Math.min(...ais, ridge0 / 30) / 1.5)), x1 = hi(Math.max(...ais, ridge0 * 8) * 1.5);
  const y0 = Math.max(1e10, lo(Math.min(...perfs, peak / 100) / 1.5)), y1 = peak * 3;
  const sx = scale(x0, x1, m.l, m.l + iw, true), sy = scale(y0, y1, m.t + ih, m.t, true);
  let g = '';
  logTicks(x0, x1).forEach((t) => { const x = sx(t); g += `<line class="grid" x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ih}"/><text class="tick" x="${x}" y="${m.t + ih + 16}" text-anchor="middle">${t >= 1000 ? t / 1000 + 'k' : t}</text>`; });
  logTicks(y0, y1).forEach((t) => { const y = sy(t); g += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}"/><text class="tick" x="${m.l - 8}" y="${y + 4}" text-anchor="end">${t >= 1e15 ? t / 1e15 + ' PF' : t / 1e12 + ' TF'}</text>`; });
  g += `<line class="axis" x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/>`;
  g += `<text class="axlabel" x="${m.l + iw / 2}" y="${h - 6}" text-anchor="middle">arithmetic intensity (FLOP / byte of HBM traffic)</text><text class="axlabel" transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">achieved FLOP/s per GPU</text>`;
  const ridge = peak / bw;
  const xr = sx(ridge), yp = sy(peak);
  g += `<polygon points="${sx(x0)},${sy(bw * x0)} ${xr},${yp} ${xr},${m.t + ih} ${sx(x0)},${m.t + ih}" fill="var(--c-kv)" opacity=".06"/><polygon points="${xr},${yp} ${m.l + iw},${yp} ${m.l + iw},${m.t + ih} ${xr},${m.t + ih}" fill="var(--c-moe)" opacity=".06"/>`;
  g += `<polyline fill="none" stroke="var(--ink)" stroke-width="2" points="${sx(x0)},${sy(bw * x0)} ${xr},${yp} ${m.l + iw},${yp}"/>`;
  g += `<line x1="${xr}" x2="${xr}" y1="${yp}" y2="${m.t + ih}" stroke="var(--muted)" stroke-dasharray="3 4"/><text class="note" x="${xr + 5}" y="${m.t + ih - 6}">ridge ≈ ${fmt.n(ridge, 0)} FLOP/B</text>`;
  g += `<text class="note" x="${m.l + 8}" y="${m.t + 14}">memory-bound</text><text class="note" x="${m.l + iw - 8}" y="${m.t + 14}" text-anchor="end">compute-bound · ${esc(peakLabel)}</text>`;
  const placed = points.map((p) => ({ p, x: sx(Math.min(Math.max(p.ai, x0 * 1.05), x1 * 0.95)), y: sy(Math.min(Math.max(p.perf, y0 * 1.05), y1 * 0.9)) }));
  const byY = [...placed].sort((a, b) => a.y - b.y); let lastY = -99;
  byY.forEach((o) => { o.ly = Math.max(o.y, lastY + 15); lastY = o.ly; });      // nudge labels apart vertically
  placed.forEach((o) => {
    const anchorLeft = o.x > m.l + iw - 150;
    g += `<circle cx="${o.x}" cy="${o.y}" r="6.5" fill="${o.p.color}" stroke="var(--panel)" stroke-width="2"${tipAttr(o.p.tip)}/>`;
    if (Math.abs(o.ly - o.y) > 3) g += `<line x1="${o.x + (anchorLeft ? -6 : 6)}" y1="${o.y}" x2="${o.x + (anchorLeft ? -14 : 14)}" y2="${o.ly}" stroke="var(--faint)" stroke-width="1"/>`;
    g += `<text class="note" x="${anchorLeft ? o.x - 16 : o.x + 16}" y="${o.ly + 4}" text-anchor="${anchorLeft ? 'end' : 'start'}" font-weight="600" fill="var(--ink)">${esc(o.p.name)}</text>`;
  });
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="roofline">${g}</svg></div>`;
}

// ------------------------------------------------------------------ stacked horizontal bar (HTML)
export function stackBar(segs, { total, unit = 'ms', height = 30, showLabels = true, fmtv } = {}) {
  const tot = total ?? segs.reduce((a, s) => a + s.v, 0);
  const f = fmtv || ((v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' ' + unit);
  return `<div class="stack" style="height:${height}px">${segs.map((s) => { const p = (s.v / tot) * 100; if (p <= 0.05) return ''; return `<div class="seg${s.hatch ? ' hatch' : ''}" style="width:${p}%;--c:${s.color}"${tipAttr(`<b>${s.label}</b><br>${f(s.v)} · ${p.toFixed(1)}%`)}>${showLabels && p > 9 ? `<span>${p.toFixed(0)}%</span>` : ''}</div>`; }).join('')}</div>`;
}
export function legendRows(segs, tot, f) {
  return `<div class="lrows">${segs.map((s) => `<div class="lrow"><i class="sw${s.hatch ? ' hatch' : ''}" style="--c:${s.color}"></i><span class="ln">${esc(s.label)}</span><span class="lv">${f(s.v)}</span><span class="lp">${((s.v / tot) * 100).toFixed(0)}%</span></div>`).join('')}</div>`;
}

// ------------------------------------------------------------------ vertical stacked (normalized) columns
export function stackedColumns({ cols, keys, w = 640, h = 280, xLabel, yLabel = 'share of step time', line }) {
  const m = { l: 44, r: line ? 54 : 12, t: 14, b: 46 }, iw = w - m.l - m.r, ih = h - m.t - m.b;
  const bw = iw / cols.length, pad = Math.min(8, bw * 0.15);
  let g = '';
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => { const y = m.t + ih * (1 - t); g += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}"/><text class="tick" x="${m.l - 6}" y="${y + 4}" text-anchor="end">${t * 100}%</text>`; });
  cols.forEach((c, i) => {
    let y = m.t + ih; const tot = keys.reduce((a, k) => a + (c.v[k.key] || 0), 0) || 1;
    if (c.oom) g += `<rect x="${m.l + i * bw + pad}" y="${m.t}" width="${bw - pad * 2}" height="${ih}" rx="4" fill="var(--bad)" opacity=".10"${tipAttr(`<b>${c.label}</b><br>KV for even one request does not fit`)}/><text class="note" x="${m.l + i * bw + bw / 2}" y="${m.t + ih / 2}" text-anchor="middle" fill="var(--bad)" font-weight="700">OOM</text>`;
    keys.forEach((k) => { const hh = ((c.v[k.key] || 0) / tot) * ih; y -= hh; if (hh > 0.3) g += `<rect x="${m.l + i * bw + pad}" y="${y}" width="${bw - pad * 2}" height="${hh}" fill="${k.color}"${tipAttr(`<b>${c.label}</b> · ${k.label}<br>${(((c.v[k.key] || 0) / tot) * 100).toFixed(0)}% of step`)}/>`; });
    g += `<text class="tick" x="${m.l + i * bw + bw / 2}" y="${m.t + ih + 16}" text-anchor="middle">${esc(c.label)}</text>`;
  });
  if (line && line.pts.some((p) => p.y > 0)) {
    const mx = Math.max(...line.pts.map((p) => p.y)) * 1.1;
    const pts = line.pts.map((p, i) => `${m.l + i * bw + bw / 2},${m.t + ih - (p.y / mx) * ih}`);
    g += `<polyline fill="none" stroke="var(--ink)" stroke-width="2" stroke-dasharray="1 0" points="${pts.join(' ')}"/>`;
    line.pts.forEach((p, i) => { g += `<circle cx="${m.l + i * bw + bw / 2}" cy="${m.t + ih - (p.y / mx) * ih}" r="3.5" fill="var(--ink)" stroke="var(--panel)"${tipAttr(p.tip)}/>`; });
    [0, 0.5, 1].forEach((t) => { g += `<text class="tick" x="${m.l + iw + 8}" y="${m.t + ih - t * ih + 4}">${fmt.n((mx * t) / 1.0, 0)}</text>`; });
    g += `<text class="axlabel" transform="translate(${w - 6} ${m.t + ih / 2}) rotate(90)" text-anchor="middle">${esc(line.label)}</text>`;
  }
  g += `<text class="axlabel" x="${m.l + iw / 2}" y="${h - 6}" text-anchor="middle">${esc(xLabel || '')}</text><text class="axlabel" transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  const legend = `<div class="legend">${keys.map((k) => `<span><i style="background:${k.color}"></i>${esc(k.label)}</span>`).join('')}${line ? `<span><i style="background:var(--ink)"></i>${esc(line.label)}</span>` : ''}</div>`;
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}">${g}</svg>${legend}</div>`;
}

// ------------------------------------------------------------------ heatmap
export function heatmap({ xs, ys, cells, w = 720, h = 330, xLabel, yLabel, xFmt, yFmt, legend }) {
  const m = { l: 58, r: 14, t: 10, b: 44 }, iw = w - m.l - m.r, ih = h - m.t - m.b, cw = iw / xs.length, ch = ih / ys.length;
  let g = '';
  ys.forEach((y, j) => { const yy = m.t + ih - (j + 1) * ch; g += `<text class="tick" x="${m.l - 8}" y="${yy + ch / 2 + 4}" text-anchor="end">${esc(yFmt(y))}</text>`; });
  xs.forEach((x, i) => { g += `<text class="tick" x="${m.l + i * cw + cw / 2}" y="${m.t + ih + 16}" text-anchor="middle">${esc(xFmt(x))}</text>`; });
  cells.forEach((c) => {
    const x = m.l + c.i * cw, y = m.t + ih - (c.j + 1) * ch;
    g += `<rect x="${x + 1}" y="${y + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" fill="${c.fill}" ${tipAttr(c.tip)}/>`;
    if (c.text) g += `<text x="${x + cw / 2}" y="${y + ch / 2 + 4}" text-anchor="middle" class="cell" fill="${c.textColor || '#fff'}" pointer-events="none">${esc(c.text)}</text>`;
  });
  g += `<text class="axlabel" x="${m.l + iw / 2}" y="${h - 6}" text-anchor="middle">${esc(xLabel)}</text><text class="axlabel" transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}">${g}</svg>${legend || ''}</div>`;
}

// ------------------------------------------------------------------ pipeline Gantt
export function gantt({ stages, micro, w = 640, tF = 1, kind = 'prefill' }) {
  const m = { l: 62, r: 10, t: 8, b: 26 }, rowH = Math.min(24, 150 / stages), gap = 4, h = m.t + m.b + stages * (rowH + gap);
  const total = micro + stages - 1, iw = w - m.l - m.r, cw = iw / total;
  let g = '';
  for (let s = 0; s < stages; s++) {
    const y = m.t + s * (rowH + gap);
    g += `<text class="tick" x="${m.l - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">stage ${s + 1}</text>`;
    for (let k = 0; k < micro; k++) { const x = m.l + (k + s) * cw; g += `<rect x="${x + 0.5}" y="${y}" width="${cw - 1}" height="${rowH}" rx="3" fill="var(--c-attn)" opacity="${0.55 + 0.45 * ((k % 3) / 2)}"${tipAttr(`${kind === 'prefill' ? 'chunk' : 'microbatch'} ${k + 1} on stage ${s + 1}`)}/>`; }
    for (let k = 0; k < total; k++) { if (k < s || k >= s + micro) g += `<rect x="${m.l + k * cw + 0.5}" y="${y}" width="${cw - 1}" height="${rowH}" rx="3" fill="var(--bad)" opacity=".14"${tipAttr('idle (pipeline bubble)')}/>`; }
  }
  g += `<text class="axlabel" x="${m.l + iw / 2}" y="${h - 6}" text-anchor="middle">time →  (${micro} ${kind === 'prefill' ? 'chunks' : 'microbatches'}, ${stages} stages)</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}">${g}</svg></div>`;
}

// ------------------------------------------------------------------ per-layer two-lane timeline
export function laneTimeline({ compute, comm, w = 640, overlap }) {
  const m = { l: 72, r: 10, t: 10, b: 26 }, laneH = 34;
  const cTot = compute.reduce((a, c) => a + c.t, 0), mTot = comm.reduce((a, c) => a + c.t, 0);
  const span = overlap ? Math.max(cTot, mTot) * 1.02 : (cTot + mTot) * 1.02;
  const iw = w - m.l - m.r, sx = (t) => m.l + (t / span) * iw;
  let g = '', t = 0;
  g += `<text class="tick" x="${m.l - 8}" y="${m.t + laneH / 2 + 4}" text-anchor="end">compute</text><text class="tick" x="${m.l - 8}" y="${m.t + laneH + 8 + laneH / 2 + 4}" text-anchor="end">comm</text>`;
  compute.forEach((c) => { const x = sx(t), ww = Math.max(sx(t + c.t) - x - 1, 1); g += `<rect x="${x}" y="${m.t}" width="${ww}" height="${laneH}" rx="4" fill="${c.color}"${tipAttr(`<b>${c.name}</b><br>${fmt.ms(c.t)}`)}/>${ww > 46 ? `<text class="cell" x="${x + ww / 2}" y="${m.t + laneH / 2 + 4}" text-anchor="middle" fill="#fff" pointer-events="none">${esc(c.short || c.name)}</text>` : ''}`; t += c.t; });
  let t2 = overlap ? cTot * 0.0 : cTot;
  const y2 = m.t + laneH + 8;
  comm.forEach((c, i) => { const start = overlap ? (cTot * (0.16 + 0.34 * i)) : t2; const x = sx(start), ww = Math.max(sx(start + c.t) - x - 1, 1); g += `<rect x="${x}" y="${y2}" width="${ww}" height="${laneH}" rx="4" fill="var(--c-comm)" ${overlap ? 'opacity=".8"' : ''}${tipAttr(`<b>${c.name}</b><br>${fmt.ms(c.t)}${overlap ? ' (overlapped with the other microbatch)' : ' (exposed)'}`)}/>${ww > 46 ? `<text class="cell" x="${x + ww / 2}" y="${y2 + laneH / 2 + 4}" text-anchor="middle" fill="#fff" pointer-events="none">${esc(c.short || c.name)}</text>` : ''}`; t2 += c.t; });
  const h = y2 + laneH + m.b;
  g += `<text class="axlabel" x="${m.l + iw / 2}" y="${h - 6}" text-anchor="middle">one layer · ${fmt.ms(overlap ? Math.max(cTot, mTot) : cTot + mTot)} ${overlap ? '(comm hidden behind the other microbatch)' : '(comm on the critical path)'}</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}">${g}</svg></div>`;
}
