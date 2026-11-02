/* Block glyphs. Every block shares one rounded frame; inside it a neutral
   miniature of the block's mechanism says what kind of block it is. Data colors
   stay on the lines: a glyph only takes its output color while processing. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  function el(parent, tag, attrs = {}, text) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = text;
    parent.appendChild(n);
    return n;
  }
  const fill = o => ({fill: 'currentColor', 'fill-opacity': o});
  const stroke = (o, w = .7) => ({fill: 'none', stroke: 'currentColor', 'stroke-opacity': o, 'stroke-width': w,
    'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
  const cell = (g, x, y, size, o) => el(g, 'rect', {x, y, width: size, height: size, rx: 1, ...fill(o)});
  const tag = (g, x, y, text, anchor = 'end') =>
    el(g, 'text', {x, y, 'text-anchor': anchor, class: 'nf-glyph-text'}, text);

  const ACTIVATIONS = {
    silu: x => x / (1 + Math.exp(-x)),
    gelu: x => .5 * x * (1 + Math.tanh(.7978845608 * (x + .044715 * x ** 3))),
    relu: x => Math.max(0, x),
    sigmoid: x => 1 / (1 + Math.exp(-x)),
    tanh: x => Math.tanh(x)
  };
  function activation(g, x, y, kind) {
    const f = ACTIVATIONS[kind], xs = Array.from({length: 17}, (_, i) => -5 + i * .5), vs = xs.map(f);
    const lo = Math.min(...vs), hi = Math.max(...vs);
    el(g, 'path', {...stroke(.75, 1.2), d: xs.map((v, i) =>
      `${i ? 'L' : 'M'}${(x + (v + 5) / 8 * 18).toFixed(1)} ${(y + 12 - (vs[i] - lo) / (hi - lo) * 12).toFixed(1)}`).join(' ')});
  }

  /* Fully connected columns: node count shows the width change, every
     input feeds every output. */
  function dense(g, cx, cy, counts, spacing = 26, pitch = 7) {
    const columns = counts.map((n, i) => Array.from({length: n}, (_, j) =>
      ({x: cx + (i - (counts.length - 1) / 2) * spacing, y: cy + (j - (n - 1) / 2) * pitch})));
    for (let i = 1; i < columns.length; i++)
      for (const a of columns[i - 1]) for (const b of columns[i])
        el(g, 'line', {x1: a.x, y1: a.y, x2: b.x, y2: b.y, ...stroke(.2, .6)});
    columns.flat().forEach(p => el(g, 'circle', {cx: p.x, cy: p.y, r: 2.2, ...fill(.6)}));
  }

  /* softmax(QKᵀ)V: queries on the left, keys above and values below the
     weight map, so each key column carries its value. */
  function attention(g, cx, cy, {keys = 4, queries = 4, pitch = 6.5, size = 5.5, values = true, labels = true}) {
    const w = keys * pitch, h = queries * pitch, x0 = cx - w / 2 + 6, y0 = cy - h / 2 + (values ? 0 : 4);
    for (let i = 0; i < queries; i++) cell(g, x0 - 4 - size, y0 + i * pitch, size, .5);
    for (let j = 0; j < keys; j++) cell(g, x0 + j * pitch, y0 - 4 - size, size, .5);
    if (values) for (let j = 0; j < keys; j++) cell(g, x0 + j * pitch, y0 + h + 4 - (pitch - size), size, .38);
    for (let i = 0; i < queries; i++) {
      const focus = (i * 2 + 1) % keys;
      for (let j = 0; j < keys; j++) cell(g, x0 + j * pitch, y0 + i * pitch, size, j === focus ? .8 : .1 + ((i * 3 + j * 5) % 5) / 22);
    }
    if (!labels) return;
    tag(g, x0 - 7 - size, y0 + h / 2 + 4, 'Q');
    tag(g, x0 - 4, y0 - 4 - size / 2 + 4, 'K');
    if (values) tag(g, x0 - 4, y0 + h + 4 + size / 2 + 3, 'V');
  }

  function grid(g, x, y, n, pitch, size, opacity) {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) cell(g, x + c * pitch, y + r * pitch, size, opacity(r, c));
  }
  /* Spatial resampling: a wide thin feature map against a small thick one. */
  function resample(g, cx, cy, direction) {
    const big = {n: 4, pitch: 7}, small = {n: 2, pitch: 7};
    const bw = big.n * big.pitch, sw = small.n * small.pitch;
    const bigX = direction === 'down' ? cx - 34 : cx + 34 - bw, smallX = direction === 'down' ? cx + 12 : cx - 12 - sw - 6;
    const by = cy - bw / 2, sy = cy - sw / 2 + 3;
    const [from, to] = direction === 'down'
      ? [[bigX + bw, by, bigX + bw, by + bw], [smallX, sy, smallX, sy + sw]]
      : [[smallX + sw + 6, sy - 6, smallX + sw + 6, sy + sw - 6], [bigX, by, bigX, by + bw]];
    el(g, 'line', {x1: from[0], y1: from[1], x2: to[0], y2: to[1], ...stroke(.25, .6)});
    el(g, 'line', {x1: from[2], y1: from[3], x2: to[2], y2: to[3], ...stroke(.25, .6)});
    grid(g, bigX, by, big.n, big.pitch, 6, (r, c) => .16 + ((r * 3 + c * 5) % 5) / 20);
    for (let layer = 2; layer >= 0; layer--)
      grid(g, smallX + layer * 3, sy - layer * 3, small.n, small.pitch, 6, () => .22 + (2 - layer) * .2);
  }

  /* Rows reduced to one summary cell; `back` broadcasts it to every row again. */
  function pooling(g, cx, cy, back) {
    const rows = [0, 1, 2, 3].map(i => cy - 13 + i * 7), left = back ? cx - 34 : cx - 20, mid = back ? cx - 3 : cx + 14;
    rows.forEach((y, i) => { cell(g, left, y - 3, 6, .25 + i * .12);
      el(g, 'line', {x1: left + 7, y1: y, x2: mid - 1, y2: cy, ...stroke(.3, .6)}); });
    cell(g, mid, cy - 3.5, 7, .85);
    if (!back) return;
    rows.forEach((y, i) => { el(g, 'line', {x1: mid + 8, y1: cy, x2: cx + 27, y2: y, ...stroke(.3, .6)});
      cell(g, cx + 28, y - 3, 6, .25 + i * .12); });
  }

  function gaussian(g, cx, baseline, width, height, opacity = .7) {
    const xs = Array.from({length: 25}, (_, i) => -3 + i * .25);
    el(g, 'path', {...stroke(opacity, 1.1), d: xs.map((v, i) =>
      `${i ? 'L' : 'M'}${(cx + v / 3 * width / 2).toFixed(1)} ${(baseline - Math.exp(-v * v / 2) * height).toFixed(1)}`).join(' ')});
    el(g, 'line', {x1: cx - width / 2, y1: baseline, x2: cx + width / 2, y2: baseline, ...stroke(.3, .6)});
  }

  const GLYPHS = {
    linear(g, b, cx, cy) {
      const fan = b.fan ?? [5, 3];
      dense(g, cx - (b.act ? 6 : 0), cy, fan, 36);
    },
    mlp(g, b, cx, cy) { dense(g, cx - (b.act ? 4 : 0), cy, b.fan ?? [3, 6, 3], 24, 6); },
    attention(g, b, cx, cy) { attention(g, cx, cy, {keys: 4}); },
    'cross-attention'(g, b, cx, cy) { attention(g, cx, cy, {keys: 6}); },
    transformer(g, b, cx, cy) {
      attention(g, cx - 22, cy + 2, {keys: 4, pitch: 5, size: 4.2, values: false, labels: false});
      dense(g, cx + 24, cy + 2, [2, 4, 2], 11, 5);
      for (const [x1, x2] of [[cx - 40, cx - 2], [cx + 8, cx + 42]])
        el(g, 'path', {...stroke(.45, .9), d: `M${x1} ${cy - 14} Q${(x1 + x2) / 2} ${cy - 30} ${x2} ${cy - 14}`});
    },
    conv(g, b, cx, cy) {
      const x0 = cx - 38, y0 = cy - 17.5, pitch = 7;
      grid(g, x0, y0, 5, pitch, 6, (r, c) => r >= 1 && r <= 3 && c >= 1 && c <= 3 ? .5 : .14);
      el(g, 'rect', {x: x0 + pitch - 1.5, y: y0 + pitch - 1.5, width: 3 * pitch + 1, height: 3 * pitch + 1, rx: 2, ...stroke(.85, 1)});
      const ox = cx + 14, oy = cy - 10.5;
      grid(g, ox, oy, 3, pitch, 6, (r, c) => r === 1 && c === 1 ? .8 : .14);
      el(g, 'line', {x1: x0 + 4 * pitch - .5, y1: y0 + pitch - 1.5, x2: ox + pitch, y2: oy + pitch, ...stroke(.4, .6)});
      el(g, 'line', {x1: x0 + 4 * pitch - .5, y1: y0 + 4 * pitch - .5, x2: ox + pitch, y2: oy + pitch + 6, ...stroke(.4, .6)});
    },
    resblock(g, b, cx, cy) {
      const pitch = 6;
      for (const dx of [-30, 6]) {
        grid(g, cx + dx, cy - 8, 4, pitch, 5, (r, c) => r >= 1 && r <= 2 && c >= 1 && c <= 2 ? .55 : .15);
        el(g, 'rect', {x: cx + dx + pitch - 1.5, y: cy - 8 + pitch - 1.5, width: 2 * pitch + 2, height: 2 * pitch + 2, rx: 1.5, ...stroke(.8, .9)});
      }
      el(g, 'path', {...stroke(.5, .9), d: `M${cx - 4} ${cy + 4} H${cx + 4} M${cx + 1} ${cy + 1} L${cx + 4} ${cy + 4} L${cx + 1} ${cy + 7}`});
      el(g, 'path', {...stroke(.5, .9), d: `M${cx - 30} ${cy - 13} Q${cx + 3} ${cy - 32} ${cx + 30} ${cy - 13}`});
      el(g, 'text', {x: cx + 30, y: cy - 14, 'text-anchor': 'start', class: 'nf-glyph-text'}, '+');
    },
    down(g, b, cx, cy) { resample(g, cx, cy, 'down'); },
    up(g, b, cx, cy) { resample(g, cx, cy, 'up'); },
    norm(g, b, cx, cy) {
      gaussian(g, cx, cy + 12, 60, 24);
      el(g, 'line', {x1: cx, y1: cy - 14, x2: cx, y2: cy + 12, ...stroke(.5, .8), 'stroke-dasharray': '2 2'});
      for (const dx of [-10, 10]) el(g, 'line', {x1: cx + dx, y1: cy + 10, x2: cx + dx, y2: cy + 15, ...stroke(.5, .8)});
    },
    patchify(g, b, cx, cy) {
      const x0 = cx - 38, y0 = cy - 14, levels = [.18, .34, .5, .66];
      levels.forEach((o, i) => el(g, 'rect', {x: x0 + (i % 2) * 14, y: y0 + Math.floor(i / 2) * 14, width: 13, height: 13, rx: 1, ...fill(o)}));
      el(g, 'path', {...stroke(.5, .9), d: `M${cx - 4} ${cy} H${cx + 8} M${cx + 4} ${cy - 3} L${cx + 8} ${cy} L${cx + 4} ${cy + 3}`});
      levels.forEach((o, i) => cell(g, cx + 18, cy - 13 + i * 7, 6, o));
    },
    sincos(g, b, cx, cy) {
      for (const [dy, period, o] of [[-8, 30, .75], [0, 15, .5], [8, 7.5, .3]]) {
        const xs = Array.from({length: 61}, (_, i) => i);
        el(g, 'path', {...stroke(o, 1), d: xs.map(i =>
          `${i ? 'L' : 'M'}${(cx - 30 + i).toFixed(1)} ${(cy + dy - 3.2 * Math.sin(2 * Math.PI * i / period)).toFixed(1)}`).join(' ')});
      }
    },
    graph(g, b, cx, cy) {
      const ring = [[-27, -11], [-20, 15], [6, 19], [27, 8], [22, -16]].map(([dx, dy]) => ({x: cx + dx, y: cy + dy}));
      ring.forEach((p, i) => { const q = ring[(i + 1) % ring.length];
        el(g, 'line', {x1: p.x, y1: p.y, x2: q.x, y2: q.y, ...stroke(.22, .7)}); });
      ring.forEach((p, i) => el(g, 'line', {x1: cx, y1: cy, x2: p.x, y2: p.y, ...stroke(.25 + i * .12, .6 + i * .25)}));
      el(g, 'rect', {x: (cx + ring[4].x) / 2 - 3, y: (cy + ring[4].y) / 2 - 3, width: 6, height: 6, rx: 1, ...fill(.55)});
      ring.forEach(p => el(g, 'circle', {cx: p.x, cy: p.y, r: 3.6, ...fill(.5)}));
      el(g, 'circle', {cx, cy, r: 5, ...fill(.9)});
    },
    pool(g, b, cx, cy) { pooling(g, cx, cy, false); },
    global(g, b, cx, cy) { pooling(g, cx, cy, true); },
    sample(g, b, cx, cy) {
      gaussian(g, cx, cy + 12, 60, 24);
      el(g, 'line', {x1: cx, y1: cy + 9, x2: cx, y2: cy + 15, ...stroke(.6, .9)});
      const sx = cx + 9, sy = cy + 12 - Math.exp(-(.9 ** 2) / 2) * 24;
      el(g, 'line', {x1: sx, y1: sy, x2: sx, y2: cy + 12, ...stroke(.5, .8), 'stroke-dasharray': '2 2'});
      el(g, 'circle', {cx: sx, cy: sy, r: 3, ...fill(.9)});
    }
  };

  /* Offset edges of the cards behind the front frame; only the repetition
     count is meaningful, never the number of visible cards. */
  function stackCards(parent, b, cls) {
    const {x, y, w, h} = b, r = 8;
    /* A card offset by less than the corner radius shows no left side above the front frame. */
    for (const o of [8, 4]) el(parent, 'path', {class: cls, d:
      (o >= r ? `M${x + o} ${y} V${y - o + r} Q${x + o} ${y - o} ${x + o + r} ${y - o}` : `M${x + r} ${y - o}`) +
      ` H${x + w + o - r} Q${x + w + o} ${y - o} ${x + w + o} ${y - o + r} V${y + h - o - r} Q${x + w + o} ${y + h - o} ${x + w} ${y + h - o}`});
  }
  /* Imported PNG/SVG icons arrive as data URIs. A brand mark keeps its own
     colors; a tinted icon is drawn as an alpha mask filled with currentColor,
     so it follows the theme and the processing tint like a built-in glyph. */
  let serial = 0;
  function icon(parent, b, box) {
    const entry = (window.NeuralFlowIcons ?? {})[b.icon];
    const attrs = {href: entry?.href ?? b.icon, ...box, preserveAspectRatio: 'xMidYMid meet'};
    if (!(b.iconTint ?? entry?.tint ?? false)) return el(parent, 'image', attrs);
    const id = `nf-icon-${++serial}`;
    el(el(parent, 'mask', {id, maskUnits: 'userSpaceOnUse', ...box, 'mask-type': 'alpha', style: 'mask-type: alpha'}), 'image', attrs);
    return el(parent, 'rect', {...box, fill: 'currentColor', 'fill-opacity': .8, mask: `url(#${id})`});
  }

  function lock(g, x, y) {
    el(g, 'path', {...stroke(.7, 1), d: `M${x + 2} ${y + 5} V${y + 3} a3 3 0 0 1 6 0 V${y + 5}`});
    el(g, 'rect', {x, y: y + 5, width: 10, height: 7, rx: 1.5, ...fill(.7)});
  }

  function draw(parent, b, frameClass) {
    if (b.layers > 1) stackCards(parent, b, frameClass);
    const frame = el(parent, 'rect', {x: b.x, y: b.y, width: b.w, height: b.h, rx: 8, class: frameClass});
    const glyph = el(parent, 'g', {class: 'nf-glyph'});
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2, iconMode = b.icon ? b.iconMode ?? 'glyph' : null;
    if (iconMode === 'glyph') {
      const size = Math.min(b.w - 40, b.h - 20);
      icon(glyph, b, {x: cx - size / 2, y: cy - size / 2, width: size, height: size});
    } else {
      const render = GLYPHS[b.glyph];
      if (!render) throw new Error(`Unknown block glyph: ${b.glyph}`);
      const badge = iconMode === 'badge', gx = cx + (badge ? 8 : 0), gy = cy + (badge ? 4 : 0);
      const scale = Math.max(.7, Math.min(1.4, b.w / 104, b.h / 58)) * (badge ? .82 : 1);
      render(el(glyph, 'g', {transform: `translate(${gx} ${gy}) scale(${scale.toFixed(3)}) translate(${-gx} ${-gy})`}), b, gx, gy);
    }
    if (iconMode === 'badge') icon(glyph, b, {x: b.x + 8, y: b.y + 7, width: 16, height: 16});
    if (b.act) activation(glyph, b.x + b.w - 26, b.y + 9, b.act);
    const badges = el(parent, 'g'), corner = iconMode === 'badge' ? 21 : 0;
    if (b.frozen) lock(badges, b.x + 9 + corner, b.y + 8);
    if (b.shared && b.labelAt !== 'right') {
      const width = b.shared.length * 7 + 12, inside = b.labelAt === 'above';
      const x = inside ? b.x + 7 + corner + (b.frozen ? 17 : 0) : b.x, y = inside ? b.y + 7 : b.y - 21 - (b.layers > 1 ? 8 : 0);
      el(badges, 'rect', {x, y, width, height: 16, rx: 4, ...fill(.1)});
      tag(badges, x + width / 2, y + 12, b.shared, 'middle');
    }
    return {frame, glyph};
  }

  window.NeuralBlocks = {draw, glyphs: Object.keys(GLYPHS), activations: Object.keys(ACTIVATIONS)};
})();
