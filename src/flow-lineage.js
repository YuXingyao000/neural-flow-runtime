/* Lineage-colored dataflow. Tensors stay put and show their current state; the
   breathing color on each line is the tensor in transit and reads downstream. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const ROOTS = ['#378ADD', '#639922', '#BA7517'];
  const FRESH = ['#7F77DD', '#D85A30', '#D4537E'];
  const GRAY = '#888780';
  /* Family shades alternate a dark and a light lightness per page theme:
     consecutive shades differ by at least 0.2, the light page's lightest shade
     keeps about 2:1 against white, the dark page's darkest stays visible. */
  const LEVELS = {light: {dark: [.38, .47], light: [.76, .68]}, dark: {dark: [.55, .6], light: [.9, .8]}};

  /* Any CSS color (#36c, white, rgb(), oklch()) to [r, g, b, alpha] through a 1x1 canvas. */
  let ink = null;
  function cssToRgb(color) {
    ink ??= document.createElement('canvas').getContext('2d', {willReadFrequently: true});
    ink.clearRect(0, 0, 1, 1);
    ink.fillStyle = '#000';
    ink.fillStyle = color;
    ink.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = ink.getImageData(0, 0, 1, 1).data;
    return [r, g, b, a / 255];
  }
  const toHex = color => /^#[0-9a-f]{6}$/i.test(color) ? color
    : '#' + cssToRgb(color).slice(0, 3).map(v => v.toString(16).padStart(2, '0')).join('');

  /* Page theme from the nearest opaque background, the host's background
     variables, or the colour-scheme preference. */
  function pageTheme(start = document.body) {
    const dark = ([r, g, b]) => r * .299 + g * .587 + b * .114 < 128;
    for (let n = start; n; n = n.parentElement) {
      const v = cssToRgb(getComputedStyle(n).backgroundColor);
      if (v[3] > .5) return dark(v) ? 'dark' : 'light';
    }
    const probe = document.createElement('span');
    probe.style.color = 'var(--page-bg, var(--background, var(--surface-0, transparent)))';
    document.body.appendChild(probe);
    const v = cssToRgb(getComputedStyle(probe).color);
    probe.remove();
    if (v[3] > .5) return dark(v) ? 'dark' : 'light';
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  const IDENTITY = new Set(['copy', 'reshape', 'view', 'gather', 'select', 'permute']);
  const GAP = 4, WIDTH = 680;
  let CELL = 12;
  /* One breathing period for solid lines and concat bands, so loops are seamless. */
  const FRONT_SPEED = 120, WAVE_SPEED = 30, WAVE = 46, BAND = 46, STEP = 3;
  const DELAY = {block: 84, junction: 14, tensor: 0, point: 0};
  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

  const toLinear = c => { c /= 255; return c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4; };
  const toByte = c => Math.round(clamp(c <= .0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - .055) * 255);
  function hexToLab(hex) {
    const n = parseInt(toHex(hex).slice(1), 16);
    const r = toLinear(n >> 16), g = toLinear(n >> 8 & 255), b = toLinear(n & 255);
    const l = Math.cbrt(.4122214708 * r + .5363325363 * g + .0514459929 * b);
    const m = Math.cbrt(.2119034982 * r + .6806995451 * g + .1073969566 * b);
    const s = Math.cbrt(.0883024619 * r + .2817188376 * g + .6299787005 * b);
    return [.2104542553 * l + .793617785 * m - .0040720468 * s,
      1.9779984951 * l - 2.428592205 * m + .4505937099 * s,
      .0259040371 * l + .7827717662 * m - .808675766 * s];
  }
  function labToHex([L, a, b]) {
    const l = (L + .3963377774 * a + .2158037573 * b) ** 3;
    const m = (L - .1055613458 * a - .0638541728 * b) ** 3;
    const s = (L - .0894841775 * a - 1.291485548 * b) ** 3;
    return '#' + [4.0767416621 * l - 3.3077115913 * m + .2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s,
      -.0041960863 * l - .7034186147 * m + 1.707614701 * s]
      .map(v => toByte(v).toString(16).padStart(2, '0')).join('');
  }

  /* Color lineage. Roots get palette families; copies keep the same style;
     concat carries bands without mixing; a learned transform of one lineage
     shifts lightness inside its family; the first merge of root lineages mixes
     them, and any later merge opens a fresh family. Residual add follows the
     stream; an all-zero branch leaves the stream unchanged.
     Semantic overrides for models whose tensors share one source: `color`
     starts a new family at a derived tensor, `joins` makes it a shade of another
     tensor's family, `structure` marks indices/masks (gray, never mixed), and
     `conditions` lists inputs that modulate without changing the family, e.g.
     a time embedding read by every block. */
  function resolveLineage(tensors, theme = pageTheme()) {
    const styles = {}, families = [];
    let roots = 0, fresh = 0;
    const family = (base, mixed, sources) => {
      const f = {id: families.length, base: toHex(base), mixed, sources, shades: 0};
      families.push(f);
      return f;
    };
    /* Shades keep the hue: a turned hue made purple read as blue and amber as
       red, i.e. a different family. The first shade goes to whichever level is
       farther from the base, so it never sits next to its root. */
    const shade = f => {
      const [L0, a0, b0] = hexToLab(f.base), k = f.shades++, lv = LEVELS[theme];
      const firstDark = Math.abs(lv.dark[0] - L0) >= Math.abs(lv.light[0] - L0);
      const dark = (k % 2 === 0) === firstDark, levels = lv[dark ? 'dark' : 'light'], gain = dark ? 1.15 : .85;
      return {kind: 'solid', hex: labToHex([levels[Math.floor(k / 2) % levels.length], a0 * gain, b0 * gain]), family: f};
    };
    const familiesOf = s => s.kind === 'bands' ? s.parts.map(p => p.family) : s.kind === 'solid' ? [s.family] : [];
    const extent = (id, axis) => styles[id].grid[axis === 0 ? 0 : 1];
    const fresher = sources => {
      const f = family(FRESH[fresh++ % FRESH.length], true, sources);
      return {kind: 'solid', hex: f.base, family: f};
    };
    function merge(inputs) {
      const weights = new Map();
      for (const s of inputs.filter(s => !s.structure)) {
        if (s.kind === 'bands') s.parts.forEach(p => weights.set(p.family, (weights.get(p.family) ?? 0) + p.weight));
        else if (s.kind === 'solid') weights.set(s.family, (weights.get(s.family) ?? 0) + 1);
      }
      const members = [...weights.keys()];
      if (!members.length) {
        const f = family(GRAY, false, []);
        return {kind: 'solid', hex: f.base, family: f};
      }
      if (members.length === 1) return shade(members[0]);
      const sources = members.flatMap(f => f.sources);
      if (members.some(f => f.mixed)) return fresher(sources);
      const total = [...weights.values()].reduce((a, b) => a + b, 0);
      const lab = [0, 0, 0];
      weights.forEach((w, f) => hexToLab(f.base).forEach((v, i) => { lab[i] += v * w / total; }));
      /* Complementary roots (blue + amber) average to gray, which belongs to indices. */
      if (Math.hypot(lab[1], lab[2]) < .05) return fresher(sources);
      const f = family(labToHex(lab), true, sources);
      return {kind: 'solid', hex: f.base, family: f};
    }
    function resolve(id) {
      if (styles[id]) return styles[id];
      const t = tensors[id];
      if (!t) throw new Error(`Unknown tensor: ${id}`);
      const inputs = (t.in ?? []).map(resolve);
      const content = (t.in ?? []).map((k, i) => (t.conditions ?? []).includes(k) ? null : inputs[i]).filter(Boolean);
      let style;
      const passes = IDENTITY.has(t.op) || t.op === 'broadcast';
      if (!t.op || t.color || (t.structure && !passes)) {
        const base = t.color ?? (t.structure ? GRAY : ROOTS[roots++ % ROOTS.length]);
        const f = family(base, false, [id]);
        style = {kind: 'solid', hex: f.base, family: f};
      } else if (t.joins) {
        style = shade(familiesOf(resolve(t.joins))[0]);
      } else if (passes) {
        style = {...inputs[0], sameAs: t.in[0]};
      } else if (t.op === 'concat') {
        const parts = [];
        t.in.forEach((k, i) => {
          const w = extent(k, t.axis), s = inputs[i];
          if (s.kind === 'bands') s.parts.forEach(p => parts.push({...p, weight: p.weight * w}));
          else if (s.kind === 'solid') parts.push({hex: s.hex, family: s.family, weight: w});
        });
        const sum = parts.reduce((a, p) => a + p.weight, 0);
        style = {kind: 'bands', parts: parts.map(p => ({...p, weight: p.weight / sum}))};
      } else if (t.op === 'gate') {
        style = t.zero ? {kind: 'zero'} : merge(content.slice(0, 1));
      } else if (t.op === 'add' && t.stream) {
        const others = t.in.filter(k => k !== t.stream).map(resolve);
        const stream = resolve(t.stream);
        style = others.every(s => s.kind === 'zero') ? {...stream, sameAs: t.stream} : shade(familiesOf(stream)[0]);
      } else {
        style = merge(content.length ? content : inputs);
      }
      style = {...style, structure: Boolean(t.structure || (passes && style.structure))};
      const grid = t.grid ?? (t.op === 'concat'
        ? (t.axis === 0
          ? [t.in.reduce((a, k) => a + styles[k].grid[0], 0), styles[t.in[0]].grid[1]]
          : [styles[t.in[0]].grid[0], t.in.reduce((a, k) => a + styles[k].grid[1], 0)])
        : inputs[0]?.grid);
      styles[id] = {...style, grid, id};
      return styles[id];
    }
    Object.keys(tensors).forEach(resolve);
    return styles;
  }

  function el(parent, tag, attrs = {}, text) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = text;
    parent.appendChild(n);
    return n;
  }
  function hash(id) { let h = 7; for (const c of id) h = (h * 31 + c.codePointAt(0)) >>> 0; return h; }
  const texture = (id, r, c) => .34 + ((hash(id) + r * 17 + c * 29) % 11) / 16;

  /* Tensor glyph: concat and copies reuse their inputs' cells, so a source's
     texture survives every operation that does not change its values. */
  /* An identity of a concat (copy / view) is drawn as that concat. */
  const visual = (tensors, styles, id) => {
    while (styles[id].sameAs && tensors[id].op !== 'broadcast' && tensors[styles[id].sameAs]?.op === 'concat') id = styles[id].sameAs;
    return id;
  };
  const cellSize = t => t.cell ?? CELL;
  function glyphSize(tensors, styles, id) {
    id = visual(tensors, styles, id);
    const t = tensors[id];
    if (t.op === 'concat') {
      const sizes = t.in.map(k => glyphSize(tensors, styles, k));
      return t.axis === 0
        ? {w: Math.max(...sizes.map(s => s.w)), h: sizes.reduce((a, s) => a + s.h, 0) + GAP * (sizes.length - 1)}
        : {w: sizes.reduce((a, s) => a + s.w, 0) + GAP * (sizes.length - 1), h: Math.max(...sizes.map(s => s.h))};
    }
    const [rows, cols] = styles[id].grid, size = cellSize(t);
    return {w: cols * size - 2, h: rows * size - 2};
  }
  /* Cell colors: textures follow identity through copies, broadcasts repeat the
     source row, concat cells come from the matching source, and an all-zero
     tensor is drawn as empty outlines. */
  function cellOf(tensors, styles, id) {
    const t = tensors[id], s = styles[id];
    if (s.kind === 'zero') return () => ({fill: 'currentColor', opacity: .08});
    if (t.op === 'broadcast') { const source = cellOf(tensors, styles, t.in[0]); return (r, c) => source(0, c); }
    if (s.sameAs) return cellOf(tensors, styles, s.sameAs);
    if (t.op === 'concat') {
      const along = t.axis === 0 ? 0 : 1;
      const spans = t.in.map(k => styles[k].grid[along]);
      return (r, c) => {
        let index = along ? c : r, k = 0;
        while (k < spans.length - 1 && index >= spans[k]) index -= spans[k++];
        return cellOf(tensors, styles, t.in[k])(along ? r : index, along ? index : c);
      };
    }
    return (r, c) => ({fill: s.hex, opacity: texture(id, r, c)});
  }
  function drawGlyph(parent, tensors, styles, id, x, y) {
    id = visual(tensors, styles, id);
    const t = tensors[id];
    if (t.op === 'concat') {
      let offset = 0;
      for (const k of t.in) {
        const size = glyphSize(tensors, styles, k);
        drawGlyph(parent, tensors, styles, k, t.axis === 0 ? x : x + offset, t.axis === 0 ? y + offset : y);
        offset += (t.axis === 0 ? size.h : size.w) + GAP;
      }
      return;
    }
    const [rows, cols] = styles[id].grid, cell = cellOf(tensors, styles, id), size = cellSize(t);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const {fill, opacity} = cell(r, c);
      el(parent, 'rect', {x: x + c * size, y: y + r * size, width: size - 2, height: size - 2, rx: 2,
        fill, 'fill-opacity': opacity.toFixed(2)});
    }
  }

  const SANS = 'Arial, "Helvetica Neue", "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif';
  const MONO = 'Consolas, Menlo, "Courier New", "Microsoft YaHei", monospace';
  const PAINT = {fill: 'rgb(0, 0, 0)', 'fill-opacity': '1', stroke: 'none', 'stroke-opacity': '1', 'stroke-width': '1px',
    'stroke-dasharray': 'none', 'stroke-linecap': 'butt', 'stroke-linejoin': 'miter', opacity: '1', 'mask-type': 'luminance',
    'paint-order': 'normal'};
  const TYPE = ['font-size', 'font-weight', 'text-anchor'];
  /* Copy a subtree with computed presentation values written as attributes,
     so the SVG renders the same without the page's CSS variables. */
  function inlineCopy(node) {
    const copy = node.cloneNode(true);
    const originals = [node, ...node.querySelectorAll('*')], copies = [copy, ...copy.querySelectorAll('*')];
    originals.forEach((n, i) => {
      const c = copies[i], computed = getComputedStyle(n);
      for (const [p, initial] of Object.entries(PAINT)) {
        const v = computed.getPropertyValue(p);
        if (v && v !== initial) c.setAttribute(p, v.replace(/px$/, ''));
      }
      if (n.tagName === 'text') {
        TYPE.forEach(p => c.setAttribute(p, computed.getPropertyValue(p).replace(/px$/, '')));
        c.setAttribute('font-family', n.matches('.nf-mono, .nf-glyph-text') ? MONO : SANS);
      }
      c.removeAttribute('class');
      c.removeAttribute('style');
    });
    return copy;
  }

  /* Words the figure itself writes; a scene's `lang` (or the shell's
     NeuralFlowLang) picks the set, labels in the scene stay as written. */
  const STRINGS = {
    zh: {output: '输出', lossNote: '参与训练 loss（模型输出），数字是加入总 loss 的乘数', outputNote: '输出的去向', terms: 'loss：', sep: '；'},
    en: {output: 'output', lossNote: 'in the training loss (model output); the number is its weight', outputNote: 'where outputs go', terms: 'loss: ', sep: '; '}
  };

  /* Tags after a tensor's name: what it is exported as, and its weight in the
     training loss (the loss terms go into a hover title). */
  const tagsOf = (n, words) => [
    ...(n.loss ? [{text: `ℒ ${n.loss.weight ?? n.loss}`, cls: 'nf-loss-tag', textCls: 'nf-loss-text'}] : []),
    ...(n.output ? [{text: n.output === true ? words.output : n.output, cls: 'nf-output-tag'}] : [])
  ];

  function anchor(node, size, side, offset = 0) {
    if (node.kind === 'point') return {x: node.x, y: node.y + offset};
    if (node.kind === 'junction') {
      const r = 13;
      return {left: {x: node.x - r, y: node.y + offset}, right: {x: node.x + r, y: node.y + offset},
        top: {x: node.x + offset, y: node.y - r}, bottom: {x: node.x + offset, y: node.y + r}}[side];
    }
    const {x, y} = node, w = size.w, h = size.h, cards = node.layers > 1 ? 8 : 0;
    return {left: {x: x - 4, y: y + h / 2 + offset}, right: {x: x + w + 4 + cards, y: y + h / 2 + offset},
      top: {x: x + w / 2 + offset, y: y - 4 - cards}, bottom: {x: x + w / 2 + offset, y: y + h + 4}}[side];
  }
  function pathOf(a, b, via) {
    if (via?.length) return `M${a.x} ${a.y} ` + [...via.map(([x, y]) => ({x, y})), b].map(p => `L${p.x} ${p.y}`).join(' ');
    const horizontal = Math.abs(b.x - a.x) >= Math.abs(b.y - a.y);
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    return horizontal ? `M${a.x} ${a.y} C${mx} ${a.y} ${mx} ${b.y} ${b.x} ${b.y}`
      : `M${a.x} ${a.y} C${a.x} ${my} ${b.x} ${my} ${b.x} ${b.y}`;
  }

  /* Front schedule: a node is reached when its slowest data input arrives;
     every input's front is rescaled to arrive at that moment, and skip edges
     never hold the main path back. */
  function schedule(nodes, edges) {
    const arrival = {}, ready = {};
    for (const [id, n] of Object.entries(nodes)) if (!n.after && !edges.some(e => e.to === id)) { arrival[id] = 0; ready[id] = 0; }
    for (let pass = 0; pass < 64 && Object.keys(ready).length < Object.keys(nodes).length; pass++) {
      for (const [id, n] of Object.entries(nodes)) {
        if (id in ready) continue;
        if (n.after) {
          if (n.after in ready) arrival[id] = ready[id] = ready[n.after];
          continue;
        }
        const incoming = edges.filter(e => e.to === id);
        if (!incoming.every(e => e.from in ready)) continue;
        const data = incoming.filter(e => e.role !== 'skip');
        const timed = data.length ? data : incoming;
        arrival[id] = Math.max(...timed.map(e => ready[e.from] + e.length));
        ready[id] = arrival[id] + (DELAY[n.kind] ?? 0);
      }
    }
    for (const e of edges) {
      e.start = ready[e.from];
      e.scale = Math.max(.2, (arrival[e.to] - e.start) / e.length);
    }
    return {arrival, ready};
  }

  function mount(svg, original) {
    const tensors = original.tensors, styles = resolveLineage(tensors, pageTheme(svg.parentElement));
    CELL = original.cell ?? 12;
    const words = STRINGS[original.lang ?? window.NeuralFlowLang] ?? STRINGS.zh;
    for (const [id, n] of Object.entries(original.nodes))
      if (!['tensor', 'block', 'junction', 'point'].includes(n.kind)) throw new Error(`Node ${id}: kind must be tensor, block, junction or point`);
    /* Text is measured in user units; a viewBox left from the previous mount
       would scale the measurements and shift the layout on every re-render. */
    svg.replaceChildren();
    svg.removeAttribute('viewBox');
    const layers = Object.fromEntries(['modules', 'tracks', 'flow', 'heads', 'nodes', 'labels'].map(k => [k, el(svg, 'g')]));
    const measure = (text, cls) => { const t = el(svg, 'text', {class: cls}, text); const w = t.getComputedTextLength(); t.remove(); return w; };
    /* The tier layout mutates its scene, so every attempt works on a copy. If
       the figure is wider than its container, lay it out once more compactly
       (narrower cards, tags under the tensor name) before letting it scale. */
    const available = Math.max(560, Math.round(svg.parentElement?.clientWidth || WIDTH));
    let scene, sizes, compact = false;
    const attempt = tight => {
      compact = tight;
      scene = structuredClone(original);
      if (scene.layout === 'tiers') for (const n of Object.values(scene.nodes)) {
        if (n.kind === 'block') { n.w ??= tight ? 88 : 104; n.h ??= 56; }
        if (n.kind !== 'junction') n.labelAt ??= 'right';
      }
      sizes = {};
      for (const [id, n] of Object.entries(scene.nodes))
        sizes[id] = n.kind === 'tensor' ? glyphSize(tensors, styles, n.tensor) : {w: n.w ?? 0, h: n.h ?? 0};
      if (scene.layout === 'tiers') NeuralLayout.tiers(scene, id => tierBox(id), {tight});
    };
    attempt(false);
    if (scene.layout === 'tiers' && original.fit !== false && scene.width > available) attempt(true);
    const marked = Object.values(scene.nodes).filter(n => n.kind === 'tensor');
    const width = scene.width ?? WIDTH;
    const legendItems = scene.layout === 'tiers' ? legendOf() : [];
    const legendRows = [];
    for (const item of legendItems) {
      const row = legendRows.at(-1);
      if (row && row.width + item.width <= width - 32) { row.items.push(item); row.width += item.width; }
      else legendRows.push({items: [item], width: item.width});
    }
    if (legendRows.length) scene.height += 10 + 22 * legendRows.length;
    svg.setAttribute('viewBox', `0 0 ${width} ${scene.height}`);
    /* Box of a node in the tier layout: its glyph or card plus the name to its
       right, the point where it is placed, and where edges meet it. */
    function tierBox(id) {
      const n = scene.nodes[id], s = sizes[id];
      if (n.kind === 'junction') return {w: 26, h: 26, ax: 13, place: (x, y) => ({x: x + 13, y: y + 13}),
        ports: (x, y) => ({x: x + 13, top: y - 1, bottom: y + 27, span: 0})};
      if (n.kind === 'point') {
        const text = n.label ? measure(n.label, 'nf-label nf-mono') + 7 : 0;
        return {w: 6 + text, h: 14, ax: 3, place: (x, y) => ({x: x + 3, y: y + 7}),
          ports: (x, y) => ({x: x + 3, top: y + 4, bottom: y + 10, span: 0})};
      }
      if (n.kind === 'block') {
        const cards = n.layers > 1 ? 8 : 0, lift = cards ? 24 : 0;
        const chip = n.shared ? measure(n.shared, 'nf-glyph-text') + 18 : 0;
        const text = Math.max(chip + measure(n.label ?? '', 'nf-label'), n.detail ? measure(n.detail, 'nf-label nf-mono') : 0);
        const h = Math.max(s.h + lift, n.detail ? 32 : 16);
        return {w: s.w + cards + 10 + text, h, ax: s.w / 2, place: (x, y) => ({x, y: y + h - s.h}),
          ports: (x, y) => ({x: x + s.w / 2, top: y + h - s.h - 4 - cards, bottom: y + h + 4, span: s.w * .6})};
      }
      const tags = tagsOf(n, words), pad = n.loss ? 4 : 0;
      const tagWidth = tags.reduce((w, t) => w + measure(t.text, 'nf-label nf-output-text') + 18, 0);
      const name = n.label ? measure(n.label, 'nf-label nf-mono') : 0;
      const text = n.label ? (compact && tags.length ? Math.max(name, tagWidth) : name + tagWidth) : 0;
      const h = Math.max(s.h, compact && tags.length ? 36 : 16) + 2 * pad;
      return {w: s.w + 2 * pad + (text ? 8 + text : 0), h, ax: pad + s.w / 2,
        place: (x, y) => ({x: x + pad, y: y + (h - s.h) / 2}),
        ports: (x, y) => ({x: x + pad + s.w / 2, top: y + (h - s.h) / 2 - 4, bottom: y + (h + s.h) / 2 + 4, span: Math.max(0, s.w - 6)})};
    }
    const tint = id => { const s = styles[id]; return s?.kind === 'solid' ? s.hex : s?.kind === 'bands' ? s.parts[0].hex : ''; };
    const label = (x, y, text, {anchor: a = 'middle', mono = false, secondary = false} = {}) =>
      el(layers.labels, 'text', {x, y, 'text-anchor': a, class: `nf-label${mono ? ' nf-mono' : ''}${secondary ? ' nf-secondary' : ''}`}, text);

    /* A tag sized from its rendered text; returns its width. */
    function outputTag(x, top, anchor, text, cls = 'nf-output-tag', textCls = '') {
      const box = el(layers.labels, 'rect', {y: top, height: 16, rx: 8, class: cls});
      const t = el(layers.labels, 'text', {y: top + 12, 'text-anchor': 'start', class: `nf-label nf-output-text ${textCls}`}, text);
      const width = t.getComputedTextLength() + 12, left = anchor === 'end' ? x - width : anchor === 'middle' ? x - width / 2 : x;
      box.setAttribute('x', left); box.setAttribute('width', width); t.setAttribute('x', left + 6);
      return width;
    }

    /* Legend inside the figure, so exported slides carry it: the loss tag, the
       output tags present, and the scene's semantic color families. */
    function legendOf() {
      const items = [], text = t => measure(t, 'nf-label');
      if (marked.some(n => n.loss)) {
        const note = words.lossNote;
        items.push({kind: 'loss', note, width: measure('ℒ', 'nf-label nf-output-text') + 18 + text(note) + 18});
      }
      const kinds = [...new Set(marked.filter(n => n.output).map(n => n.output === true ? words.output : n.output))];
      if (kinds.length) {
        const note = words.outputNote;
        items.push({kind: 'output', kinds, note,
          width: kinds.reduce((w, k) => w + measure(k, 'nf-label nf-output-text') + 16, 0) + text(note) + 18});
      }
      for (const f of scene.families ?? []) items.push({kind: 'family', ...f, width: 18 + text(f.label) + 16});
      return items;
    }
    legendRows.forEach((row, r) => {
      let x = 16;
      const y = scene.height - 22 * (legendRows.length - r) - 4;
      for (const item of row.items) {
        if (item.kind === 'loss') x += outputTag(x, y, 'start', 'ℒ', 'nf-loss-tag', 'nf-loss-text') + 6;
        if (item.kind === 'output') for (const k of item.kinds) x += outputTag(x, y, 'start', k, 'nf-output-tag') + 4;
        if (item.kind === 'family') {
          el(layers.labels, 'rect', {x, y: y + 3, width: 10, height: 10, rx: 2, fill: toHex(item.color)});
          x += 16;
        }
        x += label(x + 2, y + 12, item.note ?? item.label, {anchor: 'start', secondary: true}).getComputedTextLength() + 18;
      }
    });

    const nodeViews = {};
    for (const [id, n] of Object.entries(scene.nodes)) {
      if (n.kind === 'point') {
        if (!n.label) continue;
        el(layers.nodes, 'circle', {cx: n.x, cy: n.y, r: 2.5, fill: 'currentColor', 'fill-opacity': .5});
        const right = (n.labelAt ?? 'left') === 'right';
        label(n.x + (right ? 7 : -7), n.y + 4, n.label, {anchor: right ? 'start' : 'end', mono: true, secondary: true});
        continue;
      }
      const g = el(layers.nodes, 'g', {'data-node': id});
      if (n.kind === 'tensor') {
        drawGlyph(g, tensors, styles, n.tensor, n.x, n.y);
        const s = sizes[id], cx = n.x + s.w / 2;
        const tags = tagsOf(n, words);
        if (n.loss) el(g, 'rect', {x: n.x - 4, y: n.y - 4, width: s.w + 8, height: s.h + 8, rx: 5, class: 'nf-output'});
        if (n.loss?.terms) el(g, 'title', {}, `${n.label ?? n.tensor} · ${words.terms}${n.loss.terms.join(words.sep)}`);
        if (n.label) {
          const where = n.labelAt ?? 'above';
          const [x, y, anchor] = where === 'right' ? [n.x + s.w + 8, n.y + s.h / 2 + 4, 'start']
            : where === 'left' ? [n.x - 8, n.y + s.h / 2 + 4, 'end'] : [cx, where === 'below' ? n.y + s.h + 16 : n.y - 8, 'middle'];
          const under = compact && where === 'right' && tags.length;
          const text = label(x, under ? y - 9 : y, n.label, {anchor, mono: true});
          if (where === 'right') {
            let cursor = under ? x : x + text.getComputedTextLength() + 6;
            for (const t of tags) cursor += outputTag(cursor, under ? y - 3 : y - 12, 'start', t.text, t.cls, t.textCls) + 6;
          } else if (tags.length) outputTag(x, where === 'below' ? y + 6 : y - 31, anchor, tags.map(t => t.text).join(' · '), tags[0].cls, tags[0].textCls);
        }
        if (n.caption) label(cx, n.y + s.h + 16, n.caption, {mono: true, secondary: true});
      } else if (n.kind === 'block') {
        nodeViews[id] = {...NeuralBlocks.draw(g, n, 'nf-frame'), tint: tint(n.out)};
        nodeViews[id].glyph.setAttribute('data-glyph', id);
        const lift = n.layers > 1 ? 6 : 0;
        if (n.labelAt === 'above') {
          label(n.x, n.y - 24 - lift, n.label, {anchor: 'start'});
          if (n.detail) label(n.x, n.y - 9 - lift, n.detail, {anchor: 'start', mono: true, secondary: true});
        } else if (n.labelAt === 'right') {
          const x = n.x + n.w + (n.layers > 1 ? 8 : 0) + 10, cy = n.y + n.h / 2, base = n.detail ? cy - 3 : cy + 4;
          let shift = 0;
          if (n.shared) {
            const chip = el(layers.labels, 'rect', {x, y: base - 12, height: 16, rx: 4, class: 'nf-output-tag'});
            const t = el(layers.labels, 'text', {x: x + 6, y: base, class: 'nf-glyph-text'}, n.shared);
            shift = t.getComputedTextLength() + 12;
            chip.setAttribute('width', shift);
            shift += 6;
          }
          label(x + shift, base, n.label, {anchor: 'start'});
          if (n.detail) label(x, cy + 13, n.detail, {anchor: 'start', mono: true, secondary: true});
        } else {
          label(n.x + n.w / 2, n.y + n.h + 16, n.label);
          if (n.detail) label(n.x + n.w / 2, n.y + n.h + 31, n.detail, {mono: true, secondary: true});
        }
        if (n.layers > 1) label(n.x + n.w + 8, n.labelAt === 'above' ? n.y - 24 - lift : n.y - 14, `×${n.layers}`, {anchor: 'end', mono: true});
      } else {
        nodeViews[id] = {frame: el(g, 'circle', {cx: n.x, cy: n.y, r: 13, class: 'nf-frame'})};
        label(n.x, n.y + 4, n.symbol);
      }
      nodeViews[id] = {...nodeViews[id], g};
    }

    const edges = scene.edges.map((spec, index) => {
      const from = scene.nodes[spec.from], to = scene.nodes[spec.to];
      const tensor = spec.tensor ?? (from.kind === 'tensor' || from.kind === 'point' ? from.tensor : from.out);
      const d = spec.d ?? pathOf(anchor(from, sizes[spec.from], spec.fromSide ?? 'right', spec.fromOffset),
        anchor(to, sizes[spec.to], spec.toSide ?? 'left', spec.toOffset), spec.via);
      const track = el(layers.tracks, 'path', {d, class: `nf-track${spec.role === 'control' ? ' nf-control' : ''}`});
      const length = track.getTotalLength();
      const end = track.getPointAtLength(length), before = track.getPointAtLength(Math.max(0, length - 2));
      const theta = Math.atan2(end.y - before.y, end.x - before.x);
      el(layers.heads, 'path', {class: 'nf-head', d: `M${end.x - 6 * Math.cos(theta - .5)} ${end.y - 6 * Math.sin(theta - .5)} L${end.x} ${end.y} L${end.x - 6 * Math.cos(theta + .5)} ${end.y - 6 * Math.sin(theta + .5)}`});
      const segments = [];
      for (let s = 0; s < length - STEP / 2; s += STEP) {
        if (spec.role === 'control' && Math.floor(s / 6) % 2) continue;
        const p = track.getPointAtLength(s), q = track.getPointAtLength(Math.min(length, s + STEP + .6));
        segments.push({s: s + STEP / 2, line: el(layers.flow, 'line', {x1: p.x, y1: p.y, x2: q.x, y2: q.y,
          'stroke-width': spec.role === 'control' ? 2 : 3})});
      }
      if (spec.label && !spec.labelPlaced) label(spec.label[0], spec.label[1], spec.label[2], {anchor: spec.label[3] ?? 'start', secondary: true});
      return {...spec, index, tensor, style: styles[tensor], d, length, segments, track};
    });
    (scene.labels ?? []).forEach(([x, y, text, a]) => label(x, y, text, {anchor: a ?? 'middle', secondary: true}));

    /* Module frames, module titles and automatic edge labels go where no edge
       and no earlier label passes: each tries a few candidate spots in turn. */
    const samples = edges.map(e => {
      const points = [];
      for (let s = 0; s <= e.length; s += 4) points.push(e.track.getPointAtLength(s));
      return points;
    });
    const taken = [...layers.labels.querySelectorAll('text')].map(t => t.getBBox());
    const clear = (box, skip) => box.x >= 2 && box.x + box.width <= width - 2 && !samples.some((points, i) => i !== skip && points.some(p =>
        p.x > box.x - 2 && p.x < box.x + box.width + 2 && p.y > box.y - 2 && p.y < box.y + box.height + 2))
      && !taken.some(b => box.x < b.x + b.width && b.x < box.x + box.width && box.y < b.y + b.height && b.y < box.y + box.height);
    /* The first clear candidate wins; when none is clear the label keeps the
       first spot with a halo in the page color so it still reads over lines. */
    const place = (candidates, text, options) => {
      for (const [x, y, anchor] of candidates) {
        const t = label(x, y, text, {...options, anchor});
        const box = t.getBBox();
        if (clear(box, -1)) { taken.push(box); return t; }
        t.remove();
      }
      const [x, y, anchor] = candidates[0], t = label(x, y, text, {...options, anchor});
      t.classList.add('nf-halo');
      taken.push(t.getBBox());
      return t;
    };
    for (const m of scene.modules ?? []) {
      if (m.repeat > 1) for (const o of [8, 4]) el(layers.modules, 'rect', {x: m.x + o, y: m.y - o, width: m.w, height: m.h, rx: 10, class: 'nf-module nf-module-card'});
      el(layers.modules, 'rect', {x: m.x, y: m.y, width: m.w, height: m.h, rx: 10, class: 'nf-module'});
      const title = m.repeat > 1 ? `${m.label} · ×${m.repeat}` : m.label, room = m.w - 24 - measure(title, 'nf-label');
      const slots = [];
      for (let dx = 0; dx <= Math.max(0, room); dx += 24) slots.push([m.x + 12 + dx, m.y + 17, 'start']);
      for (let dx = 0; dx <= Math.max(0, room); dx += 24) slots.push([m.x + 12 + dx, m.y + m.h - 8, 'start']);
      place(slots, title, {secondary: true});
    }
    edges.forEach(e => {
      if (!e.labelPlaced) return;
      const slots = [];
      for (const f of [.3, .5, .15, .7, .4, .6, .2, .8, .85]) {
        const p = e.track.getPointAtLength(e.length * f);
        for (const [dx, dy] of [[8, 4], [-8, 4], [20, 4], [-20, 4], [8, -8], [-8, -8], [8, 16], [-8, 16]])
          slots.push([p.x + dx, p.y + dy, dx > 0 ? 'start' : 'end']);
      }
      place(slots, e.label[2], {secondary: true});
    });

    const {arrival, ready} = schedule(scene.nodes, edges);
    const finish = Math.max(...Object.values(ready)) + 40;
    const frontSpeed = scene.frontSeconds ? finish / scene.frontSeconds : FRONT_SPEED;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');

    /* Segments overlap slightly to hide seams, so translucent strokes would bead.
       Composite against the page backdrop here and draw every segment opaque. */
    let backdrop = [255, 255, 255], backdropAt = -Infinity;
    function readBackdrop() {
      for (let n = svg.parentElement; n; n = n.parentElement) {
        const v = cssToRgb(getComputedStyle(n).backgroundColor);
        if (v[3] > .5) return v.slice(0, 3);
      }
      const probe = el(svg, 'rect', {width: 0, height: 0});
      probe.style.fill = 'var(--page-bg, var(--background, var(--surface-0, transparent)))';
      const host = cssToRgb(getComputedStyle(probe).fill);
      probe.remove();
      if (host[3] > .5) return host.slice(0, 3);
      const ink = cssToRgb(getComputedStyle(svg).color);
      return ink[0] + ink[1] + ink[2] > 380 ? [31, 31, 31] : [255, 255, 255];
    }
    const rgb = color => cssToRgb(color).slice(0, 3);
    const over = (c, a) => c.map((v, i) => Math.round(backdrop[i] + (v - backdrop[i]) * a));
    const css = c => `rgb(${c.join(',')})`, hex = c => '#' + c.map(v => v.toString(16).padStart(2, '0')).join('');
    const solid = new Map(), parts = new Map();
    for (const e of edges) {
      if (e.style.kind === 'solid') solid.set(e, rgb(e.style.hex));
      if (e.style.kind === 'bands') parts.set(e, e.style.parts.map(p => rgb(p.hex)));
    }

    function segmentColor(e, seg, time, reached) {
      const style = e.style, phase = e.start + seg.s - WAVE_SPEED * time;
      if (style.kind === 'bands') {
        let u = ((phase / BAND) % 1 + 1) % 1, index = 0;
        while (index < style.parts.length - 1 && u >= style.parts[index].weight) u -= style.parts[index++].weight;
        return over(parts.get(e)[index], (.26 + .69 * Math.sin(Math.PI * u / style.parts[index].weight) ** 2) * reached);
      }
      return over(solid.get(e), (.3 + .6 * (.5 + .5 * Math.cos(2 * Math.PI * phase / WAVE))) * reached);
    }

    function draw(time) {
      const now = performance.now();
      if (now - backdropAt > 500) { backdrop = readBackdrop(); backdropAt = now; }
      const front = frontSpeed * time;
      for (const e of edges) {
        for (const seg of e.segments) {
          const reached = clamp((front - (e.start + seg.s * e.scale)) / 16);
          if (!reached || e.style.kind === 'zero') { seg.line.setAttribute('stroke-opacity', 0); continue; }
          seg.line.setAttribute('stroke', css(segmentColor(e, seg, time, reached)));
          seg.line.setAttribute('stroke-opacity', 1);
        }
      }
      for (const [id, n] of Object.entries(scene.nodes)) {
        const view = nodeViews[id];
        if (!view) continue;
        if (n.kind === 'tensor') { view.g.setAttribute('opacity', front >= arrival[id] ? 1 : .2); continue; }
        const busy = front >= arrival[id] && front < ready[id];
        view.frame.classList.toggle('nf-busy', busy);
        if (view.glyph) view.glyph.style.color = busy && view.tint ? view.tint : '';
      }
    }

    /* Standalone SVG for slides and papers. Strokes are composited against the
       export background. What PowerPoint, PDF and LaTeX show is a clean frame:
       one path per edge, concat as alternating dashed source colors. Where SVG
       animation runs (browsers), SMIL swaps in the segmented flow, replays the
       front once and loops the breathing. */
    function exportSVG({animated = true, background = '#ffffff', frames = 12} = {}) {
      cancelAnimationFrame(frame); paused = true;
      const steady = finish / frontSpeed + 1, seconds = v => `${(+v).toFixed(3)}s`;
      if (background) { background = toHex(background); backdrop = rgb(background); backdropAt = Infinity; }
      /* No CSS transitions while reading computed styles: they would capture mid-fade values. */
      svg.closest('.neural-flow')?.classList.add('nf-still');
      draw(steady);
      const tinted = {};
      for (const [id, view] of Object.entries(nodeViews)) {
        if (!view.glyph || !view.tint) continue;
        view.glyph.style.transition = 'none';
        view.glyph.style.color = view.tint;
        tinted[id] = inlineCopy(view.glyph);
        view.glyph.style.color = '';
        getComputedStyle(view.glyph).color;
      }
      const copy = inlineCopy(svg);
      Object.values(nodeViews).forEach(view => { if (view.glyph) view.glyph.style.transition = ''; });
      svg.closest('.neural-flow')?.classList.remove('nf-still');
      copy.setAttribute('xmlns', NS);
      copy.setAttribute('width', width);
      copy.setAttribute('height', scene.height);
      ['role', 'aria-label'].forEach(a => copy.removeAttribute(a));
      const flow = copy.children[2], lines = [...flow.children];
      const still = el(svg, 'g');
      for (const e of edges) {
        const control = e.role === 'control', base = {d: e.d, fill: 'none', 'stroke-width': control ? 2 : 3};
        if (e.style.kind === 'solid') el(still, 'path', {...base, stroke: css(over(solid.get(e), .8)),
          'stroke-linecap': control ? 'butt' : 'round', ...(control ? {'stroke-dasharray': '6 6'} : {})});
        if (e.style.kind === 'bands') {
          let offset = 0;
          e.style.parts.forEach((part, i) => {
            const dash = part.weight * BAND;
            el(still, 'path', {...base, stroke: css(over(parts.get(e)[i], .85)), 'stroke-dasharray': `${dash.toFixed(2)} ${(BAND - dash).toFixed(2)}`,
              'stroke-dashoffset': (-offset).toFixed(2)});
            offset += dash;
          });
        }
      }
      still.remove();
      flow.before(still);
      if (animated) {
        flow.setAttribute('display', 'none');
        el(flow, 'set', {attributeName: 'display', to: 'inline', begin: '0s'});
        el(still, 'set', {attributeName: 'display', to: 'none', begin: '0s'});
      } else flow.remove();
      let index = 0;
      for (const e of edges) for (const seg of e.segments) {
        const line = lines[index++];
        if (!animated) continue;
        if (e.style.kind === 'zero') { line.remove(); continue; }
        const period = (e.style.kind === 'bands' ? BAND : WAVE) / WAVE_SPEED;
        const values = Array.from({length: frames + 1}, (_, k) => hex(segmentColor(e, seg, k * period / frames, 1)));
        el(line, 'animate', {attributeName: 'stroke', dur: seconds(period), repeatCount: 'indefinite', values: values.join(';')});
        const reach = (e.start + seg.s * e.scale) / frontSpeed;
        if (reach > .02) el(line, 'set', {attributeName: 'stroke-opacity', to: 0, dur: seconds(reach)});
      }
      for (const [id, n] of Object.entries(scene.nodes)) {
        const node = copy.querySelector(`[data-node="${id}"]`);
        if (!node || !animated) continue;
        const reached = arrival[id] / frontSpeed, busy = (ready[id] - arrival[id]) / frontSpeed;
        if (n.kind === 'tensor') { if (reached > .02) el(node, 'set', {attributeName: 'opacity', to: .2, dur: seconds(reached)}); continue; }
        if (busy < .02) continue;
        el(node.firstElementChild.tagName === 'path' ? node.querySelector('rect, circle') : node.firstElementChild,
          'set', {attributeName: 'stroke-opacity', to: .85, begin: seconds(reached), dur: seconds(busy)});
        if (!tinted[id]) continue;
        const glyph = node.querySelector(`[data-glyph="${id}"]`), tint = tinted[id];
        tint.querySelectorAll('[id]').forEach(m => {
          const old = m.id; m.id = `${old}-tint`;
          tint.querySelectorAll(`[mask="url(#${old})"]`).forEach(u => u.setAttribute('mask', `url(#${old}-tint)`));
        });
        tint.setAttribute('opacity', 0);
        el(tint, 'set', {attributeName: 'opacity', to: 1, begin: seconds(reached), dur: seconds(busy)});
        glyph.after(tint);
      }
      if (background) copy.insertBefore(el(copy, 'rect', {width: '100%', height: '100%', fill: background}), copy.firstChild);
      backdropAt = -Infinity;
      const text = new XMLSerializer().serializeToString(copy);
      paused = false; origin = performance.now(); start();
      return text;
    }

    let origin = performance.now(), frame = 0, paused = false, pausedAt = 0, dead = false;
    function loop(now) { if (dead || paused) return; draw((now - origin) / 1000); frame = requestAnimationFrame(loop); }
    function start() {
      cancelAnimationFrame(frame);
      if (reduced.matches) { draw(finish / frontSpeed); return; }
      frame = requestAnimationFrame(loop);
    }
    start();
    return {
      replay() { origin = performance.now(); paused = false; start(); },
      pause() { if (paused) return; paused = true; pausedAt = performance.now(); cancelAnimationFrame(frame); },
      resume() { if (!paused) return; origin += performance.now() - pausedAt; paused = false; start(); },
      seek(seconds) {
        cancelAnimationFrame(frame);
        paused = true; pausedAt = performance.now(); origin = pausedAt - seconds * 1000;
        draw(seconds);
      },
      get paused() { return paused; },
      frontSeconds: finish / frontSpeed,
      exportSVG,
      inspect() {
        return {
          tensors: Object.fromEntries(Object.entries(styles).map(([id, s]) => [id, s.kind === 'bands'
            ? {kind: 'bands', parts: s.parts.map(p => ({hex: p.hex, weight: +p.weight.toFixed(3)})), grid: s.grid}
            : {kind: s.kind, hex: s.hex, mixed: s.family?.mixed, sources: s.family?.sources, sameAs: s.sameAs, grid: s.grid}])),
          edges: edges.map(e => ({from: e.from, to: e.to, tensor: e.tensor, role: e.role ?? 'data',
            start: Math.round(e.start), scale: +e.scale.toFixed(2)})),
          arrival,
          layout: {width: scene.width ?? WIDTH, available, compact}
        };
      },
      destroy() { dead = true; cancelAnimationFrame(frame); }
    };
  }

  /* Scene files register with NeuralFlowScene({...}) in the order they load. */
  window.NeuralFlowScenes = window.NeuralFlowScenes ?? [];
  window.NeuralFlowScene = scene => { window.NeuralFlowScenes.push(scene); };
  window.NeuralFlow = {resolveLineage, pageTheme, mount};
})();
