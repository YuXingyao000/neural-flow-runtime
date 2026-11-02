/* Tier layout. Key tensors anchor the rows and blocks sit in the gaps between
   tensor rows: ranks follow the longest path with tensors on even rows and
   blocks/junctions on odd rows, so blocks that read the same row line up.
   Long edges get dummy points to be ordered and routed; x positions balance
   each node over its neighbours by isotonic regression. Edges run straight
   through a band and bend with vertical-tangent curves between bands. */
(() => {
  'use strict';
  const BLOCKS = new Set(['block', 'junction']);
  const DUMMY = 8, GAP = 38, MARGIN = 16;

  function topological(ids, edges) {
    const indegree = Object.fromEntries(ids.map(id => [id, 0]));
    edges.forEach(e => indegree[e.to]++);
    const queue = ids.filter(id => !indegree[id]), order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const e of edges) if (e.from === id && !--indegree[e.to]) queue.push(e.to);
    }
    if (order.length !== ids.length) throw new Error('tier layout needs an acyclic graph');
    return order;
  }

  /* Minimise Σ(x−desired)² keeping order and minimum gaps (pool adjacent violators). */
  function fit(desired, widths, gaps) {
    const offset = [];
    let acc = 0;
    desired.forEach((_, i) => { offset.push(acc); acc += widths[i] + (gaps[i] ?? 0); });
    const pools = [];
    desired.forEach((d, i) => {
      pools.push({sum: d - offset[i], n: 1});
      while (pools.length > 1 && pools.at(-2).sum / pools.at(-2).n > pools.at(-1).sum / pools.at(-1).n) {
        const last = pools.pop();
        pools.at(-1).sum += last.sum;
        pools.at(-1).n += last.n;
      }
    });
    const out = [];
    for (const pool of pools) for (let k = 0; k < pool.n; k++) out.push(pool.sum / pool.n + offset[out.length]);
    return out;
  }

  function tiers(scene, box, {tight = false} = {}) {
    const SEP = tight ? 14 : 22, DUMMY_SEP = tight ? 5 : 8;
    const ids = Object.keys(scene.nodes), edges = scene.edges;
    const odd = id => BLOCKS.has(scene.nodes[id].kind);
    const order = topological(ids, edges), rank = {};
    for (const id of order) {
      let r = Math.max(-1, ...edges.filter(e => e.to === id).map(e => rank[e.from])) + 1;
      if ((r % 2 === 1) !== odd(id)) r += 1;
      rank[id] = r;
    }
    for (const id of [...order].reverse()) {
      if (edges.some(e => e.to === id)) continue;
      let r = Math.min(...edges.filter(e => e.from === id).map(e => rank[e.to])) - 1;
      if ((r % 2 === 1) !== odd(id)) r -= 1;
      if (Number.isFinite(r)) rank[id] = Math.max(rank[id], r);
    }

    /* Weakly connected parts are laid out alone and packed afterwards; left
       together they drift apart, since nothing ties one to another. */
    const part = Object.fromEntries(ids.map(id => [id, id]));
    const root = id => part[id] === id ? id : (part[id] = root(part[id]));
    edges.forEach(e => { part[root(e.from)] = root(e.to); });

    /* A graph whose sources are all blocks would leave the first tensor row empty. */
    const lowest = Math.min(...Object.values(rank));
    for (const id of ids) rank[id] -= lowest;
    const top = Math.max(...Object.values(rank));
    const layers = Array.from({length: top + 1}, () => []);
    const items = {};
    for (const id of ids) {
      const item = {id, b: box(id), up: [], down: [], part: root(id), rank: rank[id]};
      item.w = item.b.w;
      items[id] = item;
      layers[rank[id]].push(item);
    }
    /* Edges leaving one tensor share their dummies rank by rank, so a tensor
       read far below runs as one line until its branches part. */
    const bundles = new Map();
    const link = (a, b) => { if (!a.down.includes(b)) { a.down.push(b); b.up.push(a); } };
    edges.forEach(e => {
      let previous = items[e.from];
      e.chain = [];
      for (let r = rank[e.from] + 1; r < rank[e.to]; r++) {
        const key = `${e.from}@${r}`;
        if (!bundles.has(key)) {
          const dummy = {dummy: true, w: DUMMY, up: [], down: [], part: root(e.from), rank: r, side: e.side};
          bundles.set(key, dummy);
          layers[r].push(dummy);
        }
        const dummy = bundles.get(key);
        e.chain.push(dummy);
        link(previous, dummy);
        previous = dummy;
      }
      link(previous, items[e.to]);
    });

    /* Order: barycentre sweeps; ties keep declaration order as the author's hint. */
    layers.forEach(layer => layer.forEach((item, k) => { item.pos = k; }));
    for (let pass = 0; pass < 12; pass++) {
      const down = pass % 2 === 0;
      const sequence = down ? layers.slice(1) : layers.slice(0, -1).reverse();
      for (const layer of sequence) {
        /* An edge with side: 'left' | 'right' keeps its long run at that edge of the figure. */
        const centre = new Map(layer.map(item => {
          if (item.side) return [item, item.side === 'left' ? -1e6 : 1e6];
          const near = down ? item.up : item.down;
          return [item, near.length ? near.reduce((a, n) => a + n.pos, 0) / near.length : item.pos];
        }));
        layer.sort((a, b) => centre.get(a) - centre.get(b));
        layer.forEach((item, k) => { item.pos = k; });
      }
    }

    /* x: pack, then pull every node's port over its neighbours' ports. */
    const gapsOf = layer => layer.map((item, k) => k + 1 < layer.length
      ? (item.dummy && layer[k + 1].dummy ? DUMMY_SEP : SEP) : 0);
    const portX = item => item.x + (item.dummy ? DUMMY / 2 : item.b.ax);
    layers.forEach(layer => {
      const placed = fit(layer.map(() => 0), layer.map(i => i.w), gapsOf(layer));
      layer.forEach((item, k) => { item.x = placed[k]; });
    });
    for (let pass = 0; pass < 40; pass++) {
      const down = pass % 2 === 0;
      for (const layer of down ? layers.slice(1) : layers.slice(0, -1).reverse()) {
        const desired = layer.map(item => {
          const near = (down ? item.up : item.down).length ? (down ? item.up : item.down) : [...item.up, ...item.down];
          if (!near.length) return item.x;
          return near.reduce((a, n) => a + portX(n), 0) / near.length - (item.dummy ? DUMMY / 2 : item.b.ax);
        });
        const placed = fit(desired, layer.map(i => i.w), gapsOf(layer));
        layer.forEach((item, k) => { item.x = placed[k]; });
      }
    }
    const all = layers.flat();
    const parts = [...new Set(ids.map(root))];
    const occupied = {};
    for (const p of parts) {
      const members = all.filter(i => i.part === p), lo = Math.min(...members.map(i => i.x));
      members.forEach(i => { i.x -= lo; });
      let offset = 0;
      for (const i of members) if (i.rank in occupied) {
        const leftmost = Math.min(...members.filter(m => m.rank === i.rank).map(m => m.x));
        offset = Math.max(offset, occupied[i.rank] + SEP - leftmost);
      }
      members.forEach(i => { i.x += offset; occupied[i.rank] = Math.max(occupied[i.rank] ?? -Infinity, i.x + i.w); });
    }
    const left = Math.min(...all.map(i => i.x)), right = Math.max(...all.map(i => i.x + i.w));
    const width = Math.max(scene.minWidth ?? 680, right - left + 2 * MARGIN);
    const shift = (width - (right - left)) / 2 - left;
    all.forEach(item => { item.x += shift; });

    /* y: bands of the tallest box in each rank. */
    const bandTop = [], bandHeight = [];
    const labelled = (scene.modules ?? []).some(m => m.nodes.some(id => rank[id] === 0));
    let y = MARGIN + (labelled ? 26 : 0);
    layers.forEach((layer, r) => {
      const h = Math.max(10, ...layer.filter(i => !i.dummy).map(i => i.b.h));
      bandTop[r] = y; bandHeight[r] = h;
      y += h + GAP;
    });
    for (const id of ids) {
      const item = items[id], r = rank[id];
      item.y = bandTop[r] + (bandHeight[r] - item.b.h) / 2;
      Object.assign(scene.nodes[id], item.b.place(item.x, item.y));
      item.port = item.b.ports(item.x, item.y);
    }

    /* Ports spread along each edge of a node in the order of the far ends. */
    const spread = (list, port, side) => {
      list.sort((a, b) => a.far - b.far);
      const step = Math.min(10, list.length > 1 ? port.span / (list.length - 1) : 0);
      list.forEach((entry, k) => { entry.edge[side] = port.x + (k - (list.length - 1) / 2) * step; });
    };
    for (const id of ids) {
      const item = items[id];
      spread(edges.filter(e => e.to === id).map(e => ({edge: e, far: e.chain.length ? portX(e.chain.at(-1)) : items[e.from].port.x})), item.port, 'inX');
      const outgoing = edges.filter(e => e.from === id).map(e => ({edge: e, far: e.chain.length ? portX(e.chain[0]) : items[e.to].port.x}));
      const groups = [...new Map(outgoing.map(o => [o.far, []])).keys()].map(far => ({far, members: outgoing.filter(o => o.far === far)}));
      const shared = groups.map(g => ({edge: {}, far: g.far, members: g.members}));
      spread(shared, item.port, 'outX');
      shared.forEach(g => g.members.forEach(o => { o.edge.outX = g.edge.outX; }));
    }
    for (const e of edges) {
      const from = items[e.from].port, to = items[e.to].port;
      const points = [[e.outX, from.bottom]];
      for (const dummy of e.chain) {
        const r = layers.findIndex(layer => layer.includes(dummy)), x = portX(dummy);
        points.push([x, bandTop[r]], [x, bandTop[r] + bandHeight[r], true]);
      }
      points.push([e.inX, to.top]);
      let d = `M${points[0][0].toFixed(1)} ${points[0][1].toFixed(1)}`;
      for (let k = 1; k < points.length; k++) {
        const [x1, y1] = points[k - 1], [x2, y2, through] = points[k], ym = (y1 + y2) / 2;
        d += through ? ` L${x2.toFixed(1)} ${y2.toFixed(1)}`
          : ` C${x1.toFixed(1)} ${ym.toFixed(1)} ${x2.toFixed(1)} ${ym.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
      }
      e.d = d;
      if (typeof e.label === 'string') {
        const [[x1, y1], [x2, y2]] = points;
        e.label = [x1 + (x2 - x1) * .15 + 8, y1 + (y2 - y1) * .3 + 4, e.label, 'start'];
        e.labelPlaced = true;
      }
    }
    /* Labels of edges that leave one port start at the same spot; step any that
       would overlap an earlier one down by a line. */
    const taken = [];
    for (const e of edges.filter(e => e.labelPlaced)) {
      const [x, , text] = e.label, w = [...text].length * 7;
      while (taken.some(t => x < t.x + t.w && t.x < x + w && Math.abs(e.label[1] - t.y) < 14)) e.label[1] += 14;
      taken.push({x, y: e.label[1], w});
    }

    /* Modules: the bounding box of their members, or a full-width band. A band
       spans its members' rows, so everything in those rows sits inside it. */
    scene.modules = (scene.modules ?? []).filter(m => m.nodes.some(id => items[id])).map(m => {
      const members = m.nodes.filter(id => items[id]).map(id => items[id]);
      const x0 = Math.min(...members.map(i => i.x)) - 12, x1 = Math.max(...members.map(i => i.x + i.w)) + 12;
      const y0 = Math.min(...members.map(i => i.y)) - 26, y1 = Math.max(...members.map(i => i.y + i.b.h)) + 10;
      /* A repeated module draws two cards 8 px up and to the right; keep them inside. */
      const deck = m.repeat > 1 ? 8 : 0;
      const box = m.band ? {x: 8, y: y0, w: width - 16 - deck, h: y1 - y0} : {x: x0, y: y0, w: x1 - x0, h: y1 - y0};
      return {...box, label: m.label, repeat: m.repeat};
    });
    scene.width = Math.max(width, ...scene.modules.map(m => m.x + m.w + (m.repeat > 1 ? 8 : 0) + 2));
    scene.height = y - GAP + MARGIN;
  }

  window.NeuralLayout = {tiers};
})();
