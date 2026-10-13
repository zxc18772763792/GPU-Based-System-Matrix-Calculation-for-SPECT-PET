// plots.js - 2D views drawn on one canvas: x-z layout map, voxel image, energy spectrum.
(function () {
  'use strict';
  const SV = (window.SV = window.SV || {});

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(Math.abs(v) < 1e-12 * span ? 0 : v);
    return out;
  }
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ------------------------------------------------------------------ host
  class Plot2D {
    constructor(canvas, tip) {
      this.cv = canvas; this.tip = tip; this.view = null; this.dirty = true;
      const loop = () => {
        requestAnimationFrame(loop);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; this.dirty = true; }
        if (!this.dirty) return;
        this.dirty = false;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
        if (this.view) this.view.draw(ctx, canvas.clientWidth, canvas.clientHeight);
      };
      requestAnimationFrame(loop);
      let drag = null;
      const pos = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      canvas.addEventListener('pointerdown', (e) => { drag = { p: pos(e), p0: pos(e) }; canvas.setPointerCapture(e.pointerId); });
      canvas.addEventListener('pointermove', (e) => {
        const p = pos(e);
        if (drag && this.view && this.view.pan) { this.view.pan(p[0] - drag.p[0], p[1] - drag.p[1]); drag.p = p; this.dirty = true; }
        const t = this.view && this.view.hover ? this.view.hover(p[0], p[1]) : null;
        if (t) { tip.innerHTML = t; tip.hidden = false; tip.style.left = `${p[0] + 14}px`; tip.style.top = `${p[1] + 10}px`; } else tip.hidden = true;
      });
      canvas.addEventListener('pointerup', (e) => {
        const p = pos(e);
        if (drag && Math.abs(p[0] - drag.p0[0]) + Math.abs(p[1] - drag.p0[1]) < 4 && this.view && this.view.click) this.view.click(p[0], p[1]);
        drag = null;
      });
      canvas.addEventListener('pointerleave', () => { tip.hidden = true; });
      canvas.addEventListener('wheel', (e) => {
        if (!this.view || !this.view.zoom) return;
        e.preventDefault();
        const p = pos(e);
        this.view.zoom(Math.exp(-e.deltaY * 0.0015), p[0], p[1]);
        this.dirty = true;
      }, { passive: false });
      new ResizeObserver(() => { this.dirty = true; }).observe(canvas);
    }
    setView(v) { this.view = v; v.host = this; this.dirty = true; }
    redraw() { this.dirty = true; }
  }
  SV.Plot2D = Plot2D;

  function frame(ctx, w, h, x0, x1, y0, y1, X, Y, labels, yTicks) {
    ctx.strokeStyle = css('--grid');
    ctx.fillStyle = css('--muted');
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui, sans-serif';
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(w / 90)))) {
      const px = X(v);
      ctx.beginPath(); ctx.moveTo(px, 8); ctx.lineTo(px, h - 26); ctx.stroke();
      if (px > 60 && px < w - 20) { ctx.textAlign = 'center'; ctx.fillText(SV.fmt(v, 3), px, h - 12); }
    }
    for (const v of yTicks || niceTicks(y0, y1, Math.max(2, Math.floor(h / 50)))) {
      const py = Y(v);
      ctx.beginPath(); ctx.moveTo(48, py); ctx.lineTo(w - 8, py); ctx.stroke();
      if (py > 30 && py < h - 30) { ctx.textAlign = 'right'; ctx.fillText(SV.fmt(v, 3), 44, py + 4); }
    }
    if (labels) {
      ctx.textAlign = 'right'; ctx.fillText(labels[0], w - 10, h - 28);
      ctx.textAlign = 'left'; ctx.fillText(labels[1], 54, 20);
    }
  }

  // ------------------------------------------------------------------ x-z layout
  class LayoutView {
    // Beam's-eye view: looking along +y (from the FOV towards the detector), x right, z up.
    // mirror = true looks along -y (from behind the detector): x points left.
    constructor(app) { this.app = app; this.scale = 4; this.cx = 0; this.cz = 0; this.filter = 'all'; this.fitted = false; this.mirror = false; }
    get sgn() { return this.mirror ? -1 : 1; }
    fit(w, h) {
      const m = this.app.model;
      if (!m) return;
      const b = m.bounds;
      const sx = (w - 70) / Math.max(1, b.max[0] - b.min[0]), sz = (h - 50) / Math.max(1, b.max[2] - b.min[2]);
      this.scale = Math.max(0.05, Math.min(sx, sz));
      this.cx = (b.max[0] + b.min[0]) / 2; this.cz = (b.max[2] + b.min[2]) / 2;
      this.fitted = true;
    }
    X(x) { return this.w / 2 + 20 + this.sgn * (x - this.cx) * this.scale; }
    Z(z) { return this.h / 2 - 10 - (z - this.cz) * this.scale; }
    pan(dx, dy) { this.cx -= this.sgn * dx / this.scale; this.cz += dy / this.scale; }
    zoom(f, px, py) {
      const [x, z] = this.toWorld(px, py);
      this.scale *= f;
      this.cx = x - this.sgn * (px - this.w / 2 - 20) / this.scale; this.cz = z + (py - this.h / 2 + 10) / this.scale;
    }
    toWorld(px, py) { return [this.cx + this.sgn * (px - this.w / 2 - 20) / this.scale, this.cz - (py - this.h / 2 + 10) / this.scale]; }
    draw(ctx, w, h) {
      this.w = w; this.h = h;
      const m = this.app.model;
      if (!m) return this.app.emptyHint(ctx, w, h);
      if (!this.fitted) this.fit(w, h);
      const xa = this.toWorld(0, 0)[0], xb = this.toWorld(w, 0)[0], z1 = this.toWorld(0, 0)[1], z0 = this.toWorld(0, h)[1];
      frame(ctx, w, h, Math.min(xa, xb), Math.max(xa, xb), z0, z1, (v) => this.X(v), (v) => this.Z(v),
        [this.mirror ? '← x (mm)' : 'x (mm) →', 'z (mm) ↑']);
      const showX = this.filter === 'all' || this.filter === 'crystals';
      const sel = this.app.sel;
      // crystals
      if (showX) {
        const cols = this.app.crystalColors;
        for (const c of m.crystals) {
          const ca = Math.cos(c.rot), sa = Math.sin(c.rot);
          ctx.beginPath();
          [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([sx, sz], k) => {
            const lx = sx * c.w / 2, lz = sz * c.h / 2;
            const px = this.X(c.x + lx * ca - lz * sa), pz = this.Z(c.z + lx * sa + lz * ca);
            if (k) ctx.lineTo(px, pz); else ctx.moveTo(px, pz);
          });
          ctx.closePath();
          const col = cols ? [cols[4 * c.index], cols[4 * c.index + 1], cols[4 * c.index + 2]] : [0.4, 0.5, 0.6];
          ctx.fillStyle = c.enabled ? SV.rgbCss(col, 0.85) : 'rgba(120,120,120,0.25)';
          ctx.fill();
          if (sel && sel.kind === 'crystal' && sel.index === c.index) { ctx.strokeStyle = '#ff4060'; ctx.lineWidth = 2; ctx.stroke(); }
        }
      }
      // plates and holes
      for (const L of m.layers) {
        if (this.filter !== 'all' && this.filter !== String(L.index)) continue;
        const lc = SV.catColor(L.index);
        ctx.strokeStyle = SV.rgbCss(lc, 0.9); ctx.lineWidth = 1.5;
        ctx.strokeRect(Math.min(this.X(-L.W / 2), this.X(L.W / 2)), this.Z(L.H / 2), L.W * this.scale, L.H * this.scale);
        ctx.fillStyle = SV.rgbCss(lc, 0.55);
        for (let k = 0; k < L.nHoles; k++) {
          const hh = m.holes[L.holeStart + k];
          if (!hh) break;
          ctx.beginPath(); ctx.arc(this.X(hh.x), this.Z(hh.z), Math.max(1, hh.r * this.scale), 0, 2 * Math.PI); ctx.fill();
          if (sel && sel.kind === 'hole' && sel.index === hh.index) { ctx.strokeStyle = '#ff4060'; ctx.lineWidth = 2; ctx.stroke(); ctx.strokeStyle = SV.rgbCss(lc, 0.9); ctx.lineWidth = 1.5; }
        }
      }
      // origin
      ctx.strokeStyle = css('--muted'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.X(0) - 6, this.Z(0)); ctx.lineTo(this.X(0) + 6, this.Z(0)); ctx.moveTo(this.X(0), this.Z(0) - 6); ctx.lineTo(this.X(0), this.Z(0) + 6); ctx.stroke();
      // viewing direction caption
      ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.fillStyle = css('--muted');
      ctx.fillText(this.app.t(this.mirror ? 'layoutViewR' : 'layoutView'), w - 12, 20);
    }
    pick(px, py) {
      const m = this.app.model;
      if (!m) return null;
      const [x, z] = this.toWorld(px, py);
      if (this.filter !== 'crystals')
        for (const hh of m.holes)
          if ((this.filter === 'all' || this.filter === String(hh.layer)) && Math.hypot(hh.x - x, hh.z - z) <= Math.max(hh.r, 2 / this.scale)) return { kind: 'hole', index: hh.index };
      if (this.filter === 'all' || this.filter === 'crystals')
        for (const c of m.crystals) {
          const dx = x - c.x, dz = z - c.z, ca = Math.cos(c.rot), sa = Math.sin(c.rot);
          if (Math.abs(dx * ca + dz * sa) <= c.w / 2 && Math.abs(-dx * sa + dz * ca) <= c.h / 2) return { kind: 'crystal', index: c.index };
        }
      return null;
    }
    hover(px, py) {
      const p = this.pick(px, py), [x, z] = this.toWorld(px, py);
      return `x=${SV.fmt(x, 4)} z=${SV.fmt(z, 4)} mm` + (p ? `<br>${this.app.describeShort(p)}` : '');
    }
    click(px, py) { const p = this.pick(px, py); if (p) this.app.select(p); }
  }
  SV.LayoutView = LayoutView;

  // ------------------------------------------------------------------ voxel image
  // Orientation modes (app.state.imgOrient):
  //   phys   : x (ix) to the right, y (iy) up    = seen from +z, same as the 3D "-z" view; detector on top at rotation 0
  //   numpy  : ix to the right, iy downwards     = imshow(row.reshape(nZ, nY, nX)[iz])
  //   matlab : iy to the right, ix downwards     = imagesc(reshape(row, nX, nY, nZ)(:,:,iz))
  class ImageView {
    constructor(app) { this.app = app; this.slice = 0; this.off = document.createElement('canvas'); }
    get mode() { return this.app.state.imgOrient || 'phys'; }
    dims() {
      const I = this.app.model.image;
      return this.mode === 'matlab' ? { cols: I.nY, rows: I.nX, cw: I.wY, rh: I.wX } : { cols: I.nX, rows: I.nY, cw: I.wX, rh: I.wY };
    }
    toCell(ix, iy) {
      const I = this.app.model.image;
      return this.mode === 'matlab' ? [iy, ix] : this.mode === 'numpy' ? [ix, iy] : [ix, I.nY - 1 - iy];
    }
    fromCell(cx, cy) {
      const I = this.app.model.image;
      return this.mode === 'matlab' ? [cy, cx] : this.mode === 'numpy' ? [cx, cy] : [cx, I.nY - 1 - cy];
    }
    // screen direction (x right, y down) of a direction (dx, dy) given in voxel x/y
    screenDir(dx, dy) { return this.mode === 'matlab' ? [dy, dx] : this.mode === 'numpy' ? [dx, dy] : [dx, -dy]; }
    rect(w, h) {
      const d = this.dims();
      const aw = d.cols * d.cw, ah = d.rows * d.rh;
      const s = Math.min((w - 140) / aw, (h - 44) / ah);
      return { x: 64 + (w - 140 - aw * s) / 2, y: 22 + (h - 44 - ah * s) / 2, w: aw * s, h: ah * s };
    }
    draw(ctx, w, h) {
      const app = this.app, vv = app.voxValues;
      if (!app.model || !vv) return app.emptyHint(ctx, w, h, app.t('hintImage'));
      const I = app.model.image, d = this.dims();
      this.slice = SV.clamp(this.slice, 0, I.nZ - 1);
      const n = I.nX * I.nY, base = this.slice * n;
      const sc = SV.makeScale(vv.values.subarray(base, base + n), app.state.log);
      this.off.width = d.cols; this.off.height = d.rows;
      const octx = this.off.getContext('2d'), im = octx.createImageData(d.cols, d.rows);
      for (let iy = 0; iy < I.nY; iy++)
        for (let ix = 0; ix < I.nX; ix++) {
          const v = vv.values[base + iy * I.nX + ix], c = SV.cmap(sc.t(v), app.state.cmap);
          const [cx, cy] = this.toCell(ix, iy), o = 4 * (cy * d.cols + cx);
          im.data[o] = c[0] * 255; im.data[o + 1] = c[1] * 255; im.data[o + 2] = c[2] * 255; im.data[o + 3] = 255;
        }
      octx.putImageData(im, 0, 0);
      const r = this.rect(w, h);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.off, r.x, r.y, r.w, r.h);
      ctx.strokeStyle = css('--border'); ctx.strokeRect(r.x, r.y, r.w, r.h);
      // selected voxel
      const s = app.sel;
      if (s && s.kind === 'voxel') {
        const p = SV.voxelPos(I, s.index, 0);
        if (p.iz === this.slice) {
          const [cx, cy] = this.toCell(p.ix, p.iy);
          ctx.strokeStyle = '#ff4060'; ctx.lineWidth = 2;
          ctx.strokeRect(r.x + cx * r.w / d.cols, r.y + cy * r.h / d.rows, r.w / d.cols, r.h / d.rows);
        }
      }
      // axis arrows
      ctx.fillStyle = css('--muted'); ctx.font = '11px system-ui, sans-serif';
      const lab = this.mode === 'matlab' ? ['iy (y) →', 'ix (x) ↓'] : this.mode === 'numpy' ? ['ix (x) →', 'iy (y) ↓'] : ['x (ix) →', 'y (iy) ↑'];
      ctx.textAlign = 'center'; ctx.fillText(lab[0], r.x + r.w / 2, r.y + r.h + 14);
      ctx.save(); ctx.translate(r.x - 12, r.y + r.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(lab[1], 0, 0); ctx.restore();
      // corner index labels: where voxel (0, 0) is
      const [c0x, c0y] = this.toCell(0, 0);
      ctx.textAlign = c0x === 0 ? 'left' : 'right';
      ctx.fillText('(0,0)', c0x === 0 ? r.x : r.x + r.w, c0y === 0 ? r.y - 5 : r.y + r.h + 14);
      // direction of the detector (+y in the world) for the current rotation, in voxel coordinates
      const a = app.state.rot * I.angle, [sx, sy] = this.screenDir(Math.sin(a), Math.cos(a));
      const ccx = r.x + r.w / 2, ccy = r.y + r.h / 2, L = 0.42 * Math.min(r.w, r.h);
      const ex = ccx + sx * L, ey = ccy + sy * L;
      ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ccx, ccy); ctx.lineTo(ex, ey); ctx.stroke();
      const ang = Math.atan2(sy, sx);
      ctx.beginPath(); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 9 * Math.cos(ang - 0.4), ey - 9 * Math.sin(ang - 0.4));
      ctx.lineTo(ex - 9 * Math.cos(ang + 0.4), ey - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      ctx.textAlign = 'center';
      ctx.fillText(app.t('detDir', app.state.rot), SV.clamp(ex + sx * 14, r.x, r.x + r.w), SV.clamp(ey + sy * 14 + 4, r.y + 8, r.y + r.h));
      // caption + colourbar
      ctx.fillStyle = css('--muted'); ctx.textAlign = 'left';
      ctx.fillText(`${vv.label}  ·  z = ${this.slice}/${I.nZ - 1}`, 8, h - 6);
      const bx = w - 58, by = 14, bh = h - 40;
      for (let k = 0; k < bh; k++) { ctx.fillStyle = SV.rgbCss(SV.cmap(1 - k / bh, app.state.cmap)); ctx.fillRect(bx, by + k, 12, 1); }
      ctx.fillStyle = css('--muted'); ctx.textAlign = 'left';
      ctx.fillText(SV.fmt(sc.hi, 3), bx + 15, by + 8);
      ctx.fillText(SV.fmt(sc.lo, 3), bx + 15, by + bh);
      if (sc.log) ctx.fillText('log', bx + 15, by + bh / 2);
    }
    voxelAt(px, py) {
      const app = this.app;
      if (!app.model || !app.voxValues) return -1;
      const I = app.model.image, d = this.dims(), r = this.rect(this.host.cv.clientWidth, this.host.cv.clientHeight);
      const cx = Math.floor((px - r.x) / r.w * d.cols), cy = Math.floor((py - r.y) / r.h * d.rows);
      if (cx < 0 || cy < 0 || cx >= d.cols || cy >= d.rows) return -1;
      const [ix, iy] = this.fromCell(cx, cy);
      return this.slice * I.nX * I.nY + iy * I.nX + ix;
    }
    hover(px, py) {
      const v = this.voxelAt(px, py);
      if (v < 0) return null;
      const p = SV.voxelPos(this.app.model.image, v, this.app.state.rot);
      return `voxel ${v}  (ix ${p.ix}, iy ${p.iy}, iz ${p.iz})<br>x, y = ${SV.fmt(p.local[0], 4)}, ${SV.fmt(p.local[1], 4)} mm<br>${SV.fmt(this.app.voxValues.values[v], 5)}`;
    }
    click(px, py) { const v = this.voxelAt(px, py); if (v >= 0) this.app.select({ kind: 'voxel', index: v }); }
  }
  SV.ImageView = ImageView;

  // ------------------------------------------------------------------ spectrum
  class SpectrumView {
    constructor(app) { this.app = app; }
    draw(ctx, w, h) {
      const app = this.app, sp = app.spectrum;
      if (!sp) return app.emptyHint(ctx, w, h, app.t('hintSpectrum'));
      const pre = app.state.specMode === 'raw' ? 'raw_' : 'blur_';
      const series = [
        ['primary', '#4ea1ff'], ['colscatter', '#f0a54a'], ['xtalscatter', '#3fb950'],
      ].map(([k, c]) => ({ k, c, v: sp.cols[pre + k] || [] }));
      const n = sp.E.length, tot = new Float32Array(n);
      for (const s of series) for (let i = 0; i < n; i++) tot[i] += s.v[i] || 0;
      const log = app.state.specLog;
      const sc = SV.makeScale(tot, log);
      const x0 = 0, x1 = sp.E[n - 1] + sp.binW / 2;
      const X = (e) => 52 + (e - x0) / (x1 - x0) * (w - 64);
      const Y = (v) => { const t = log ? sc.t(v) : v / (sc.hi || 1); return h - 28 - SV.clamp(t, 0, 1) * (h - 44); };
      this.X = X; this.x0 = x0; this.x1 = x1; this.tot = tot; this.series = series;
      // windows
      for (const win of app.windows()) {
        ctx.fillStyle = 'rgba(78,161,255,0.12)';
        ctx.fillRect(X(win.lo), 8, X(win.hi) - X(win.lo), h - 36);
      }
      let yt = null;
      if (log) {
        yt = [];
        for (let d = Math.ceil(Math.log10(sc.lo)); d <= Math.floor(Math.log10(sc.hi)); d++) yt.push(Math.pow(10, d));
      }
      frame(ctx, w, h, x0, x1, log ? sc.lo : 0, sc.hi, X, (v) => Y(v), ['E (keV)', app.t('countsPerPhoton')], yt);
      const line = (vals, color, width) => {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
        for (let i = 0; i < n; i++) { const px = X(sp.E[i]), py = Y(vals[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
        ctx.stroke();
      };
      for (const s of series) line(s.v, s.c, 1.2);
      line(tot, css('--text'), 1.6);
      // legend
      ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left';
      let ly = 20;
      for (const [label, color] of [[app.t('total'), css('--text')], [app.t('primary'), '#4ea1ff'], [app.t('colscatter'), '#f0a54a'], [app.t('xtalscatter'), '#3fb950']]) {
        ctx.fillStyle = color; ctx.fillRect(w - 150, ly - 8, 10, 3); ctx.fillText(label, w - 135, ly - 4); ly += 15;
      }
    }
    hover(px) {
      const sp = this.app.spectrum;
      if (!sp || !this.X) return null;
      const e = this.x0 + (px - 52) / (this.host.cv.clientWidth - 64) * (this.x1 - this.x0);
      const i = SV.clamp(Math.round(e / sp.binW - 0.5), 0, sp.E.length - 1);
      return `E = ${SV.fmt(sp.E[i], 4)} keV<br>${this.app.t('total')}: ${SV.fmt(this.tot[i], 3)}` +
        this.series.map((s) => `<br>${this.app.t(s.k)}: ${SV.fmt(s.v[i], 3)}`).join('');
    }
  }
  SV.SpectrumView = SpectrumView;
})();
