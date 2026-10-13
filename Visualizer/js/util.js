// util.js - math, colormaps and small helpers shared by the viewer (global SV namespace)
(function () {
  'use strict';
  const SV = (window.SV = window.SV || {});

  // ------------------------------------------------------------- vec / mat4
  const V = {
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  };
  SV.V = V;

  SV.mat4 = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    ortho(l, r, b, t, n, f) {
      return new Float32Array([2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, -2 / (f - n), 0,
        -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1]);
    },
    lookAt(eye, target, up) {
      const z = V.norm(V.sub(eye, target)), x = V.norm(V.cross(up, z)), y = V.cross(z, x);
      return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
        -V.dot(x, eye), -V.dot(y, eye), -V.dot(z, eye), 1]);
    },
    mul(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++)
        for (let r = 0; r < 4; r++) {
          let s = 0;
          for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
          o[c * 4 + r] = s;
        }
      return o;
    },
  };

  // ------------------------------------------------------------- colormaps
  const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  const MAPS = {
    viridis: ['#440154', '#481a6c', '#472f7d', '#414487', '#39568c', '#31688e', '#2a788e', '#23888e',
      '#1f988b', '#22a884', '#35b779', '#54c568', '#7ad151', '#a5db36', '#d2e21b', '#fde725'].map(hex),
    inferno: ['#000004', '#1f0c48', '#550f6d', '#88226a', '#ba3655', '#e35933', '#f98e09', '#f9cb35', '#fcffa4'].map(hex),
    gray: ['#000000', '#ffffff'].map(hex),
  };
  SV.colormapNames = Object.keys(MAPS);
  SV.cmap = function (t, name) {
    const m = MAPS[name] || MAPS.viridis;
    if (!(t >= 0)) t = 0;
    if (t > 1) t = 1;
    const x = t * (m.length - 1), i = Math.min(m.length - 2, Math.floor(x)), f = x - i;
    const a = m[i], b = m[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };
  // categorical colours for discrete values (layers, flags, materials)
  const CAT = ['#4ea1ff', '#f0a54a', '#3fb950', '#e05d8a', '#a371f7', '#2fc4c4', '#d2a8ff', '#ffd33d', '#ff7b72', '#79c0ff'].map(hex);
  SV.catColor = (i) => CAT[((i % CAT.length) + CAT.length) % CAT.length];
  SV.rgbCss = (c, a) => a === undefined
    ? `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`
    : `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;

  // value range helper with optional log scale
  SV.makeScale = function (values, log) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!isFinite(v)) continue;
      if (log && v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (log) {
      lo = Math.max(lo, hi * 1e-6);
      const a = Math.log10(lo), b = Math.log10(hi);
      return { lo, hi, log: true, t: (v) => (v > 0 && b > a ? (Math.log10(Math.max(v, lo)) - a) / (b - a) : 0) };
    }
    if (hi === lo) return { lo, hi, log: false, t: () => 0.5 };  // constant data: mid colour
    return { lo, hi, log: false, t: (v) => (v - lo) / (hi - lo) };
  };

  // ------------------------------------------------------------- formatting
  SV.fmt = function (v, d = 4) {
    if (v === undefined || v === null || Number.isNaN(v)) return '–';
    if (typeof v !== 'number') return String(v);
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(1, d - 1));
    return Number(v.toPrecision(d)).toString();
  };
  SV.bytes = function (n) {
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 2 : 0)} ${u[i]}`;
  };
  SV.esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  SV.clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  SV.storage = {
    get(k, d) { try { const v = localStorage.getItem('sv.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('sv.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
})();
