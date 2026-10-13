// app.js - viewer state, scene building, picking, panels and file loading.
(function () {
  'use strict';
  const SV = window.SV;
  const $ = (id) => document.getElementById(id);
  const t = SV.t;

  const VTX_COL = { 0: [1, 1, 1], 1: [1, 0.3, 0.3], 2: [0.3, 0.85, 1], 3: [0.4, 0.95, 0.45], 5: [0.9, 0.45, 1], 6: [1, 0.85, 0.2], 7: [0.45, 0.45, 0.45] };
  const HOLE_COL = [1.0, 0.78, 0.38];

  const App = {
    t,
    model: null,
    f32: {},
    sysmats: [],          // {src, sm}
    cur: -1,              // selected sysmat
    spectrum: null, tracks: null, summary: null,
    sel: null,
    voxValues: null,      // {values, label, kind}
    projValues: null,     // per detector, projection of the selected voxel
    detSumValues: null,   // per detector, sum over voxels
    crystalColors: null,
    xtalInst: null,       // crystal index -> [batch, instance]
    state: {
      cmap: SV.storage.get('cmap', 'viridis'), log: false, colorBy: 'muTot', rot: 0, plateOpacity: 0.3,
      specMode: 'blur', specLog: true,
    },
  };
  SV.app = App;

  // ================================================================ status
  function status(msg, err) { const s = $('status'); s.textContent = msg || ''; s.classList.toggle('err', !!err); }
  function fail(e) { console.error(e); status(t('stError', e.message || e), true); }
  App.emptyHint = function (ctx, w, h, text) {
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted');
    ctx.font = '13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text || t('hintEmpty'), w / 2, h / 2);
  };

  // ================================================================ loading
  async function loadSources(list) {
    let n = 0, geom = false;
    for (const src of list) {
      const kind = SV.classify(src.name);
      if (!kind) continue;
      try {
        if (['col', 'det', 'img', 'phy'].includes(kind)) {
          App.f32[kind] = new Float32Array(await src.read(0, src.size - (src.size % 4)));
          geom = true;
        } else if (kind === 'sysmat') {
          if (!App.sysmats.some((s) => s.src.path === src.path && s.src.size === src.size)) App.sysmats.push({ src, sm: null });
        } else if (kind === 'spectrum') App.spectrum = SV.parseSpectrum(await src.text());
        else if (kind === 'tracks') App.tracks = JSON.parse(await src.text());
        else if (kind === 'summary') App.summary = JSON.parse(await src.text());
        n++;
      } catch (e) { fail(e); }
    }
    if (geom && (App.f32.col || App.f32.det)) buildModel();
    if (App.cur < 0 && App.sysmats.length) App.cur = App.sysmats.length - 1;
    refreshSysmat();
    buildTracks();
    refreshMC();
    refreshChips();
    App.plot.redraw();
    if (n) status(t('stLoaded', n));
    return n;
  }

  function demo() {
    App.f32 = SV.demoParams();
    App.sysmats = []; App.cur = -1; App.spectrum = App.tracks = App.summary = null;
    buildModel();
    refreshSysmat(); buildTracks(); refreshMC(); refreshChips();
    status(t('stDemo'));
  }

  // ================================================================ model + scene
  function buildModel() {
    const m = SV.parseParams(App.f32);
    App.model = m;
    App.sel = null; App.voxValues = null; App.projValues = null; App.detSumValues = null;
    $('hint3d').hidden = true;
    const I = m.image;
    $('rotIndex').max = Math.max(0, I.nRot - 1);
    $('rotIndex').value = 0; App.state.rot = 0;
    $('detIdx').max = Math.max(0, m.crystals.length - 1);
    $('voxIdx').max = Math.max(0, I.nVox - 1);
    $('sliceIdx').max = Math.max(0, I.nZ - 1);
    $('sliceIdx').value = Math.floor(I.nZ / 2);
    App.views.image.slice = Math.floor(I.nZ / 2);
    buildModelFilterLabels();
    App.views.layout.filter = 'all'; App.views.layout.fitted = false;
    $('rotLabel').textContent = '0 (0°)';

    buildScene();
    App.renderer.fit(m.bounds.min, m.bounds.max);
    refreshSummary();
    refreshChecks();
    showInfo();
    App.plot.redraw();
  }

  function buildScene() {
    const m = App.model, R = App.renderer;
    // plates
    const pi = new Float32Array(m.layers.length * 7), pc = new Float32Array(m.layers.length * 4);
    m.layers.forEach((L, i) => {
      pi.set([0, (L.y0 + L.y1) / 2, 0, 0, L.W, L.T, L.H], 7 * i);
      pc.set([...SV.catColor(L.index).map((v) => 0.55 + 0.45 * v), App.state.plateOpacity], 4 * i);
    });
    R.setInstances('plates', 'box', pi, pc, { transparent: App.state.plateOpacity < 0.999, order: 2, visible: $('showPlates').checked });
    // holes
    const hi = new Float32Array(m.holes.length * 7), hc = new Float32Array(m.holes.length * 4);
    m.holes.forEach((h, i) => {
      const L = m.layers[h.layer];
      const y1 = Math.max(h.y1, L.y0), y2 = Math.min(h.y2, L.y1);   // the part that carves the plate
      hi.set([h.x, (y1 + y2) / 2, h.z, 0, 2 * h.r, Math.max(y2 - y1, 0.01), 2 * h.r], 7 * i);
      hc.set([...HOLE_COL, 1], 4 * i);
    });
    R.setInstances('holes', 'cyl', hi, hc, { order: 1, visible: $('showHoles').checked });
    // crystals (enabled opaque, disabled translucent)
    App.xtalInst = new Array(m.crystals.length);
    const en = m.crystals.filter((c) => c.enabled), dis = m.crystals.filter((c) => !c.enabled);
    const inst = (list, name) => {
      const a = new Float32Array(list.length * 7);
      list.forEach((c, i) => { a.set([c.x, c.y, c.z, c.rot, c.w, c.t, c.h], 7 * i); App.xtalInst[c.index] = [name, i]; });
      return a;
    };
    R.setInstances('crystals', 'box', inst(en, 'crystals'), new Float32Array(en.length * 4), { order: 0, visible: $('showCrystals').checked });
    R.setInstances('disabled', 'box', inst(dis, 'disabled'), new Float32Array(dis.length * 4).fill(0.5).map((v, i) => (i % 4 === 3 ? 0.18 : v)),
      { transparent: true, order: 3, visible: $('showDisabled').checked });
    recolorCrystals();
    buildFov();
    buildAxes();
  }

  function buildFov() {
    const m = App.model, R = App.renderer;
    if (!m || !m.hasImage || m.image.nVox <= 0) { R.removeLines('fov'); return; }
    const I = m.image, a = App.state.rot * I.angle, c = Math.cos(a), s = Math.sin(a);
    const hx = I.nX * I.wX / 2, hy = I.nY * I.wY / 2, hz = I.nZ * I.wZ / 2;
    const P = (x, y, z) => { x += I.shX; y += I.shY; z += I.shZ; return [x * c - y * s, x * s + y * c, z]; };
    const corners = [];
    for (const z of [-hz, hz]) for (const [x, y] of [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) corners.push(P(x, y, z));
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const pos = [], col = [];
    for (const [i, j] of E) { pos.push(...corners[i], ...corners[j]); col.push(0.3, 0.85, 0.95, 0.9, 0.3, 0.85, 0.95, 0.9); }
    // arrow showing the voxel +y direction of this rotation
    const o = P(0, 0, 0), tip = P(0, hy * 0.8, 0);
    pos.push(...o, ...tip); col.push(0.3, 0.85, 0.95, 0.5, 0.3, 0.85, 0.95, 0.5);
    R.setLines('fov', new Float32Array(pos), new Float32Array(col), { visible: $('showFov').checked });
    buildVoxelCloud();
  }

  function buildAxes() {
    const m = App.model, b = m.bounds;
    const L = 0.12 * Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const pos = new Float32Array([0, 0, 0, L, 0, 0, 0, 0, 0, 0, L, 0, 0, 0, 0, 0, 0, L]);
    const col = new Float32Array([1, 0.42, 0.42, 1, 1, 0.42, 0.42, 1, 0.32, 0.81, 0.4, 1, 0.32, 0.81, 0.4, 1, 0.31, 0.63, 1, 1, 0.31, 0.63, 1, 1]);
    App.renderer.setLines('axes', pos, col, { visible: $('showAxes').checked, noClip: true });
  }

  function buildVoxelCloud() {
    const R = App.renderer, m = App.model, vv = App.voxValues;
    if (!m || !vv || !$('showVoxels').checked) { R.removeLines('voxels'); return; }
    const I = m.image, sc = SV.makeScale(vv.values, App.state.log);
    const pos = [], col = [];
    const thr = App.state.log ? sc.lo : sc.hi * 0.02;
    for (let v = 0; v < I.nVox; v++) {
      const x = vv.values[v];
      if (!(x > thr)) continue;
      const p = SV.voxelPos(I, v, App.state.rot).world, c = SV.cmap(sc.t(x), App.state.cmap);
      pos.push(...p); col.push(c[0], c[1], c[2], 0.95);
      if (pos.length > 3 * 500000) break;
    }
    R.setLines('voxels', new Float32Array(pos), new Float32Array(col), { mode: 'points', size: 5 });
  }

  function buildTracks() {
    const R = App.renderer, tr = App.tracks;
    R.removeLines('tracks'); R.removeLines('trackPts');
    if (!tr || !tr.tracks) return;
    const E0 = tr.E0 || 140.5;
    const pos = [], col = [], pp = [], pc = [];
    for (const k of tr.tracks) {
      const v = k.v;
      for (let i = 0; i < v.length; i++) {
        const type = v[i][4];
        if (VTX_COL[type] && type !== 0) { pp.push(v[i][0], v[i][1], v[i][2]); pc.push(...VTX_COL[type], 1); }
        if (i + 1 < v.length) {
          const c = SV.cmap(0.3 + 0.7 * SV.clamp(v[i][3] / E0, 0, 1), 'inferno');
          pos.push(v[i][0], v[i][1], v[i][2], v[i + 1][0], v[i + 1][1], v[i + 1][2]);
          col.push(...c, 0.75, ...c, 0.75);
        }
      }
    }
    const vis = $('showTracks').checked;
    R.setLines('tracks', new Float32Array(pos), new Float32Array(col), { visible: vis });
    R.setLines('trackPts', new Float32Array(pp), new Float32Array(pc), { mode: 'points', size: 5, visible: vis });
  }

  // ================================================================ colouring
  function crystalValue(c, mode) {
    switch (mode) {
      case 'muTot': return c.muTot;
      case 'peFrac': return c.muTot > 0 ? c.muPE / c.muTot : 0;
      case 'eres': return c.eres;
      case 'rot': return c.rot * 180 / Math.PI;
      case 'y': return c.yraw;
      case 'index': return c.index;
      case 'projection': return App.projValues ? App.projValues[c.index] : NaN;
      case 'detSum': return App.detSumValues ? App.detSumValues[c.index] : NaN;
      default: return 0;
    }
  }
  function recolorCrystals() {
    const m = App.model;
    if (!m) return;
    const mode = App.state.colorBy, n = m.crystals.length;
    const cols = new Float32Array(4 * n);
    let legend = '';
    const label = $('colorBy').selectedOptions[0] ? $('colorBy').selectedOptions[0].textContent : mode;
    if (mode === 'flag') {
      const flags = [...new Set(m.crystals.map((c) => Math.round(c.flag)))].sort((a, b) => a - b);
      m.crystals.forEach((c) => cols.set([...SV.catColor(flags.indexOf(Math.round(c.flag))), 1], 4 * c.index));
      legend = `<div>${SV.esc(label)}</div>` + flags.slice(0, 10).map((f, i) =>
        `<span style="color:${SV.rgbCss(SV.catColor(i))}">■</span> ${f}`).join('&nbsp; ');
    } else {
      const vals = m.crystals.filter((c) => c.enabled).map((c) => crystalValue(c, mode));
      const have = vals.some((v) => isFinite(v));
      const sc = SV.makeScale(vals, App.state.log);
      m.crystals.forEach((c) => {
        const v = crystalValue(c, mode);
        cols.set(isFinite(v) ? [...SV.cmap(sc.t(v), App.state.cmap), 1] : [0.35, 0.38, 0.42, 1], 4 * c.index);
      });
      const stops = Array.from({ length: 9 }, (_, i) => SV.rgbCss(SV.cmap(i / 8, App.state.cmap))).join(',');
      legend = have
        ? `<div>${SV.esc(label)}${sc.log ? ' (log)' : ''}</div><div class="bar" style="background:linear-gradient(90deg,${stops})"></div>` +
          `<div class="lab"><span>${SV.fmt(sc.lo)}</span><span>${SV.fmt(sc.hi)}</span></div>`
        : `<div>${SV.esc(label)}: –</div>`;
    }
    App.crystalColors = cols;
    const en = m.crystals.filter((c) => c.enabled);
    const ec = new Float32Array(en.length * 4);
    en.forEach((c, i) => ec.set(cols.subarray(4 * c.index, 4 * c.index + 4), 4 * i));
    App.renderer.setColors('crystals', ec);
    $('legend').innerHTML = legend;
    App.plot.redraw();
  }

  // ================================================================ picking
  function pick3D(e) {
    const m = App.model;
    if (!m) return null;
    const R = App.renderer, { o, d } = R.ray(e.clientX, e.clientY);
    const clip = R.clip;
    const visibleAt = (tt) => {
      const p = [o[0] + d[0] * tt, o[1] + d[1] * tt, o[2] + d[2] * tt];
      return p[0] * clip[0] + p[1] * clip[1] + p[2] * clip[2] + clip[3] <= 0;
    };
    const slab = (oo, dd, mn, mx) => {
      let t0 = -Infinity, t1 = Infinity;
      for (let a = 0; a < 3; a++) {
        if (Math.abs(dd[a]) < 1e-12) { if (oo[a] < mn[a] || oo[a] > mx[a]) return null; continue; }
        let u = (mn[a] - oo[a]) / dd[a], v = (mx[a] - oo[a]) / dd[a];
        if (u > v) [u, v] = [v, u];
        t0 = Math.max(t0, u); t1 = Math.min(t1, v);
      }
      return t0 <= t1 && t1 >= 0 ? [t0, t1] : null;
    };
    let best = null;
    const consider = (tt, sel) => { if (tt >= 0 && visibleAt(tt) && (!best || tt < best.t)) best = { t: tt, sel }; };
    // crystals
    for (const c of m.crystals) {
      const batch = c.enabled ? 'crystals' : 'disabled';
      if (!R.isVisible(batch)) continue;
      const ca = Math.cos(c.rot), sa = Math.sin(c.rot);
      const dx = o[0] - c.x, dz = o[2] - c.z;
      const lo = [dx * ca + dz * sa, o[1] - c.y, -dx * sa + dz * ca], ld = [d[0] * ca + d[2] * sa, d[1], -d[0] * sa + d[2] * ca];
      const h = slab(lo, ld, [-c.w / 2, -c.t / 2, -c.h / 2], [c.w / 2, c.t / 2, c.h / 2]);
      if (h) consider(Math.max(h[0], 0), { kind: 'crystal', index: c.index });
    }
    // holes (cylinders along y)
    if (R.isVisible('holes'))
      for (const h of m.holes) {
        const L = m.layers[h.layer], y1 = Math.max(h.y1, L.y0), y2 = Math.min(h.y2, L.y1);
        const ox = o[0] - h.x, oz = o[2] - h.z, A = d[0] * d[0] + d[2] * d[2], B = ox * d[0] + oz * d[2], C = ox * ox + oz * oz - h.r * h.r;
        const cands = [];
        if (A > 1e-12) { const D = B * B - A * C; if (D >= 0) { const s = Math.sqrt(D); cands.push((-B - s) / A, (-B + s) / A); } }
        if (Math.abs(d[1]) > 1e-12) cands.push((y1 - o[1]) / d[1], (y2 - o[1]) / d[1]);
        for (const tt of cands) {
          if (!(tt >= 0)) continue;
          const y = o[1] + d[1] * tt, px = ox + d[0] * tt, pz = oz + d[2] * tt;
          if (y >= y1 - 1e-6 && y <= y2 + 1e-6 && px * px + pz * pz <= h.r * h.r * 1.0001) consider(tt, { kind: 'hole', index: h.index });
        }
      }
    if (best) return best.sel;
    // plates only when nothing else was hit
    if (R.isVisible('plates'))
      for (const L of m.layers) {
        const h = slab(o, d, [-L.W / 2, L.y0, -L.H / 2], [L.W / 2, L.y1, L.H / 2]);
        if (h) consider(Math.max(h[0], 0), { kind: 'layer', index: L.index });
      }
    return best ? best.sel : null;
  }

  App.select = function (sel) {
    App.sel = sel;
    const R = App.renderer;
    if (sel && sel.kind === 'crystal' && App.xtalInst[sel.index]) R.setSelection(...App.xtalInst[sel.index]);
    else if (sel && sel.kind === 'hole') R.setSelection('holes', sel.index);
    else if (sel && sel.kind === 'layer') R.setSelection('plates', sel.index);
    else R.setSelection(null, -1);
    if (sel && sel.kind === 'crystal') {
      $('detIdx').value = sel.index;
      const sm = curSysmat();
      if (sm && (!App.voxValues || App.voxValues.kind === 'row')) showRow().catch(fail);
    }
    if (sel && sel.kind === 'voxel') {
      $('voxIdx').value = sel.index;
      if (curSysmat()) showCol().catch(fail);
    }
    showInfo();
    App.plot.redraw();
  };

  function selCentre(sel) {
    const m = App.model;
    if (!sel || !m) return null;
    if (sel.kind === 'crystal') { const c = m.crystals[sel.index]; return [c.x, c.y, c.z]; }
    if (sel.kind === 'hole') { const h = m.holes[sel.index]; return [h.x, (h.y1 + h.y2) / 2, h.z]; }
    if (sel.kind === 'layer') { const L = m.layers[sel.index]; return [0, (L.y0 + L.y1) / 2, 0]; }
    if (sel.kind === 'voxel') return SV.voxelPos(m.image, sel.index, App.state.rot).world;
    return null;
  }

  // ================================================================ info panel
  const row = (k, v) => `<tr><td>${SV.esc(k)}</td><td>${v}</td></tr>`;
  const v3 = (a) => a.map((x) => SV.fmt(x, 5)).join(', ');
  App.describeShort = function (sel) {
    const m = App.model;
    if (sel.kind === 'crystal') { const c = m.crystals[sel.index]; return `${t('iCrystal')} #${c.index} (${SV.fmt(c.x)}, ${SV.fmt(c.y)}, ${SV.fmt(c.z)})`; }
    if (sel.kind === 'hole') { const h = m.holes[sel.index]; return `${t('iHole')} #${h.index} · L${h.layer} · r=${SV.fmt(h.r)} mm`; }
    if (sel.kind === 'layer') return `${t('iLayer')} ${sel.index}`;
    if (sel.kind === 'voxel') return `${t('iVoxel')} #${sel.index}`;
    return '';
  };
  function showInfo() {
    const box = $('info'), m = App.model, sel = App.sel;
    if (!m || !sel) { box.className = 'muted'; box.textContent = t('infoHint'); return; }
    box.className = '';
    let h = '';
    if (sel.kind === 'crystal') {
      const c = m.crystals[sel.index];
      h += `<div class="title">${t('iCrystal')} #${c.index}</div><table class="kv">`;
      h += row(t('iCentre'), v3([c.x, c.y, c.z]) + ' mm') + row(t('iYraw'), SV.fmt(c.yraw, 5) + ' mm');
      h += row(t('iSize'), `${SV.fmt(c.w)} × ${SV.fmt(c.t)} × ${SV.fmt(c.h)} mm`);
      h += row(t('iRot'), SV.fmt(c.rot * 180 / Math.PI, 4) + '°');
      h += row(t('iMu'), `${SV.fmt(c.muTot)} / ${SV.fmt(c.muPE)} / ${SV.fmt(c.muC)} mm⁻¹`);
      h += row(t('iPeFrac'), c.muTot > 0 ? SV.fmt(c.muPE / c.muTot, 3) : '–');
      h += row(t('iEres'), SV.fmt(c.eres * 100, 3) + ' %') + row(t('iFlag'), SV.fmt(c.flag)) + row(t('iEnabled'), c.enabled ? t('yes') : t('no'));
      if (App.projValues) h += row(t('iProj'), SV.fmt(App.projValues[c.index], 5));
      if (App.detSumValues) h += row(t('iDetSum'), SV.fmt(App.detSumValues[c.index], 5));
      h += '</table>';
    } else if (sel.kind === 'hole') {
      const o = m.holes[sel.index];
      h += `<div class="title">${t('iHole')} #${o.index}</div><table class="kv">`;
      h += row(t('iLayerOf'), o.layer) + row(t('iLocal'), o.local);
      h += row('x, z', `${SV.fmt(o.x, 5)}, ${SV.fmt(o.z, 5)} mm`) + row(t('iRadius'), SV.fmt(o.r, 5) + ' mm');
      h += row(t('iYrange'), `${SV.fmt(o.y1, 5)} … ${SV.fmt(o.y2, 5)} mm`) + row(t('iYraw'), `${SV.fmt(o.y1raw, 5)} … ${SV.fmt(o.y2raw, 5)}`);
      h += row(t('iMu'), `${SV.fmt(o.muTot)} / ${SV.fmt(o.muPE)} / ${SV.fmt(o.muC)} mm⁻¹`) + row(t('iFlag'), SV.fmt(o.flag));
      h += '</table>';
    } else if (sel.kind === 'layer') {
      const L = m.layers[sel.index];
      h += `<div class="title">${t('iLayer')} ${L.index}</div><table class="kv">`;
      h += row(t('iPlate'), `${SV.fmt(L.W)} × ${SV.fmt(L.T)} × ${SV.fmt(L.H)} mm`) + row(t('iYrange'), `${SV.fmt(L.y0, 5)} … ${SV.fmt(L.y1, 5)} mm`);
      h += row(t('iOffset'), SV.fmt(L.offset) + ' mm') + row(t('iHolesN'), L.nHoles);
      h += row(t('iMu'), `${SV.fmt(L.muTot)} / ${SV.fmt(L.muPE)} / ${SV.fmt(L.muC)} mm⁻¹`);
      if (L.muTot > 0) h += row('1/μ', SV.fmt(1 / L.muTot, 4) + ' mm');
      h += '</table>';
    } else if (sel.kind === 'voxel') {
      const p = SV.voxelPos(m.image, sel.index, App.state.rot);
      h += `<div class="title">${t('iVoxel')} #${sel.index}</div><table class="kv">`;
      h += row(t('iIndex3'), `${p.ix}, ${p.iy}, ${p.iz}`) + row(t('iLocalPos'), v3(p.local) + ' mm');
      h += row(t('iWorldPos', App.state.rot), v3(p.world) + ' mm');
      if (App.voxValues) h += row(t('iValue'), SV.fmt(App.voxValues.values[sel.index], 5));
      h += '</table>';
    }
    box.innerHTML = h;
  }

  // ================================================================ side panels
  function refreshSummary() {
    const m = App.model, tb = $('summaryTable');
    if (!m) { tb.innerHTML = ''; return; }
    const I = m.image, P = m.physics, en = m.crystals.filter((c) => c.enabled).length;
    let h = row(t('sLayers'), m.layers.length) + row(t('sHoles'), m.holes.length) + row(t('sCrystals'), `${en} / ${m.crystals.length}`);
    if (m.hasImage) {
      h += row(t('sFov'), `${I.nX} × ${I.nY} × ${I.nZ}`) + row(t('sVoxel'), `${SV.fmt(I.wX)} × ${SV.fmt(I.wY)} × ${SV.fmt(I.wZ)} mm`);
      h += row(t('sRot'), `${I.nRot} × ${SV.fmt(I.angle * 180 / Math.PI)}°`) + row(t('sFov2col'), SV.fmt(m.fov2col) + ' mm');
      h += row(t('sSysmat'), SV.bytes(m.crystals.length * I.nVox * 4));
    }
    if (m.hasPhysics) {
      h += row(t('sE0'), SV.fmt(P.E0) + ' keV');
      h += row(t('sWindow'), P.sameWindow > 0.5 ? `${SV.fmt(P.lo)} – ${SV.fmt(P.hi)} keV` : t('perCrystal'));
    }
    tb.innerHTML = h;
  }
  function refreshChecks() {
    const ul = $('checkList');
    ul.innerHTML = '';
    if (!App.model) return;
    const items = SV.checkModel(App.model);
    const sm = curSysmat(true);
    if (sm && !sm.ok) items.unshift({ level: 'error', key: 'smBad', args: [SV.bytes(sm.src.size)] });
    for (const it of items) {
      const li = document.createElement('li');
      li.className = it.level;
      li.textContent = t(it.key, ...(it.args || []));
      ul.appendChild(li);
    }
  }
  function refreshChips() {
    const box = $('fileChips');
    const chip = (label, on) => `<span class="chip ${on ? 'on' : ''}">${SV.esc(label)}</span>`;
    box.innerHTML = chip('Collimator', !!App.f32.col) + chip('Detector', !!App.f32.det) + chip('Image', !!App.f32.img) + chip('Physics', !!App.f32.phy) +
      chip(`sysmat ×${App.sysmats.length}`, App.sysmats.length > 0) + chip('spectrum', !!App.spectrum) + chip('tracks', !!App.tracks) + chip('summary', !!App.summary);
  }

  function curSysmat(any) {
    const e = App.sysmats[App.cur];
    if (!e || !App.model) return null;
    if (!e.sm || e.sm.nDet !== App.model.crystals.length || e.sm.nVox !== App.model.image.nVox)
      e.sm = new SV.Sysmat(e.src, App.model.crystals.length, App.model.image.nVox);
    return any ? e.sm : (e.sm.ok ? e.sm : null);
  }
  function refreshSysmat() {
    const sel = $('sysmatSel');
    sel.innerHTML = '';
    App.sysmats.forEach((s, i) => {
      const o = document.createElement('option');
      o.value = i; o.textContent = `${s.src.name} (${SV.bytes(s.src.size)})`;
      sel.appendChild(o);
    });
    sel.value = App.cur;
    const sm = curSysmat(true), info = $('sysmatInfo');
    if (!App.sysmats.length) info.textContent = '';
    else if (!App.model) info.textContent = t('stNeedGeom');
    else info.textContent = sm.ok ? t('smOk', sm.nRot, sm.nDet, sm.nVox) : t('smBad', SV.bytes(sm.src.size));
    for (const id of ['rowBtn', 'colBtn', 'sensBtn']) $(id).disabled = !(sm && sm.ok);
    refreshChecks();
  }

  function rotForSysmat(sm) { return Math.min(App.state.rot, sm.nRot - 1); }
  async function showRow() {
    const sm = curSysmat();
    if (!sm) return status(t('stNoSysmat'), true);
    const d = SV.clamp(parseInt($('detIdx').value, 10) || 0, 0, sm.nDet - 1), r = rotForSysmat(sm);
    status(t('stRow', d));
    App.voxValues = { values: await sm.row(r, d), label: t('vRow', d, r), kind: 'row' };
    $('showVoxels').checked = true;
    buildVoxelCloud();
    setTab('image');
    status(t('stDone'));
  }
  async function showCol() {
    const sm = curSysmat();
    if (!sm) return status(t('stNoSysmat'), true);
    const v = SV.clamp(parseInt($('voxIdx').value, 10) || 0, 0, sm.nVox - 1), r = rotForSysmat(sm);
    status(t('stCol', v));
    App.projValues = await sm.column(r, v);
    App.state.colorBy = 'projection'; $('colorBy').value = 'projection';
    if (!App.sel || App.sel.kind !== 'voxel' || App.sel.index !== v) { App.sel = { kind: 'voxel', index: v }; showInfo(); }
    recolorCrystals();
    status(t('stDone'));
  }
  async function showSens() {
    const sm = curSysmat();
    if (!sm) return status(t('stNoSysmat'), true);
    const r = rotForSysmat(sm), bar = $('progress');
    bar.hidden = false;
    status(t('stSens', r));
    try {
      const res = await sm.sums(r, (f) => { bar.firstChild.style.width = `${(100 * f).toFixed(1)}%`; });
      App.voxValues = { values: res.vox, label: t('vSens', r), kind: 'sens' };
      App.detSumValues = res.det;
    } finally { bar.hidden = true; }
    $('showVoxels').checked = true;
    buildVoxelCloud();
    setTab('image');
    status(t('stDone'));
  }

  App.windows = function () {
    const s = App.summary, m = App.model;
    const E0 = s ? s.energy_keV : (m && m.physics.E0) || 0;
    const res = m && m.crystals.length ? m.crystals.reduce((a, c) => a + c.eres, 0) / m.crystals.length : 0.1;
    const out = [];
    if (s && s.windows) for (const w of s.windows) out.push(w.per_crystal ? { lo: (1 - res / 2) * E0, hi: (1 + res / 2) * E0 } : { lo: w.lo, hi: w.hi });
    else if (m && m.hasPhysics && E0 > 0) out.push(m.physics.sameWindow > 0.5 ? { lo: m.physics.lo, hi: m.physics.hi } : { lo: (1 - res / 2) * E0, hi: (1 + res / 2) * E0 });
    return out;
  };

  function refreshMC() {
    const s = App.summary, tb = $('mcTable');
    let h = '';
    if (s) {
      h += row(t('mE0'), SV.fmt(s.energy_keV) + ' keV') + row(t('mPhysics'), `${s.physics}${s.rayleigh ? ' +Rayleigh' : ''}${s.fluorescence ? ' +K-fluo' : ''}`);
      h += row(t('mHist'), SV.fmt(s.histories_per_voxel)) + row(t('mTotal'), SV.fmt(s.total_histories));
      (s.mean_sensitivity || []).forEach((w) => {
        h += row(t('mSens', w.window), SV.fmt(w.total, 4));
        if (w.primary !== undefined && w.total > 0)
          h += row(t('mFrac'), [w.primary, w.colscatter, w.xtalscatter].map((x) => (100 * x / w.total).toFixed(1) + '%').join(' / '));
      });
      h += row(t('mBackend'), s.backend + (s.devices ? ` ×${s.devices.length}` : '')) + row(t('mTime'), SV.fmt(s.wall_seconds, 4) + ' s');
    }
    if (App.tracks) h += row(t('mTracks'), `${App.tracks.tracks.length} (voxel ${App.tracks.voxel})`);
    tb.innerHTML = h;
    App.plot.redraw();
  }

  // ================================================================ tabs
  function setTab(name) {
    document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
    $('ctlLayout').hidden = name !== 'layout';
    $('ctlImage').hidden = name !== 'image';
    App.plot.setView(App.views[name]);
    SV.storage.set('tab', name);
  }

  function setClip() {
    const ax = $('clipAxis').value;
    if (!ax || !App.model) return App.renderer.setClip(null);
    const a = { x: 0, y: 1, z: 2 }[ax], b = App.model.bounds;
    const pos = b.min[a] + (b.max[a] - b.min[a]) * parseFloat($('clipPos').value);
    const n = [0, 0, 0];
    n[a] = $('clipFlip').checked ? -1 : 1;
    App.renderer.setClip([n[0], n[1], n[2], -pos * n[a]]);
  }

  // ================================================================ init
  function init() {
    SV.applyI18n();
    $('langBtn').textContent = SV.lang() === 'zh' ? 'EN' : '中文';
    try {
      App.renderer = new SV.Renderer($('gl'));
    } catch (e) {
      $('hint3d').textContent = 'WebGL2 is required: ' + e.message;
      App.renderer = { setInstances() {}, setColors() {}, setLines() {}, removeLines() {}, setVisible() {}, isVisible() { return false; }, setSelection() {}, setClip() {}, fit() {}, view() {}, clip: [0, 0, 0, -1], cam: {}, removeBatch() {}, setTransparent() {} };
    }
    App.plot = new SV.Plot2D($('c2d'), $('tip2d'));
    App.views = { layout: new SV.LayoutView(App), image: new SV.ImageView(App), spectrum: new SV.SpectrumView(App) };
    setTab(SV.storage.get('tab', 'layout'));

    const cm = $('cmap');
    SV.colormapNames.forEach((n) => { const o = document.createElement('option'); o.value = n; o.textContent = n; cm.appendChild(o); });
    cm.value = App.state.cmap;

    // files
    $('fileInput').addEventListener('change', (e) => { loadSources([...e.target.files].map((f) => new SV.LocalSource(f))).catch(fail); e.target.value = ''; });
    $('dirInput').addEventListener('change', (e) => { loadSources([...e.target.files].map((f) => new SV.LocalSource(f))).catch(fail); e.target.value = ''; });
    $('demoBtn').addEventListener('click', demo);
    let dragDepth = 0;
    window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; $('dropOverlay').hidden = false; });
    window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('dropOverlay').hidden = true; } });
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {
      e.preventDefault(); dragDepth = 0; $('dropOverlay').hidden = true;
      loadSources([...e.dataTransfer.files].map((f) => new SV.LocalSource(f))).catch(fail);
    });
    $('langBtn').addEventListener('click', () => {
      SV.setLang(SV.lang() === 'zh' ? 'en' : 'zh');
      $('langBtn').textContent = SV.lang() === 'zh' ? 'EN' : '中文';
      SV.applyI18n();
      refreshSummary(); refreshChecks(); refreshSysmat(); refreshMC(); showInfo(); recolorCrystals();
      if (App.model) { const v = App.views.layout.filter; buildModelFilterLabels(); App.views.layout.filter = v; $('layoutFilter').value = v; }
      App.plot.redraw();
    });

    // display
    const vis = { showPlates: ['plates'], showHoles: ['holes'], showCrystals: ['crystals'], showFov: ['fov'], showAxes: ['axes'], showTracks: ['tracks', 'trackPts'], showDisabled: ['disabled'] };
    for (const [id, names] of Object.entries(vis)) $(id).addEventListener('change', (e) => names.forEach((n) => App.renderer.setVisible(n, e.target.checked)));
    $('showVoxels').addEventListener('change', buildVoxelCloud);
    $('plateOpacity').addEventListener('input', (e) => {
      App.state.plateOpacity = parseFloat(e.target.value);
      if (App.model) {
        const pc = new Float32Array(App.model.layers.length * 4);
        App.model.layers.forEach((L, i) => pc.set([...SV.catColor(L.index).map((v) => 0.55 + 0.45 * v), App.state.plateOpacity], 4 * i));
        App.renderer.setColors('plates', pc);
        App.renderer.setTransparent('plates', App.state.plateOpacity < 0.999);
      }
    });
    $('colorBy').addEventListener('change', (e) => { App.state.colorBy = e.target.value; recolorCrystals(); });
    $('cmap').addEventListener('change', (e) => { App.state.cmap = e.target.value; SV.storage.set('cmap', e.target.value); recolorCrystals(); buildVoxelCloud(); });
    $('logScale').addEventListener('change', (e) => { App.state.log = e.target.checked; recolorCrystals(); buildVoxelCloud(); });
    for (const id of ['clipAxis', 'clipPos', 'clipFlip']) $(id).addEventListener('input', setClip);
    $('rotIndex').addEventListener('input', (e) => {
      App.state.rot = parseInt(e.target.value, 10);
      const I = App.model ? App.model.image : null;
      $('rotLabel').textContent = I ? `${App.state.rot} (${SV.fmt(App.state.rot * I.angle * 180 / Math.PI, 4)}°)` : '0';
      if (App.model) { buildFov(); showInfo(); }
    });
    document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => App.renderer.view(b.dataset.view)));
    $('fitBtn').addEventListener('click', () => App.model && App.renderer.fit(App.model.bounds.min, App.model.bounds.max));
    $('orthoCam').addEventListener('change', (e) => { App.renderer.cam.ortho = e.target.checked; App.renderer.dirty = true; });

    // sysmat
    $('sysmatSel').addEventListener('change', (e) => { App.cur = parseInt(e.target.value, 10); refreshSysmat(); });
    $('rowBtn').addEventListener('click', () => showRow().catch(fail));
    $('colBtn').addEventListener('click', () => showCol().catch(fail));
    $('sensBtn').addEventListener('click', () => showSens().catch(fail));
    $('detIdx').addEventListener('change', (e) => App.model && App.select({ kind: 'crystal', index: SV.clamp(parseInt(e.target.value, 10) || 0, 0, App.model.crystals.length - 1) }));

    // MC
    $('specMode').addEventListener('change', (e) => { App.state.specMode = e.target.value; App.plot.redraw(); });
    $('specLog').addEventListener('change', (e) => { App.state.specLog = e.target.checked; App.plot.redraw(); });

    // tabs
    document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));
    $('layoutFilter').addEventListener('change', (e) => { App.views.layout.filter = e.target.value; App.plot.redraw(); });
    $('sliceIdx').addEventListener('input', (e) => { App.views.image.slice = parseInt(e.target.value, 10); App.plot.redraw(); });

    // 3D picking / hover
    const R = App.renderer;
    R.onClick = (e) => { const s = pick3D(e); App.select(s); };
    $('gl').addEventListener('dblclick', () => { const c = selCentre(App.sel); if (c) { R.cam.target = c; R.dirty = true; } });
    let hoverT = 0;
    R.onHover = (e) => {
      const now = performance.now();
      if (now - hoverT < 40) return;
      hoverT = now;
      const s = pick3D(e), tip = $('tip3d');
      if (s) { tip.innerHTML = SV.esc(App.describeShort(s)); tip.hidden = false; tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 10}px`; } else tip.hidden = true;
    };
    $('gl').addEventListener('pointerleave', () => { $('tip3d').hidden = true; });

    refreshChips();
    // served by serve.py? list the data directory and load what is there
    SV.serverFiles().then((info) => {
      if (!info) return;
      $('serverBox').hidden = false;
      $('serverRoot').textContent = info.root;
      const list = $('serverList');
      for (const f of info.files) {
        const b = document.createElement('button');
        b.innerHTML = `<span>${SV.esc(f.path)}</span><span>${SV.bytes(f.size)}</span>`;
        b.addEventListener('click', () => loadSources([new SV.RemoteSource(f.path, f.size)]).catch(fail));
        list.appendChild(b);
      }
      const auto = info.files.filter((f) => !f.path.includes('/') && SV.classify(f.path));
      status(t('stServer', info.root));
      if (auto.length) loadSources(auto.map((f) => new SV.RemoteSource(f.path, f.size))).catch(fail);
    });
  }

  function buildModelFilterLabels() {
    const lf = $('layoutFilter'), m = App.model;
    lf.innerHTML = '';
    const opt = (v, txt) => { const o = document.createElement('option'); o.value = v; o.textContent = txt; lf.appendChild(o); };
    opt('all', t('layoutAll')); opt('crystals', t('layoutCrystals'));
    m.layers.forEach((L) => opt(String(L.index), t('layoutLayer', L.index)));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
