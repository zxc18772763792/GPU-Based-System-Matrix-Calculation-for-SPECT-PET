// params.js - parse Params_Collimator / Detector / Image / Physics (.dat, float32) into a
// model of geometric elements, run consistency checks, and build a demo geometry.
(function () {
  'use strict';
  const SV = (window.SV = window.SV || {});

  const MAX_LAYERS = 10;

  // f32: {col, det, img, phy} Float32Arrays (img/phy may be missing)
  SV.parseParams = function (f32) {
    const col = f32.col, det = f32.det;
    const img = f32.img || new Float32Array(100), phy = f32.phy || new Float32Array(100);
    const issues = [];
    const fov2col = img[11] || 0;
    const m = { layers: [], holes: [], crystals: [], issues, fov2col, hasImage: !!f32.img, hasPhysics: !!f32.phy };

    // ---------------------------------------------------------------- image / physics
    m.image = {
      nX: Math.floor(img[0] + 0.001), nY: Math.floor(img[1] + 0.001), nZ: Math.floor(img[2] + 0.001),
      wX: img[3], wY: img[4], wZ: img[5], nRot: Math.max(1, Math.floor(img[6] + 0.001)), angle: img[7],
      shX: img[8], shY: img[9], shZ: img[10], fov2col,
    };
    m.image.nVox = m.image.nX * m.image.nY * m.image.nZ;
    m.physics = {
      useCompton: phy[0], savePE: phy[1], saveCompton: phy[2], saveSum: phy[3], sameWindow: phy[4],
      lo: phy[5], hi: phy[6], E0: phy[7], calcCrystalGeo: phy[8], calcCollimatorGeo: phy[9],
    };

    // ---------------------------------------------------------------- collimator
    if (col) {
      const nL = Math.floor(col[0] + 0.001);
      if (nL < 0 || nL > MAX_LAYERS) issues.push({ level: 'error', key: 'chkLayers', args: [nL] });
      let hs = 0;
      for (let l = 0; l < Math.min(Math.max(nL, 0), MAX_LAYERS); l++) {
        const b = (l + 1) * 10;
        const L = {
          index: l, nHoles: Math.floor(col[b] + 0.001), W: col[b + 1], T: col[b + 2], H: col[b + 3], offset: col[b + 4],
          muTot: col[b + 5], muPE: col[b + 6], muC: col[b + 7], holeStart: hs,
        };
        L.y0 = fov2col + L.offset - L.T / 2;
        L.y1 = fov2col + L.offset + L.T / 2;
        m.layers.push(L);
        hs += Math.max(0, L.nHoles);
      }
      const need = 100 + hs * 9;
      if (need > col.length) issues.push({ level: 'error', key: 'chkHoleFile', args: [hs, col.length] });
      else if (need > 80000) issues.push({ level: 'warn', key: 'chkHole80k', args: [hs] });
      for (const L of m.layers) {
        for (let k = 0; k < L.nHoles; k++) {
          const i = L.holeStart + k, o = 100 + 9 * i;
          if (o + 8 >= col.length) break;
          let y1 = col[o + 1] + fov2col, y2 = col[o + 2] + fov2col;
          m.holes.push({
            index: i, layer: L.index, local: k, x: col[o], y1raw: col[o + 1], y2raw: col[o + 2], y1: Math.min(y1, y2), y2: Math.max(y1, y2),
            z: col[o + 3], r: Math.abs(col[o + 4]), muTot: col[o + 5], muPE: col[o + 6], muC: col[o + 7], flag: col[o + 8],
          });
        }
      }
    }

    // ---------------------------------------------------------------- detector
    if (det) {
      const n = Math.floor(det[0] + 0.001);
      if (12 * n + 1 > det.length) issues.push({ level: 'error', key: 'chkDetFile', args: [n, det.length] });
      for (let i = 0; i < n && 12 * i + 12 < det.length; i++) {
        const q = (k) => det[12 * i + k];
        m.crystals.push({
          index: i, x: q(1), yraw: q(2), y: q(2) + fov2col, z: q(3), w: q(4), t: q(5), h: q(6),
          muTot: q(7), muPE: q(8), muC: q(9), eres: q(10), rot: q(11), flag: q(12), enabled: q(7) > 0.01,
        });
      }
    }
    m.bounds = SV.modelBounds(m);
    return m;
  };

  SV.crystalExtent = function (c) {
    const ca = Math.abs(Math.cos(c.rot)), sa = Math.abs(Math.sin(c.rot));
    return [ca * c.w / 2 + sa * c.h / 2, c.t / 2, sa * c.w / 2 + ca * c.h / 2];
  };

  SV.modelBounds = function (m) {
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    const add = (c, e) => { for (let a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], c[a] - e[a]); mx[a] = Math.max(mx[a], c[a] + e[a]); } };
    for (const L of m.layers) add([0, (L.y0 + L.y1) / 2, 0], [L.W / 2, L.T / 2, L.H / 2]);
    for (const c of m.crystals) add([c.x, c.y, c.z], SV.crystalExtent(c));
    const I = m.image;
    if (I.nVox > 0) {
      const r = Math.hypot(I.nX * I.wX / 2 + Math.abs(I.shX), I.nY * I.wY / 2 + Math.abs(I.shY));
      add([0, 0, I.shZ], [r, r, I.nZ * I.wZ / 2]);
    }
    if (!isFinite(mn[0])) return { min: [-50, -50, -50], max: [50, 50, 50] };
    return { min: mn, max: mx };
  };

  // voxel centre (unrotated) and rotated world position for rotation index rot
  SV.voxelPos = function (I, v, rot) {
    const nxy = I.nX * I.nY, iz = Math.floor(v / nxy), rem = v - iz * nxy, iy = Math.floor(rem / I.nX), ix = rem - iy * I.nX;
    const x = (ix - I.nX / 2 + 0.5) * I.wX + I.shX, y = (iy - I.nY / 2 + 0.5) * I.wY + I.shY, z = (iz - I.nZ / 2 + 0.5) * I.wZ + I.shZ;
    const a = (rot || 0) * I.angle, c = Math.cos(a), s = Math.sin(a);
    return { ix, iy, iz, local: [x, y, z], world: [x * c - y * s, x * s + y * c, z] };
  };

  // ---------------------------------------------------------------- checks
  SV.checkModel = function (m) {
    const out = m.issues.slice();
    const push = (level, key, ...args) => out.push({ level, key, args });
    const I = m.image, P = m.physics;
    if (!m.hasImage) push('info', 'chkNoImage');
    else if (I.nVox <= 0) push('error', 'chkNoVoxels');
    if (!m.crystals.length) push('error', 'chkNoCrystals');

    // holes inside their plate / spanning the plate
    let outside = 0, blind = 0, overlaps = 0;
    for (const h of m.holes) {
      const L = m.layers[h.layer];
      if (Math.abs(h.x) + h.r > L.W / 2 + 1e-4 || Math.abs(h.z) + h.r > L.H / 2 + 1e-4) outside++;
      if (h.y1 > L.y0 + 1e-3 || h.y2 < L.y1 - 1e-3) blind++;
    }
    if (outside) push('warn', 'chkHoleOutside', outside);
    if (blind) push('info', 'chkHoleBlind', blind);
    // hole overlaps (grid per layer)
    for (const L of m.layers) {
      const hs = m.holes.filter((h) => h.layer === L.index);
      const rmax = hs.reduce((a, h) => Math.max(a, h.r), 0);
      if (!hs.length || rmax <= 0) continue;
      const cell = 2 * rmax, grid = new Map();
      for (const h of hs) {
        const k = `${Math.floor(h.x / cell)},${Math.floor(h.z / cell)}`;
        (grid.get(k) || grid.set(k, []).get(k)).push(h);
      }
      for (const h of hs) {
        const gx = Math.floor(h.x / cell), gz = Math.floor(h.z / cell);
        for (let dx = -1; dx <= 1; dx++)
          for (let dz = -1; dz <= 1; dz++)
            for (const o of grid.get(`${gx + dx},${gz + dz}`) || [])
              if (o.index > h.index && Math.hypot(o.x - h.x, o.z - h.z) < o.r + h.r - 1e-4 &&
                  Math.min(o.y2, h.y2) > Math.max(o.y1, h.y1)) overlaps++;
      }
    }
    if (overlaps) push('warn', 'chkHoleOverlap', overlaps);

    // crystals
    const en = m.crystals.filter((c) => c.enabled);
    const dis = m.crystals.length - en.length;
    if (dis) push('info', 'chkDisabled', dis);
    let badMu = 0, noRes = 0;
    for (const c of en) {
      if (c.muPE + c.muC > c.muTot * 1.001) badMu++;
      if (!(c.eres > 0)) noRes++;
      if (!(c.w > 0 && c.t > 0 && c.h > 0)) push('error', 'chkXtalSize', c.index);
    }
    if (badMu) push('warn', 'chkMuSum', badMu);
    if (noRes && P.sameWindow < 0.5) push('warn', 'chkNoRes', noRes);
    // crystal overlaps: uniform grid on AABBs, then exact test in the x-z plane (separating axes)
    if (en.length) {
      const ext = en.map(SV.crystalExtent);
      const cell = Math.max(0.5, en.reduce((m, c) => Math.min(m, Math.max(c.w, c.t, c.h)), Infinity));   // no spread: > 1e5 crystals
      const grid = new Map();
      en.forEach((c, i) => {
        const e = ext[i];
        for (let gx = Math.floor((c.x - e[0]) / cell); gx <= Math.floor((c.x + e[0]) / cell); gx++)
          for (let gy = Math.floor((c.y - e[1]) / cell); gy <= Math.floor((c.y + e[1]) / cell); gy++)
            for (let gz = Math.floor((c.z - e[2]) / cell); gz <= Math.floor((c.z + e[2]) / cell); gz++) {
              const k = `${gx},${gy},${gz}`;
              (grid.get(k) || grid.set(k, []).get(k)).push(i);
            }
      });
      const seen = new Set();
      const pairs = [];
      const corners = (c) => {
        const ca = Math.cos(c.rot), sa = Math.sin(c.rot), out2 = [];
        for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
          const lx = sx * c.w / 2, lz = sz * c.h / 2;
          out2.push([c.x + lx * ca - lz * sa, c.z + lx * sa + lz * ca]);
        }
        return out2;
      };
      const sepAxis = (A, B) => {
        for (const P2 of [A, B])
          for (let k = 0; k < 4; k++) {
            const p = P2[k], q = P2[(k + 1) % 4], ax = [q[1] - p[1], p[0] - q[0]];
            const pr = (Q) => Q.map((v) => v[0] * ax[0] + v[1] * ax[1]);
            const a = pr(A), b = pr(B);
            if (Math.max(...a) <= Math.min(...b) + 1e-4 || Math.max(...b) <= Math.min(...a) + 1e-4) return true;
          }
        return false;
      };
      for (const list of grid.values())
        for (let a = 0; a < list.length; a++)
          for (let b = a + 1; b < list.length; b++) {
            const i = Math.min(list[a], list[b]), j = Math.max(list[a], list[b]), key = i * 1e6 + j;
            if (seen.has(key)) continue;
            seen.add(key);
            const A = en[i], B = en[j];
            if (Math.abs(A.y - B.y) >= (A.t + B.t) / 2 - 1e-4) continue;
            if (!sepAxis(corners(A), corners(B))) pairs.push([A.index, B.index]);
          }
      if (pairs.length) push('warn', 'chkXtalOverlap', pairs.length, pairs.slice(0, 3).map((p) => `#${p[0]}/#${p[1]}`).join(', '));
      // crystals cutting through collimator plates
      let inPlate = 0;
      en.forEach((c, i) => {
        const e = ext[i];
        for (const L of m.layers)
          if (c.y + e[1] > L.y0 && c.y - e[1] < L.y1 && Math.abs(c.x) - e[0] < L.W / 2 && Math.abs(c.z) - e[2] < L.H / 2) { inPlate++; break; }
      });
      if (inPlate) push('warn', 'chkXtalInPlate', inPlate);
    }

    // FOV must stay in front of the collimator for all rotations
    if (I.nVox > 0 && m.layers.length) {
      const front = Math.min(...m.layers.map((L) => L.y0));
      const r = Math.hypot(I.nX * I.wX / 2 + Math.abs(I.shX), I.nY * I.wY / 2 + Math.abs(I.shY));
      if (r >= front) push('warn', 'chkFovCollide', SV.fmt(r), SV.fmt(front));
    }
    if (I.nRot > 1 && Math.abs(I.nRot * I.angle - 2 * Math.PI) > 1e-3 && Math.abs(I.nRot * I.angle - Math.PI) > 1e-3)
      push('info', 'chkRotSpan', SV.fmt(I.nRot * I.angle * 180 / Math.PI));
    if (m.hasPhysics && P.E0 > 0 && P.sameWindow > 0.5 && !(P.lo < P.E0 && P.E0 < P.hi)) push('warn', 'chkWindow', P.lo, P.hi, P.E0);
    if (!out.some((x) => x.level !== 'info')) push('ok', 'chkAllOk');
    return out;
  };

  // ---------------------------------------------------------------- demo geometry
  // multi-pinhole tungsten plate + pixelated NaI(Tl) block, 140.5 keV
  SV.demoParams = function () {
    const col = new Float32Array(80000), det = new Float32Array(80000), img = new Float32Array(100), phy = new Float32Array(100);
    const W = 90, T = 6, H = 90, nP = 5, pitch = 14, r = 0.8;
    col[0] = 1;
    col.set([nP * nP, W, T, H, 0, 3.4065, 3.2165, 0.1899], 10);
    let h = 0;
    for (let i = 0; i < nP; i++)
      for (let k = 0; k < nP; k++, h++)
        col.set([(i - (nP - 1) / 2) * pitch, -T / 2, T / 2, (k - (nP - 1) / 2) * pitch, r, 0, 0, 0, 1], 100 + 9 * h);
    const nC = 40, cp = 2.0;
    det[0] = nC * nC;
    let i = 0;
    for (let iz = 0; iz < nC; iz++)
      for (let ix = 0; ix < nC; ix++, i++)
        det.set([(ix - (nC - 1) / 2) * cp, 32, (iz - (nC - 1) / 2) * cp, 1.8, 8, 1.8, 0.2431, 0.2032, 0.0398, 0.10, 0, ix < nC / 2 ? 1 : 2], 1 + 12 * i);
    img.set([48, 48, 1, 1, 1, 1, 12, 2 * Math.PI / 12, 0, 0, 0, 42]);
    phy.set([1, 1, 1, 1, 0, 126.45, 154.55, 140.5, 1, 1]);
    return { col, det, img, phy };
  };
})();
