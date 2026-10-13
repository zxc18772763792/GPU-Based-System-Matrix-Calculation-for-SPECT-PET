// Orientation regression tests for the viewer (headless Chromium via Playwright).
//   npm i -D playwright      (or a global install:  NODE_PATH=$(npm root -g) node ...)
//   node Visualizer/tests/orientation.test.js [screenshot-dir]
// On the built-in demo geometry it checks: axis directions in every preset view (WebGL pixels), the
// screen-axis note, the layout map (+y view and mirrored), voxel-image orientation modes and
// click-back, the 3D voxel marker vs. the physical image, the detector arrow vs. rotation, and the
// sign of the crystal rotation (3D pixels and layout picking) against the computation code.
const { chromium } = require('playwright');
const path = require('path');
const PAGE = 'file://' + path.resolve(__dirname, '..', 'index.html');
const OUT = process.argv[2] || null;
let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? '  ok  ' : ' FAIL '} ${msg}`); if (!ok) fails++; };
(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(PAGE);
  await page.click('#demoBtn');
  await page.waitForTimeout(400);

  // ---------------------------------------------------------------- 1. axes in every preset
  const axisDirs = async (v) => page.evaluate((v) => {
    const R = SV.app.renderer, gl = R.gl;
    const vis = {}; for (const n of [...R.batches.keys(), ...R.lines.keys()]) { vis[n] = R.isVisible(n); R.setVisible(n, n === 'axes'); }
    R.cam.target = [0, 0, 0];
    R.setView(v); R.render();
    const w = R.canvas.width, h = R.canvas.height, px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const acc = { r: [0, 0, 0], g: [0, 0, 0], b: [0, 0, 0] };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const o = 4 * (y * w + x), r = px[o], g = px[o + 1], b = px[o + 2];
      const k = r > 90 && r > 1.6 * g && r > 1.6 * b ? 'r' : g > 70 && g > 1.4 * r && g > 1.3 * b ? 'g' : b > 80 && b > 1.25 * g && b > 1.8 * r ? 'b' : null;
      if (k) { acc[k][0] += x; acc[k][1] += y; acc[k][2]++; }
    }
    for (const n in vis) R.setVisible(n, vis[n]);
    const out = {};
    for (const k of ['r', 'g', 'b']) { const n = acc[k][2]; out[k] = n > 15 ? [acc[k][0] / n - w / 2, acc[k][1] / n - h / 2] : null; }
    out.note = document.getElementById('viewNote').textContent;
    return out;
  }, v);
  const dirOf = (p) => (!p ? 'none' : Math.abs(p[0]) > 3 * Math.abs(p[1]) ? (p[0] > 0 ? 'right' : 'left') : Math.abs(p[1]) > 3 * Math.abs(p[0]) ? (p[1] > 0 ? 'up' : 'down') : 'oblique');
  const expect = {
    beam: { r: 'right', g: 'none', b: 'up', note: ['+x', '+z'] },
    beamR: { r: 'left', g: 'none', b: 'up', note: ['−x', '+z'] },
    axial: { r: 'right', g: 'up', b: 'none', note: ['+x', '+y'] },
    side: { r: 'none', g: 'right', b: 'up', note: ['+y', '+z'] },
  };
  for (const [v, e] of Object.entries(expect)) {
    const d = await axisDirs(v);
    const got = { r: dirOf(d.r), g: dirOf(d.g), b: dirOf(d.b) };
    check(got.r === e.r && got.g === e.g && got.b === e.b, `view ${v}: +x ${got.r}, +y ${got.g}, +z ${got.b}   (note: "${d.note}")`);
    check(d.note.includes(e.note[0]) && d.note.includes(e.note[1]), `view ${v}: note names right=${e.note[0]} up=${e.note[1]}`);
  }
  // iso: all three visible, z up
  const iso = await axisDirs('iso');
  check(iso.r && iso.g && iso.b && dirOf(iso.b) === 'up', `view iso: all axes visible, +z up (${iso.note})`);
  // vertical-axis switch keeps the camera side and makes +y the screen-up axis
  await page.selectOption('#upAxis', 'y');
  const upy = await page.evaluate(() => { const { u } = SV.app.renderer.frame(); return u; });
  check(upy[1] > 0.5, `vertical axis y: screen-up has +y component ${upy[1].toFixed(2)}`);
  await page.selectOption('#upAxis', 'z');

  // ---------------------------------------------------------------- 2. layout map
  const layout = await page.evaluate(() => {
    const app = SV.app, L = app.views.layout, m = app.model;
    const c = m.crystals.reduce((a, b) => (b.x + b.z > a.x + a.z ? b : a));   // +x, +z corner crystal
    const res = {};
    for (const mirror of [false, true]) {
      L.mirror = mirror; L.draw(document.getElementById('c2d').getContext('2d'), L.w || 800, L.h || 300);
      const px = L.X(c.x), pz = L.Z(c.z), ox = L.X(0), oz = L.Z(0);
      res[mirror] = { right: px > ox, up: pz < oz, pick: (L.pick(px, pz) || {}).index === c.index };
    }
    L.mirror = false;
    return { res, idx: c.index, x: c.x, z: c.z };
  });
  check(layout.res.false.right && layout.res.false.up && layout.res.false.pick, `layout (+y view): crystal #${layout.idx} at x=${layout.x}, z=${layout.z} drawn right/up, picked back`);
  check(!layout.res.true.right && layout.res.true.up && layout.res.true.pick, `layout mirrored (−y view): same crystal drawn left/up, picked back`);

  // ---------------------------------------------------------------- 3. voxel image orientation
  await page.click('.tab[data-tab=image]');
  const hot = await page.evaluate(() => {
    const app = SV.app, I = app.model.image;
    const ix = I.nX - 3, iy = I.nY - 6, v = iy * I.nX + ix, vals = new Float32Array(I.nVox); vals[v] = 1;
    app.voxValues = { values: vals, label: 'test', kind: 'test' };
    return { v, ix, iy, nX: I.nX, nY: I.nY };
  });
  for (const [mode, want] of [['phys', ['right', 'top']], ['numpy', ['right', 'bottom']], ['matlab', ['right', 'bottom']]]) {
    await page.selectOption('#imgOrient', mode);
    await page.waitForTimeout(150);
    const r = await page.evaluate(() => {
      const cv = document.getElementById('c2d'), ctx = cv.getContext('2d'), w = cv.width, h = cv.height, d = ctx.getImageData(0, 0, w, h).data;
      const view = SV.app.views.image, rect = view.rect(cv.clientWidth, cv.clientHeight), dpr = w / cv.clientWidth;
      let sx = 0, sy = 0, n = 0;
      for (let y = Math.floor(rect.y * dpr); y < (rect.y + rect.h) * dpr; y++) for (let x = Math.floor(rect.x * dpr); x < (rect.x + rect.w) * dpr; x++) {
        const o = 4 * (y * w + x);
        if (d[o] > 200 && d[o + 1] > 200 && d[o + 2] < 90) { sx += x; sy += y; n++; }   // yellow = max of viridis
      }
      const cx = sx / n / dpr, cy = sy / n / dpr;
      return { n, fx: (cx - rect.x) / rect.w, fy: (cy - rect.y) / rect.h, back: view.voxelAt(cx, cy) };
    });
    const horiz = r.fx > 0.5 ? 'right' : 'left', vert = r.fy < 0.5 ? 'top' : 'bottom';
    check(r.n > 0 && horiz === want[0] && vert === want[1] && r.back === hot.v,
      `image ${mode}: hot voxel (ix ${hot.ix}, iy ${hot.iy}) at ${horiz}/${vert} (fx ${r.fx.toFixed(2)}, fy ${r.fy.toFixed(2)}), click-back -> ${r.back}`);
  }
  // matlab: horizontal position follows iy, vertical follows ix
  await page.selectOption('#imgOrient', 'matlab');
  const ml = await page.evaluate((hot) => { const v = SV.app.views.image; return [v.toCell(hot.ix, hot.iy), v.dims()]; }, hot);
  check(ml[0][0] === hot.iy && ml[0][1] === hot.ix && ml[1].cols === hot.nY, `image matlab: cell = (col iy, row ix) = (${ml[0]})`);
  await page.selectOption('#imgOrient', 'phys');

  // ---------------------------------------------------------------- 4. same voxel in 3D (view along -z) vs physical image
  const v3 = await page.evaluate((hot) => {
    const app = SV.app, R = app.renderer, I = app.model.image;
    app.select({ kind: 'voxel', index: hot.v });
    R.cam.target = [I.shX, I.shY, I.shZ]; R.setView('axial'); R.render();
    const vp = R.viewProj(), p = SV.voxelPos(I, hot.v, app.state.rot).world;
    const pr = (q) => { const x = vp[0] * q[0] + vp[4] * q[1] + vp[8] * q[2] + vp[12], y = vp[1] * q[0] + vp[5] * q[1] + vp[9] * q[2] + vp[13], w = vp[3] * q[0] + vp[7] * q[1] + vp[11] * q[2] + vp[15]; return [x / w, y / w]; };
    const a = pr(p), o = pr([I.shX, I.shY, I.shZ]);
    return { right: a[0] > o[0], up: a[1] > o[1], marker: R.lines.has('selmark') };
  }, hot);
  check(v3.marker && v3.right && v3.up, `3D view along −z: selected voxel marker is right/up of the FOV centre (same as the physical image)`);

  // ---------------------------------------------------------------- 5. detector arrow vs rotation
  const arrows = await page.evaluate(() => {
    const app = SV.app, view = app.views.image, I = app.model.image, out = [];
    for (const rot of [0, 3, 6, 9]) {
      const a = rot * I.angle, [sx, sy] = view.screenDir(Math.sin(a), Math.cos(a));
      // world +y expressed in local voxel coordinates must map back to world +y
      const loc = [Math.sin(a), Math.cos(a)], wx = loc[0] * Math.cos(a) - loc[1] * Math.sin(a), wy = loc[0] * Math.sin(a) + loc[1] * Math.cos(a);
      out.push({ rot, deg: Math.round(a * 180 / Math.PI), sx: Math.round(sx), sy: Math.round(sy), wx: +wx.toFixed(6), wy: +wy.toFixed(6) });
    }
    return out;
  });
  const want = { 0: [0, -1], 3: [1, 0], 6: [0, 1], 9: [-1, 0] };   // phys mode: up, right, down, left
  for (const a of arrows)
    check(a.sx === want[a.rot][0] && a.sy === want[a.rot][1] && a.wx === 0 && a.wy === 1,
      `detector arrow rotation ${a.rot} (${a.deg}°): screen (${a.sx}, ${a.sy}); local->world gives (${a.wx}, ${a.wy})`);

  await page.setViewportSize({ width: 1400, height: 900 });
  await page.evaluate(() => { const R = SV.app.renderer; R.fit(SV.app.model.bounds.min, SV.app.model.bounds.max); R.setView('iso'); });
  await page.click('.tab[data-tab=image]');
  await page.fill('#rotIndex', '2'); await page.dispatchEvent('#rotIndex', 'input');
  await page.waitForTimeout(400);
  if (OUT) await page.screenshot({ path: OUT + '/orient_iso.png' });
  await page.evaluate(() => SV.app.renderer.setView('beam'));
  await page.click('.tab[data-tab=layout]');
  await page.waitForTimeout(300);
  if (OUT) await page.screenshot({ path: OUT + '/orient_beam.png' });
  // ---------------------------------------------------------------- 6. crystal rotation sign
  const f32 = (n, set) => { const a = new Float32Array(n); set(a); return Buffer.from(a.buffer); };
  const rot = Math.PI / 6;
  await page.goto(PAGE);
  await page.setInputFiles('#fileInput', [
    { name: 'Params_Collimator.dat', mimeType: 'application/octet-stream', buffer: f32(80000, (a) => { a[0] = 0; }) },
    { name: 'Params_Detector.dat', mimeType: 'application/octet-stream', buffer: f32(80000, (a) => { a[0] = 1; a.set([0, 30, 0, 40, 4, 2, 0.3, 0.2, 0.1, 0.1, rot, 1], 1); }) },
    { name: 'Params_Image.dat', mimeType: 'application/octet-stream', buffer: f32(100, (a) => a.set([4, 4, 1, 1, 1, 1, 1, 0, 0, 0, 0, 10])) },
  ]);
  await page.waitForTimeout(500);
  const r = await page.evaluate((rot) => {
    const app = SV.app, R = app.renderer, gl = R.gl;
    for (const n of [...R.batches.keys(), ...R.lines.keys()]) R.setVisible(n, n === 'crystals');
    app.renderer.setColors('crystals', new Float32Array([1, 0, 0, 1]));
    R.cam.target = [0, 40, 0]; R.cam.dist = 120; R.setView('beam'); R.render();
    const w = R.canvas.width, h = R.canvas.height, px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let n = 0, mx = 0, my = 0, sxx = 0, sxy = 0;
    const pts = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const o = 4 * (y * w + x); if (px[o] > 60 && px[o + 1] < 40) pts.push([x, y]); }
    for (const [x, y] of pts) { mx += x; my += y; n++; }
    mx /= n; my /= n;
    for (const [x, y] of pts) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); }
    const slope3d = sxy / sxx;                       // screen y up (readPixels), so > 0 means up-right
    // layout map and picking use the same convention
    const L = app.views.layout, c = app.model.crystals[0];
    L.draw(document.getElementById('c2d').getContext('2d'), 800, 300);
    const tip = [c.x + 18 * Math.cos(rot), c.z + 18 * Math.sin(rot)], wrong = [c.x + 18 * Math.cos(rot), c.z - 18 * Math.sin(rot)];
    return { slope3d, expected: Math.tan(rot), pickTip: !!L.pick(L.X(tip[0]), L.Z(tip[1])), pickWrong: !!L.pick(L.X(wrong[0]), L.Z(wrong[1])) };
  }, rot);
  check(Math.abs(r.slope3d - r.expected) < 0.06, `crystal rotation +30°: 3D long-axis slope ${r.slope3d.toFixed(3)} (tan30° = ${r.expected.toFixed(3)})`);
  check(r.pickTip && !r.pickWrong, `crystal rotation +30°: layout picks the +30° tip (${r.pickTip}) and not the −30° one (${r.pickWrong})`);
  console.log(errors.length ? 'ERRORS ' + JSON.stringify(errors) : 'no console errors');
  if (errors.length) fails++;
  console.log(fails ? `${fails} check(s) FAILED` : 'all orientation checks passed');
  await browser.close();
  process.exit(fails ? 1 : 0);
})();
