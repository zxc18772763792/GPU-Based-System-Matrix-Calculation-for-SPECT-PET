// renderer.js - minimal WebGL2 renderer: instanced boxes / cylinders (rotated about y),
// line sets, point sets, a clip plane, selection highlight and an orbit camera.
(function () {
  'use strict';
  const SV = (window.SV = window.SV || {});
  const { V, mat4 } = SV;

  const MESH_VS = `#version 300 es
  layout(location=0) in vec3 aPos;
  layout(location=1) in vec3 aNor;
  layout(location=2) in vec4 iCR;    // centre xyz, rotation about y
  layout(location=3) in vec3 iSize;  // full extents
  layout(location=4) in vec4 iCol;
  uniform mat4 uVP;
  out vec3 vNor; out vec4 vCol; out vec3 vW; flat out int vId;
  void main() {
    vec3 p = aPos * iSize;
    float c = cos(iCR.w), s = sin(iCR.w);
    vec3 r = vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
    vNor = vec3(aNor.x * c - aNor.z * s, aNor.y, aNor.x * s + aNor.z * c);
    vW = r + iCR.xyz; vCol = iCol; vId = gl_InstanceID;
    gl_Position = uVP * vec4(vW, 1.0);
  }`;
  const MESH_FS = `#version 300 es
  precision highp float;
  in vec3 vNor; in vec4 vCol; in vec3 vW; flat in int vId;
  uniform vec4 uClip; uniform vec3 uEye; uniform int uSel; uniform vec3 uSelCol;
  out vec4 o;
  void main() {
    if (dot(vec4(vW, 1.0), uClip) > 0.0) discard;
    vec3 n = normalize(vNor);
    vec3 l = normalize(uEye - vW);
    float d = abs(dot(n, l));
    float k = 0.38 + 0.62 * d;
    vec3 col = vCol.rgb * k;
    if (vId == uSel) col = mix(col, uSelCol, 0.65);
    o = vec4(col, vId == uSel ? max(vCol.a, 0.9) : vCol.a);
  }`;
  const LINE_VS = `#version 300 es
  layout(location=0) in vec3 aPos;
  layout(location=1) in vec4 aCol;
  uniform mat4 uVP; uniform float uPt;
  out vec4 vCol; out vec3 vW;
  void main() { vW = aPos; vCol = aCol; gl_Position = uVP * vec4(aPos, 1.0); gl_PointSize = uPt; }`;
  const LINE_FS = `#version 300 es
  precision highp float;
  in vec4 vCol; in vec3 vW; uniform vec4 uClip; uniform int uRound; out vec4 o;
  void main() {
    if (dot(vec4(vW, 1.0), uClip) > 0.0) discard;
    if (uRound == 1) { vec2 q = gl_PointCoord - 0.5; if (dot(q, q) > 0.25) discard; }
    o = vCol;
  }`;

  function compile(gl, vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
    return { p, u };
  }

  // unit box [-0.5,0.5]^3 and unit cylinder along y (radius 0.5, height 1)
  function boxMesh() {
    const P = [], N = [], I = [];
    const faces = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
      [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
    for (const [n, u, v] of faces) {
      const b = P.length / 3;
      for (const [su, sv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        P.push(0.5 * (n[0] + su * u[0] + sv * v[0]), 0.5 * (n[1] + su * u[1] + sv * v[1]), 0.5 * (n[2] + su * u[2] + sv * v[2]));
        N.push(...n);
      }
      I.push(b, b + 1, b + 2, b, b + 2, b + 3);
    }
    return { P, N, I };
  }
  function cylMesh(seg) {
    const P = [], N = [], I = [];
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2, x = Math.cos(a), z = Math.sin(a);
      P.push(0.5 * x, -0.5, 0.5 * z, 0.5 * x, 0.5, 0.5 * z);
      N.push(x, 0, z, x, 0, z);
    }
    for (let i = 0; i < seg; i++) { const b = 2 * i; I.push(b, b + 1, b + 3, b, b + 3, b + 2); }
    for (const y of [-0.5, 0.5]) {
      const c = P.length / 3;
      P.push(0, y, 0); N.push(0, Math.sign(y), 0);
      for (let i = 0; i <= seg; i++) {
        const a = (i / seg) * Math.PI * 2;
        P.push(0.5 * Math.cos(a), y, 0.5 * Math.sin(a)); N.push(0, Math.sign(y), 0);
      }
      for (let i = 0; i < seg; i++) I.push(c, c + 1 + i, c + 2 + i);
    }
    return { P, N, I };
  }

  class Renderer {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
      if (!gl) throw new Error('WebGL2 not available');
      this.gl = gl;
      this.mesh = compile(gl, MESH_VS, MESH_FS);
      this.line = compile(gl, LINE_VS, LINE_FS);
      this.geoms = { box: this._geom(boxMesh()), cyl: this._geom(cylMesh(20)) };
      this.batches = new Map();
      this.lines = new Map();
      this.clip = [0, 0, 0, -1];
      this.sel = { batch: null, id: -1 };
      this.bg = [0.06, 0.07, 0.09];
      // orbit camera: spherical angles around the chosen vertical axis ('z' or 'y')
      this.cam = { target: [0, 0, 0], dist: 300, theta: -2.2, phi: 0.45, fov: 40 * Math.PI / 180, ortho: false, up: 'z' };
      this.axisLength = 20;   // world length of the axis lines (labelled on the overlay)
      this.overlay = null;    // optional 2D canvas for the orientation gizmo and axis labels
      this.dirty = true;
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
      this._installControls();
    }
    _geom(m) {
      const gl = this.gl;
      const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(m.P), gl.STATIC_DRAW);
      const nb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, nb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(m.N), gl.STATIC_DRAW);
      const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(m.I), gl.STATIC_DRAW);
      return { pb, nb, ib, n: m.I.length };
    }

    // inst: Float32Array, 7 per instance (cx, cy, cz, rotY, sx, sy, sz); col: Float32Array, 4 per instance
    setInstances(name, geom, inst, col, opts = {}) {
      const gl = this.gl;
      this.removeBatch(name);
      const g = this.geoms[geom];
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.pb); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.nb); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
      const ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ib); gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 28, 0); gl.vertexAttribDivisor(2, 1);
      gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 28, 16); gl.vertexAttribDivisor(3, 1);
      const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(4, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.ib);
      gl.bindVertexArray(null);
      this.batches.set(name, { vao, ib, cb, g, count: inst.length / 7, transparent: !!opts.transparent, visible: opts.visible !== false, order: opts.order || 0 });
      this.dirty = true;
    }
    setColors(name, col) {
      const b = this.batches.get(name);
      if (!b) return;
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, b.cb);
      gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
      this.dirty = true;
    }
    setTransparent(name, t) { const b = this.batches.get(name); if (b) { b.transparent = t; this.dirty = true; } }
    removeBatch(name) {
      const b = this.batches.get(name);
      if (!b) return;
      const gl = this.gl;
      gl.deleteVertexArray(b.vao); gl.deleteBuffer(b.ib); gl.deleteBuffer(b.cb);
      this.batches.delete(name);
      this.dirty = true;
    }
    // pos: Float32Array (3/vertex), col: Float32Array (4/vertex), mode 'lines' | 'points'
    setLines(name, pos, col, opts = {}) {
      const gl = this.gl;
      this.removeLines(name);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      this.lines.set(name, { vao, pb, cb, n: pos.length / 3, mode: opts.mode || 'lines', size: opts.size || 6, visible: opts.visible !== false, noClip: !!opts.noClip });
      this.dirty = true;
    }
    removeLines(name) {
      const l = this.lines.get(name);
      if (!l) return;
      this.gl.deleteVertexArray(l.vao); this.gl.deleteBuffer(l.pb); this.gl.deleteBuffer(l.cb);
      this.lines.delete(name);
      this.dirty = true;
    }
    setVisible(name, v) {
      const b = this.batches.get(name) || this.lines.get(name);
      if (b) { b.visible = v; this.dirty = true; }
    }
    isVisible(name) { const b = this.batches.get(name) || this.lines.get(name); return !!(b && b.visible); }
    setSelection(batch, id) { this.sel = { batch, id }; this.dirty = true; }
    setClip(plane) { this.clip = plane || [0, 0, 0, -1]; this.dirty = true; }

    // ---------------------------------------------------------------- camera
    // right-handed orbit basis (e1, e2, eu): eu is the screen-vertical world axis
    basis() {
      return this.cam.up === 'y' ? { e1: [0, 0, 1], e2: [1, 0, 0], eu: [0, 1, 0] } : { e1: [1, 0, 0], e2: [0, 1, 0], eu: [0, 0, 1] };
    }
    eye() {
      const c = this.cam, cp = Math.cos(c.phi), B = this.basis();
      const d = V.add(V.add(V.scale(B.e1, cp * Math.cos(c.theta)), V.scale(B.e2, cp * Math.sin(c.theta))), V.scale(B.eu, Math.sin(c.phi)));
      return V.add(c.target, V.scale(d, c.dist));
    }
    // camera frame: f forward, r screen-right, u screen-up (world vectors)
    frame() {
      const f = V.norm(V.sub(this.cam.target, this.eye())), r = V.norm(V.cross(f, this.basis().eu));
      return { f, r, u: V.cross(r, f) };
    }
    viewProj() {
      const c = this.cam, w = this.canvas.width, h = this.canvas.height, aspect = w / Math.max(1, h);
      const eye = this.eye();
      const view = mat4.lookAt(eye, c.target, this.basis().eu);
      let proj;
      if (c.ortho) {
        const hh = c.dist * Math.tan(c.fov / 2);
        proj = mat4.ortho(-hh * aspect, hh * aspect, -hh, hh, -c.dist * 20, c.dist * 20);
      } else proj = mat4.perspective(c.fov, aspect, Math.max(c.dist * 0.005, 0.01), c.dist * 50);
      return mat4.mul(proj, view);
    }
    fit(min, max) {
      const c = this.cam;
      c.target = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
      const r = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2 || 50;
      c.dist = r / Math.sin(c.fov / 2) * 0.8;
      this.dirty = true;
    }
    // Preset views. Each one states the viewing direction and matches a 2D view:
    //   beam  : look along +y (from the FOV towards the detector), x right, z up  = layout map
    //   beamR : look along -y (from behind the detector),           x left,  z up  = mirrored layout map
    //   axial : look along -z (from +z),                            x right, y up  = voxel image ("physical")
    //   side  : look along -x (from +x),                            y right, z up
    static get VIEWS() {
      return {
        iso: { up: 'z', theta: -2.2, phi: 0.45 },
        isoY: { up: 'y', theta: 0.75, phi: 0.5 },
        beam: { up: 'z', theta: -Math.PI / 2, phi: 0 },
        beamR: { up: 'z', theta: Math.PI / 2, phi: 0 },
        axial: { up: 'y', theta: 0, phi: 0 },
        side: { up: 'z', theta: 0, phi: 0 },
      };
    }
    setView(name) {
      const v = Renderer.VIEWS[name] || Renderer.VIEWS.iso;
      Object.assign(this.cam, v);
      this.dirty = true;
      if (this.onViewChange) this.onViewChange(name);
    }
    view(name) { this.setView(name); }
    // change the vertical axis but keep looking from (about) the same side
    setUp(up) {
      if (up === this.cam.up) return;
      const d = V.norm(V.sub(this.eye(), this.cam.target));
      this.cam.up = up;
      const B = this.basis();
      this.cam.phi = SV.clamp(Math.asin(SV.clamp(V.dot(d, B.eu), -1, 1)), -1.55, 1.55);
      this.cam.theta = Math.atan2(V.dot(d, B.e2), V.dot(d, B.e1));
      this.dirty = true;
    }
    // ray through canvas pixel (CSS px)
    ray(px, py) {
      const rect = this.canvas.getBoundingClientRect();
      const x = (2 * (px - rect.left)) / rect.width - 1, y = 1 - (2 * (py - rect.top)) / rect.height;
      const c = this.cam, eye = this.eye();
      const { f, r, u } = this.frame();
      const aspect = rect.width / Math.max(1, rect.height), t = Math.tan(c.fov / 2);
      if (c.ortho) {
        const hh = c.dist * t;
        const o = V.add(eye, V.add(V.scale(r, x * hh * aspect), V.scale(u, y * hh)));
        return { o, d: f };
      }
      return { o: eye, d: V.norm(V.add(f, V.add(V.scale(r, x * t * aspect), V.scale(u, y * t)))) };
    }

    _installControls() {
      const cv = this.canvas;
      let drag = null;
      cv.addEventListener('contextmenu', (e) => e.preventDefault());
      cv.addEventListener('pointerdown', (e) => {
        drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, pan: e.button === 2 || e.button === 1 || e.shiftKey };
        cv.setPointerCapture(e.pointerId);
      });
      cv.addEventListener('pointermove', (e) => {
        if (!drag) { if (this.onHover) this.onHover(e); return; }
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        const c = this.cam;
        if (drag.pan) {
          const { r, u } = this.frame();
          const s = (2 * c.dist * Math.tan(c.fov / 2)) / cv.clientHeight;
          c.target = V.add(c.target, V.add(V.scale(r, -dx * s), V.scale(u, dy * s)));
        } else {
          c.theta -= dx * 0.008;
          c.phi = SV.clamp(c.phi + dy * 0.008, -1.55, 1.55);
        }
        this.dirty = true;
      });
      cv.addEventListener('pointerup', (e) => {
        if (drag && Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) < 4 && this.onClick) this.onClick(e);
        drag = null;
      });
      cv.addEventListener('wheel', (e) => {
        e.preventDefault();
        this.cam.dist *= Math.exp(e.deltaY * 0.0012);
        this.dirty = true;
      }, { passive: false });
      new ResizeObserver(() => { this.dirty = true; }).observe(cv);
    }

    _loop() {
      requestAnimationFrame(this._loop);
      const cv = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; this.dirty = true; }
      if (!this.dirty) return;
      this.dirty = false;
      this.render();
    }

    render() {
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      const vp = this.viewProj(), eye = this.eye();

      // lines / points (opaque)
      gl.useProgram(this.line.p);
      gl.uniformMatrix4fv(this.line.u.uVP, false, vp);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      for (const l of this.lines.values()) {
        if (!l.visible || !l.n) continue;
        gl.uniform4fv(this.line.u.uClip, l.noClip ? [0, 0, 0, -1] : this.clip);
        gl.uniform1f(this.line.u.uPt, l.size * Math.min(window.devicePixelRatio || 1, 2));
        gl.uniform1i(this.line.u.uRound, l.mode === 'points' ? 1 : 0);
        gl.bindVertexArray(l.vao);
        gl.drawArrays(l.mode === 'points' ? gl.POINTS : gl.LINES, 0, l.n);
      }

      // meshes: opaque first, then transparent without depth writes
      gl.useProgram(this.mesh.p);
      gl.uniformMatrix4fv(this.mesh.u.uVP, false, vp);
      gl.uniform4fv(this.mesh.u.uClip, this.clip);
      gl.uniform3fv(this.mesh.u.uEye, eye);
      gl.uniform3fv(this.mesh.u.uSelCol, [1.0, 0.25, 0.35]);
      const list = [...this.batches.entries()].filter(([, b]) => b.visible && b.count > 0).sort((a, b) => a[1].order - b[1].order);
      for (const pass of [false, true]) {
        gl.depthMask(!pass);
        for (const [name, b] of list) {
          if (b.transparent !== pass) continue;
          gl.uniform1i(this.mesh.u.uSel, this.sel.batch === name ? this.sel.id : -1);
          gl.bindVertexArray(b.vao);
          gl.drawElementsInstanced(gl.TRIANGLES, b.g.n, gl.UNSIGNED_SHORT, 0, b.count);
        }
      }
      gl.depthMask(true);
      gl.bindVertexArray(null);
      this.drawOverlay(vp);
      if (this.onRender) this.onRender();
    }

    // ---------------------------------------------------------------- overlay
    // Orientation gizmo (bottom right) that turns with the camera, plus labels at the
    // tips of the 3D axis lines, so the direction of every axis is always readable.
    drawOverlay(vp) {
      const ov = this.overlay;
      if (!ov) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2), W = ov.clientWidth, H = ov.clientHeight;
      if (ov.width !== Math.round(W * dpr) || ov.height !== Math.round(H * dpr)) { ov.width = Math.round(W * dpr); ov.height = Math.round(H * dpr); }
      const ctx = ov.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const AX = [['x', [1, 0, 0], '#ff6b6b'], ['y', [0, 1, 0], '#51cf66'], ['z', [0, 0, 1], '#4ea1ff']];
      ctx.font = '600 12px system-ui, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      // labels at the 3D axis tips
      if (this.isVisible('axes')) {
        const proj = (p) => {
          const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12], y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
          const w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
          return w > 1e-6 ? [(x / w * 0.5 + 0.5) * W, (0.5 - y / w * 0.5) * H] : null;
        };
        const o = proj([0, 0, 0]);
        for (const [n, a, col] of AX) {
          const q = proj(V.scale(a, this.axisLength * 1.12));
          if (!q || !o || Math.hypot(q[0] - o[0], q[1] - o[1]) < 6) continue;
          ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(14,17,22,0.9)'; ctx.strokeText('+' + n, q[0], q[1]);
          ctx.fillStyle = col; ctx.fillText('+' + n, q[0], q[1]);
        }
      }
      // gizmo
      const { f, r, u } = this.frame(), cx = W - 52, cy = H - 52, L = 34;
      ctx.fillStyle = 'rgba(21,26,33,0.8)';
      ctx.beginPath(); ctx.arc(cx, cy, 46, 0, 2 * Math.PI); ctx.fill();
      const items = AX.map(([n, a, col]) => ({ n, col, sx: V.dot(a, r), sy: V.dot(a, u), depth: V.dot(a, f) }));
      items.sort((p, q) => q.depth - p.depth);   // far axes first
      for (const it of items) {
        const ex = cx + it.sx * L, ey = cy - it.sy * L, away = it.depth > 0.2;
        ctx.strokeStyle = it.col; ctx.globalAlpha = away ? 0.45 : 1; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.fillStyle = it.col;
        ctx.beginPath(); ctx.arc(ex, ey, 8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#0e1116'; ctx.fillText(it.n, ex, ey + 0.5);
        ctx.globalAlpha = 1;
      }
    }
  }
  SV.Renderer = Renderer;
})();
