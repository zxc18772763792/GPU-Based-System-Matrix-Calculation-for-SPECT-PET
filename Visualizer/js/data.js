// data.js - data sources (local File objects or files served by serve.py through
// HTTP Range requests) and streamed access to [rotation][detector][voxel] system matrices.
(function () {
  'use strict';
  const SV = (window.SV = window.SV || {});

  class LocalSource {
    constructor(file) { this.file = file; this.name = file.name; this.path = file.webkitRelativePath || file.name; this.size = file.size; }
    async read(offset, length) { return this.file.slice(offset, offset + length).arrayBuffer(); }
    async text() { return this.file.text(); }
    async strided(offset, stride, count) {
      const out = new Float32Array(count);
      const span = (count - 1) * stride + 4;
      if (span <= 128 * 1024 * 1024) {            // one contiguous read
        const buf = new DataView(await this.read(offset, span));
        for (let i = 0; i < count; i++) out[i] = buf.getFloat32(i * stride, true);
        return out;
      }
      const batch = 512;
      for (let s = 0; s < count; s += batch) {
        const n = Math.min(batch, count - s);
        const parts = await Promise.all(Array.from({ length: n }, (_, k) => this.read(offset + (s + k) * stride, 4)));
        parts.forEach((b, k) => { out[s + k] = new DataView(b).getFloat32(0, true); });
      }
      return out;
    }
  }

  class RemoteSource {
    constructor(path, size) { this.path = path; this.name = path.split('/').pop(); this.size = size; this.url = 'data/' + path.split('/').map(encodeURIComponent).join('/'); }
    async read(offset, length) {
      const r = await fetch(this.url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
      if (!r.ok) throw new Error(`${this.name}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }
    async text() { const r = await fetch(this.url); if (!r.ok) throw new Error(`${this.name}: HTTP ${r.status}`); return r.text(); }
    async strided(offset, stride, count) {
      const q = new URLSearchParams({ file: this.path, offset, stride, count });
      const r = await fetch('api/strided?' + q);
      if (!r.ok) throw new Error(`${this.name}: HTTP ${r.status}`);
      return new Float32Array(await r.arrayBuffer());
    }
  }
  SV.LocalSource = LocalSource;
  SV.RemoteSource = RemoteSource;

  // list files exposed by serve.py (null when the page is not served by it)
  SV.serverFiles = async function () {
    if (location.protocol === 'file:') return null;
    try {
      const r = await fetch('api/files');
      if (!r.ok) return null;
      return await r.json();
    } catch (e) { return null; }
  };

  // classify a file by its name
  SV.classify = function (name) {
    const n = name.toLowerCase();
    if (n.endsWith('.dat')) {
      // only the four parameter files (Params_Collimator.dat, ...), not other .dat files of a run
      const m = /^params?_(collimator|detector|image|physics)\.dat$/.exec(n.split('/').pop());
      return m ? { collimator: 'col', detector: 'det', image: 'img', physics: 'phy' }[m[1]] : null;
    }
    if (n.endsWith('_relerr.sysmat')) return 'relerr';    // relative statistical errors of an MC matrix, not a matrix
    if (n.endsWith('.sysmat')) return 'sysmat';
    if (n.endsWith('_spectrum.csv')) return 'spectrum';
    if (n.endsWith('_tracks.json')) return 'tracks';
    if (n.endsWith('_summary.json')) return 'summary';
    return null;
  };

  // ------------------------------------------------------------------ sysmat
  class Sysmat {
    constructor(src, nDet, nVox) {
      this.src = src; this.nDet = nDet; this.nVox = nVox;
      const per = nDet * nVox * 4;
      this.nRot = per > 0 ? src.size / per : 0;
      this.ok = per > 0 && Number.isInteger(this.nRot) && this.nRot >= 1;
    }
    // detector response over all voxels (one contiguous read)
    async row(rot, det) {
      const off = ((rot * this.nDet + det) * this.nVox) * 4;
      return new Float32Array(await this.src.read(off, this.nVox * 4));
    }
    // projection of one voxel onto all detectors (strided read)
    async column(rot, vox) {
      return this.src.strided(((rot * this.nDet) * this.nVox + vox) * 4, this.nVox * 4, this.nDet);
    }
    // stream one rotation: per-voxel sensitivity (sum over detectors) and per-detector sum
    async sums(rot, onProgress, signal) {
      const vox = new Float64Array(this.nVox), det = new Float64Array(this.nDet);
      const rowsPerChunk = Math.max(1, Math.floor((32 * 1024 * 1024) / (this.nVox * 4)));
      for (let d0 = 0; d0 < this.nDet; d0 += rowsPerChunk) {
        if (signal && signal.aborted) throw new Error('aborted');
        const n = Math.min(rowsPerChunk, this.nDet - d0);
        const f = new Float32Array(await this.src.read(((rot * this.nDet + d0) * this.nVox) * 4, n * this.nVox * 4));
        for (let r = 0; r < n; r++) {
          let s = 0;
          const base = r * this.nVox;
          for (let v = 0; v < this.nVox; v++) { const x = f[base + v]; vox[v] += x; s += x; }
          det[d0 + r] = s;
        }
        if (onProgress) onProgress((d0 + n) / this.nDet);
      }
      return { vox: Float32Array.from(vox), det: Float32Array.from(det) };
    }
  }
  SV.Sysmat = Sysmat;

  // ------------------------------------------------------------------ MC outputs
  SV.parseSpectrum = function (text) {
    const lines = text.trim().split(/\r?\n/);
    const head = lines[0].split(',');
    const cols = {};
    head.forEach((h, i) => { cols[h.trim()] = []; });
    for (let k = 1; k < lines.length; k++) {
      const p = lines[k].split(',');
      head.forEach((h, i) => cols[h.trim()].push(parseFloat(p[i])));
    }
    const n = cols.E_low_keV ? cols.E_low_keV.length : 0;
    const E = new Float32Array(n);
    for (let i = 0; i < n; i++) E[i] = (cols.E_low_keV[i] + cols.E_high_keV[i]) / 2;
    return { E, cols, binW: n ? cols.E_high_keV[0] - cols.E_low_keV[0] : 1 };
  };
})();
