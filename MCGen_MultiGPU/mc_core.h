// mc_core.h
// Monte Carlo photon transport core shared by the CUDA (multi-GPU) and CPU backends.
//
// Everything in this header is plain C++ that compiles both with nvcc (as
// __host__ __device__ code) and with a regular host compiler, so the GPU and
// CPU backends run *exactly* the same physics and geometry code.
//
// Physics (photon energies 1 keV .. 1.1 MeV, tuned for 140.5 keV and 511 keV):
//   * photoelectric absorption, optional K-shell fluorescence (Z >= 30)
//   * incoherent scattering, Klein-Nishina (free electron) angular sampling
//   * coherent (Rayleigh) scattering with independent-atom form factors
//   * electrons are absorbed locally (range << crystal size at these energies)
// Tracking:
//   * delta (Woodcock) tracking inside axis-aligned regions with per-region,
//     energy dependent majorants; straight flights between regions
//   * exact analog flight inside collimator holes (one step per hole)
//   * uniform grids for hole and crystal look-ups
//
// Units: mm, keV, attenuation coefficients in 1/mm.
// Geometry conventions are those of the analytic modules (see ReadMe.md).

#pragma once
#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__)
#define MC_HD __host__ __device__ __forceinline__
#else
#define MC_HD inline
#endif

namespace mc {

constexpr float kElectronMass = 510.99895f;   // keV
constexpr float kHcKeVA = 12.398419843f;      // h*c in keV*Angstrom
constexpr float kPi = 3.14159265358979f;
constexpr float kBig = 1e30f;
constexpr float kPush = 1e-3f;                // mm, nudge across boundaries

constexpr int kMaxLayers = 10;                // same limit as the analytic code
constexpr int kMaxRegions = 16;
constexpr int kMaxFluo = 12;                  // K lines per material
constexpr int kMaxDeposits = 16;              // crystals hit per history
constexpr int kMaxStack = 4;                  // secondary (fluorescence) photons
constexpr int kMaxWindows = 4;
constexpr int kMaxSteps = 100000;
constexpr int kNumClasses = 3;                // primary / collimator-scatter / crystal-scatter
constexpr int kMaxTargets = 32;               // collimator holes aimed at per voxel

enum EventClass { kPrimary = 0, kCollimatorScatter = 1, kCrystalScatter = 2 };
enum VertexType { kVtxEmit = 0, kVtxPhoto = 1, kVtxCompton = 2, kVtxRayleigh = 3,
                  kVtxEscape = 4, kVtxFluoEmit = 5, kVtxCutoff = 6, kVtxKilled = 7 };
enum WindowMode { kWinGauss = 0, kWinIdeal = 1, kWinNone = 2 };
enum FlagBits { kFlagCollimator = 1, kFlagXtalScatter = 2, kFlagScattered = 4 };

struct V3 { float x, y, z; };
MC_HD V3 make3(float x, float y, float z) { V3 v; v.x = x; v.y = y; v.z = z; return v; }
MC_HD V3 operator+(V3 a, V3 b) { return make3(a.x + b.x, a.y + b.y, a.z + b.z); }
MC_HD V3 operator-(V3 a, V3 b) { return make3(a.x - b.x, a.y - b.y, a.z - b.z); }
MC_HD V3 operator*(V3 a, float s) { return make3(a.x * s, a.y * s, a.z * s); }
MC_HD float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct alignas(16) F4 { float x, y, z, w; };
#if !defined(__CUDA_ARCH__)
MC_HD int max(int a, int b) { return a > b ? a : b; }
MC_HD int min(int a, int b) { return a < b ? a : b; }
#endif

// ---------------------------------------------------------------------------
// Philox4x32-10 counter based RNG (Salmon et al., SC'11).
// Every history owns an independent stream keyed by (seed) and counted by
// (draw, history, voxel, rotation): results do not depend on how work is
// split across threads, launches, GPUs or machines.
// ---------------------------------------------------------------------------
struct Philox {
    uint32_t c0, c1, c2, c3, k0, k1;
    uint32_t b[4];
    int idx;

    MC_HD void init(uint64_t seed, uint32_t history, uint32_t voxel, uint32_t rotation) {
        k0 = (uint32_t)seed; k1 = (uint32_t)(seed >> 32);
        c0 = 0; c1 = history; c2 = voxel; c3 = rotation;
        idx = 4;
    }
    MC_HD static uint32_t mulhilo(uint32_t a, uint32_t x, uint32_t& hi) {
#if defined(__CUDA_ARCH__)
        hi = __umulhi(a, x);
        return a * x;
#else
        uint64_t p = (uint64_t)a * (uint64_t)x;
        hi = (uint32_t)(p >> 32);
        return (uint32_t)p;
#endif
    }
    MC_HD static void block(uint32_t in0, uint32_t in1, uint32_t in2, uint32_t in3,
                            uint32_t key0, uint32_t key1, uint32_t* out) {
        uint32_t x0 = in0, x1 = in1, x2 = in2, x3 = in3;
        for (int r = 0; r < 10; ++r) {
            uint32_t hi0, hi1;
            uint32_t lo0 = mulhilo(0xD2511F53u, x0, hi0);
            uint32_t lo1 = mulhilo(0xCD9E8D57u, x2, hi1);
            uint32_t n0 = hi1 ^ x1 ^ key0;
            uint32_t n2 = hi0 ^ x3 ^ key1;
            x0 = n0; x1 = lo1; x2 = n2; x3 = lo0;
            key0 += 0x9E3779B9u; key1 += 0xBB67AE85u;
        }
        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
    }
    MC_HD uint32_t next() {
        if (idx >= 4) { block(c0, c1, c2, c3, k0, k1, b); ++c0; idx = 0; }
        return b[idx++];
    }
    // uniform in (0, 1]
    MC_HD float uniform() { return (float)((next() >> 8) + 1u) * (1.0f / 16777216.0f); }
    MC_HD float normal() {  // Box-Muller, one value
        float u1 = uniform(), u2 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * kPi * u2);
    }
};

// ---------------------------------------------------------------------------
// Material tables (device or host pointers, filled by the host setup)
// ---------------------------------------------------------------------------
struct Materials {
    int nMat, nE;
    float logEmin, invDlogE;           // log-spaced energy grid
    const float* muPE;                 // [nMat*nE]   1/mm
    const float* muInc;                // [nMat*nE]
    const float* muCoh;                // [nMat*nE]
    const float* muTot;                // [nMat*nE]
    const float* regMaj;               // [nRegions*nE] region majorants
    const int* nFluo;                  // [nMat]
    const float* fluoE;                // [nMat*kMaxFluo] line energies (keV)
    const float* fluoP;                // [(nMat*nE)*kMaxFluo] probability per photo-absorption
    int nFF;                           // Rayleigh form-factor grid (log-spaced in x^2)
    float ffX2min, ffLogX2min, ffInvDlog;
    const float* ffX2;                 // [nFF]
    const float* ffCum;                // [nMat*nFF] normalised cumulative of F^2 d(x^2)
    const int* hasFF;                  // [nMat]
};

struct Geometry {
    // collimator layers (axis-aligned plates, holes along +y)
    int nLayers;
    float layXh[kMaxLayers], layY1[kMaxLayers], layY2[kMaxLayers], layZh[kMaxLayers];
    int layMat[kMaxLayers];
    // what fills the plate's y-range outside the plate: 0 nothing, 1 perfect absorber,
    // 2 the layer material up to |x| <= shXh, |z| <= shZh (plate embedded in a frame)
    int shieldMode;
    float shXh[kMaxLayers], shZh[kMaxLayers];
    float hgOx[kMaxLayers], hgOz[kMaxLayers], hgInvCell[kMaxLayers];
    int hgNx[kMaxLayers], hgNz[kMaxLayers], hgBase[kMaxLayers];
    const int* hgCellStart;            // CSR over all layers' cells (+1 entry)
    const int* hgItems;
    const F4* hole0;                   // x, z, r^2, y1
    const F4* hole1;                   // y2, r, 0, 0
    const int* holeMat;                // material or -1 (vacuum)
    // crystals (boxes rotated about y)
    int nDet;
    const F4* xtal0;                   // cx, cy, cz, cos(rot)
    const F4* xtal1;                   // half width (x), half thickness (y), half height (z), sin(rot)
    const int* xtalMat;                // material or -1 (disabled)
    const float* xtalEres;             // FWHM energy resolution at E0 (fraction)
    float cgOx, cgOy, cgOz, cgInvCell;
    int cgNx, cgNy, cgNz;
    const int* cgCellStart;
    const int* cgItems;
    // regions for majorant segmentation
    int nRegions;
    float regMin[kMaxRegions][3], regMax[kMaxRegions][3];
    int regLayer[kMaxRegions];         // collimator layer index or -1 (detector region)
    // bounding boxes for direction biasing: everything that can interact / enabled crystals
    float asmMin[3], asmMax[3];
    float detMin[3], detMax[3];
    // hole-aperture targeting: layer whose holes are aimed at (-1 = off), the largest
    // acceptance tan(theta) of its channels and the largest hole radius
    int tgtLayer, tgtHoleStart, tgtHoleCount;   // target layer and its hole index range
    float tgtAcc, tgtTan, tgtRmax;   // acceptance factor x 2r/L, its maximum over holes, max radius
};

struct SimConsts {
    int nX, nY, nZ;
    float wX, wY, wZ, shX, shY, shZ, angPerRot;
    float E0;
    uint64_t seed;
    int srcUniform;                    // 1: uniform in voxel, 0: voxel centre
    int dirRect;                       // 1: importance sampling towards the assembly, 0: 4pi
    float detBias;                     // mixture weight of the crystal-block rectangle
    float holeBias;                    // mixture weight of the hole-aperture component
    float tgtMargin;                   // aperture square half size = r * (1 + tgtMargin)
    int primaryOnly;                   // 1: analytic-equivalent mode (any scatter kills, no fluo)
    int singleScatter;                 // 1: first-order scatter reference (one Compton, PE scoring only)
    int fluo;
    float eCut;
    // detector response
    int winMode;
    float eresExp;                     // R(E) = R0 * (E0/E)^eresExp
    int nWin;
    float winLo[kMaxWindows], winHi[kMaxWindows];
    int winAuto[kMaxWindows];          // 1: per crystal (1 -/+ R0/2) * E0
    int components;                    // 1: tally the 3 event classes separately
    int nSpecBins;
    float specBinW;
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
struct Photon {
    V3 p, d, inv;
    float E, ef;
    int ei, flags, firstXtal;
};

MC_HD V3 invDir(V3 d) {
    return make3(fabsf(d.x) > 1e-20f ? 1.0f / d.x : (d.x >= 0 ? kBig : -kBig),
                 fabsf(d.y) > 1e-20f ? 1.0f / d.y : (d.y >= 0 ? kBig : -kBig),
                 fabsf(d.z) > 1e-20f ? 1.0f / d.z : (d.z >= 0 ? kBig : -kBig));
}

MC_HD void setEnergy(const Materials& M, Photon& ph, float E) {
    ph.E = E;
    float t = (logf(E) - M.logEmin) * M.invDlogE;
    float tmax = (float)(M.nE - 1) - 1e-4f;
    t = fminf(fmaxf(t, 0.0f), tmax);
    ph.ei = (int)t;
    ph.ef = t - (float)ph.ei;
}

MC_HD float tab(const float* t, int base, const Photon& ph) {
    float a = t[base + ph.ei], b = t[base + ph.ei + 1];
    return a + ph.ef * (b - a);
}
MC_HD float muTot(const Materials& M, int m, const Photon& ph) { return tab(M.muTot, m * M.nE, ph); }

MC_HD void setDir(Photon& ph, V3 d) { ph.d = d; ph.inv = invDir(d); }

// slab test; returns entry/exit distances (t0 > t1 or t0 >= kBig/2: miss)
MC_HD void slabs(const float* mn, const float* mx, V3 p, V3 inv, float& t0, float& t1) {
    float ax = (mn[0] - p.x) * inv.x, bx = (mx[0] - p.x) * inv.x;
    float ay = (mn[1] - p.y) * inv.y, by = (mx[1] - p.y) * inv.y;
    float az = (mn[2] - p.z) * inv.z, bz = (mx[2] - p.z) * inv.z;
    t0 = fmaxf(fmaxf(fminf(ax, bx), fminf(ay, by)), fminf(az, bz));
    t1 = fminf(fminf(fmaxf(ax, bx), fmaxf(ay, by)), fmaxf(az, bz));
}

// direction from polar cosine/azimuth relative to d (PENELOPE convention)
MC_HD V3 rotateDir(V3 d, float cosT, float phi) {
    float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
    float cp = cosf(phi), sp = sinf(phi);
    V3 n;
    if (fabsf(d.z) > 0.99999f) {
        n = make3(sinT * cp, sinT * sp, d.z > 0 ? cosT : -cosT);
    } else {
        float t = sqrtf(1.0f - d.z * d.z);
        n.x = d.x * cosT + sinT * (d.x * d.z * cp - d.y * sp) / t;
        n.y = d.y * cosT + sinT * (d.y * d.z * cp + d.x * sp) / t;
        n.z = d.z * cosT - t * sinT * cp;
    }
    float l = 1.0f / sqrtf(dot(n, n));
    return n * l;
}

MC_HD V3 isotropic(Philox& rng) {
    float c = 2.0f * rng.uniform() - 1.0f;
    float s = sqrtf(fmaxf(0.0f, 1.0f - c * c));
    float ph = 2.0f * kPi * rng.uniform();
    return make3(s * cosf(ph), s * sinf(ph), c);
}

// ---------------------------------------------------------------------------
// Interaction sampling
// ---------------------------------------------------------------------------
// Klein-Nishina (Butcher & Messel / Geant4 G4KleinNishinaCompton).
// Returns the scattered photon energy and the cosine of the scattering angle.
MC_HD float sampleKleinNishina(float E, Philox& rng, float& cosT) {
    float k = E / kElectronMass;
    float eps0 = 1.0f / (1.0f + 2.0f * k);
    float eps0sq = eps0 * eps0;
    float a1 = -logf(eps0);
    float a2 = a1 + 0.5f * (1.0f - eps0sq);
    float eps, epssq, onecost, greject;
    int guard = 0;
    do {
        if (a1 > a2 * rng.uniform()) {
            eps = expf(-a1 * rng.uniform());
            epssq = eps * eps;
        } else {
            epssq = eps0sq + (1.0f - eps0sq) * rng.uniform();
            eps = sqrtf(epssq);
        }
        onecost = (1.0f - eps) / (eps * k);
        float sint2 = onecost * (2.0f - onecost);
        greject = 1.0f - eps * sint2 / (1.0f + epssq);
    } while (greject < rng.uniform() && ++guard < 1000);
    cosT = fminf(1.0f, fmaxf(-1.0f, 1.0f - onecost));
    return eps * E;
}

MC_HD float ffCumAt(const Materials& M, int m, float x2) {
    const float* c = M.ffCum + m * M.nFF;
    if (x2 <= M.ffX2min) return c[0] * x2 / M.ffX2min;
    float t = (logf(x2) - M.ffLogX2min) * M.ffInvDlog;
    int k = (int)t;
    if (k >= M.nFF - 1) return c[M.nFF - 1];
    float xa = M.ffX2[k], xb = M.ffX2[k + 1];
    return c[k] + (c[k + 1] - c[k]) * (x2 - xa) / (xb - xa);
}

MC_HD float ffInvCum(const Materials& M, int m, float A) {
    const float* c = M.ffCum + m * M.nFF;
    if (A <= c[0]) return M.ffX2min * A / c[0];
    int lo = 0, hi = M.nFF - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (c[mid] <= A) lo = mid; else hi = mid;
    }
    float den = c[hi] - c[lo];
    float t = den > 0.0f ? (A - c[lo]) / den : 0.0f;
    return M.ffX2[lo] + t * (M.ffX2[hi] - M.ffX2[lo]);
}

// Rayleigh: x = sin(theta/2)/lambda; sample x^2 from F^2(x) then accept with
// the Thomson factor (1+cos^2)/2 (PENELOPE / EGSnrc method).
MC_HD float sampleRayleigh(const Materials& M, int m, float E, Philox& rng) {
    if (!M.hasFF[m]) {  // pure Thomson
        for (int i = 0; i < 1000; ++i) {
            float c = 2.0f * rng.uniform() - 1.0f;
            if (2.0f * rng.uniform() <= 1.0f + c * c) return c;
        }
        return 1.0f;
    }
    float xmax = E / kHcKeVA;
    float xmax2 = xmax * xmax;
    float Amax = ffCumAt(M, m, xmax2);
    for (int i = 0; i < 1000; ++i) {
        float x2 = ffInvCum(M, m, rng.uniform() * Amax);
        float c = 1.0f - 2.0f * x2 / xmax2;
        c = fminf(1.0f, fmaxf(-1.0f, c));
        if (2.0f * rng.uniform() <= 1.0f + c * c) return c;
    }
    return 1.0f;
}

// ---------------------------------------------------------------------------
// Geometry queries
// ---------------------------------------------------------------------------
MC_HD int holeAt(const Geometry& G, int m, float x, float y, float z) {
    int cx = (int)floorf((x - G.hgOx[m]) * G.hgInvCell[m]);
    int cz = (int)floorf((z - G.hgOz[m]) * G.hgInvCell[m]);
    if (cx < 0 || cz < 0 || cx >= G.hgNx[m] || cz >= G.hgNz[m]) return -1;
    int cell = G.hgBase[m] + cx * G.hgNz[m] + cz;
    int e = G.hgCellStart[cell + 1];
    for (int k = G.hgCellStart[cell]; k < e; ++k) {
        int h = G.hgItems[k];
        F4 a = G.hole0[h];
        float dx = x - a.x, dz = z - a.y;
        if (dx * dx + dz * dz < a.z && y >= a.w && y <= G.hole1[h].x) return h;
    }
    return -1;
}

// distance to leave hole h (cylinder along y) from inside
MC_HD float holeExit(const Geometry& G, int h, V3 p, V3 d) {
    F4 a = G.hole0[h];
    float y2 = G.hole1[h].x;
    float t = kBig;
    if (d.y > 1e-12f) t = (y2 - p.y) / d.y;
    else if (d.y < -1e-12f) t = (a.w - p.y) / d.y;
    float A = d.x * d.x + d.z * d.z;
    if (A > 1e-12f) {
        float ox = p.x - a.x, oz = p.z - a.y;
        float B = ox * d.x + oz * d.z;
        float C = ox * ox + oz * oz - a.z;
        float disc = fmaxf(B * B - A * C, 0.0f);
        float tr = (-B + sqrtf(disc)) / A;
        t = fminf(t, fmaxf(tr, 0.0f));
    }
    return t;
}

MC_HD int crystalAt(const Geometry& G, V3 p) {
    int cx = (int)floorf((p.x - G.cgOx) * G.cgInvCell);
    int cy = (int)floorf((p.y - G.cgOy) * G.cgInvCell);
    int cz = (int)floorf((p.z - G.cgOz) * G.cgInvCell);
    if (cx < 0 || cy < 0 || cz < 0 || cx >= G.cgNx || cy >= G.cgNy || cz >= G.cgNz) return -1;
    int cell = (cx * G.cgNy + cy) * G.cgNz + cz;
    int e = G.cgCellStart[cell + 1];
    for (int k = G.cgCellStart[cell]; k < e; ++k) {
        int c = G.cgItems[k];
        F4 a = G.xtal0[c], b = G.xtal1[c];
        float dx = p.x - a.x, dz = p.z - a.z;
        float lx = dx * a.w + dz * b.w;
        float lz = -dx * b.w + dz * a.w;
        float ly = p.y - a.y;
        if (fabsf(lx) <= b.x && fabsf(ly) <= b.y && fabsf(lz) <= b.z) return c;
    }
    return -1;
}

// kind: 0 nothing, 1 collimator (plate or hole), 2 crystal (idx = crystal), 3 black shield
MC_HD int locate(const Geometry& G, V3 p, int& kind, int& idx) {
    idx = -1;
    for (int m = 0; m < G.nLayers; ++m) {
        if (p.y < G.layY1[m] || p.y > G.layY2[m]) continue;
        if (fabsf(p.x) > G.layXh[m] || fabsf(p.z) > G.layZh[m]) {
            if (G.shieldMode == 1) { kind = 3; return -1; }
            if (G.shieldMode == 2 && fabsf(p.x) <= G.shXh[m] && fabsf(p.z) <= G.shZh[m]) { kind = 1; return G.layMat[m]; }
            continue;
        }
        kind = 1;
        int h = holeAt(G, m, p.x, p.y, p.z);
        return h >= 0 ? G.holeMat[h] : G.layMat[m];
    }
    int c = crystalAt(G, p);
    if (c >= 0) { kind = 2; idx = c; return G.xtalMat[c]; }
    kind = 0;
    return -1;
}

// ---------------------------------------------------------------------------
// History bookkeeping
// ---------------------------------------------------------------------------
struct Deposits {
    int n;
    int det[kMaxDeposits];
    float e[kMaxDeposits];
    int cls[kMaxDeposits];
};

struct Secondary { V3 p; float E; int flags, firstXtal; };
struct Stack { int n; Secondary s[kMaxStack]; };

MC_HD int classOf(const Photon& ph, int xtal) {
    if (ph.flags & kFlagCollimator) return kCollimatorScatter;
    if (ph.firstXtal >= 0 && ph.firstXtal != xtal) return kCrystalScatter;
    return kPrimary;
}

MC_HD void addDeposit(Deposits& D, int det, float e, int cls) {
    for (int i = 0; i < D.n; ++i)
        if (D.det[i] == det) { D.e[i] += e; return; }
    if (D.n < kMaxDeposits) { D.det[D.n] = det; D.e[D.n] = e; D.cls[D.n] = cls; ++D.n; }
}

// Optional vertex recorder (used for the visualiser's track overlay)
struct Tracer {
    int active;
    int track;
    float* buf;          // 6 floats per vertex: x y z E type track
    int* count;
    int capacity;
    MC_HD void vertex(V3 p, float E, int type) const {
        if (!active) return;
#if defined(__CUDA_ARCH__)
        int i = atomicAdd(count, 1);
#else
        int i = (*count)++;
#endif
        if (i >= capacity) return;
        float* v = buf + 6 * i;
        v[0] = p.x; v[1] = p.y; v[2] = p.z; v[3] = E; v[4] = (float)type; v[5] = (float)track;
    }
};

// ---------------------------------------------------------------------------
// Transport of one photon until absorption / escape.
// ---------------------------------------------------------------------------
MC_HD void transport(const Geometry& G, const Materials& M, const SimConsts& C,
                     Photon ph, Philox& rng, Deposits& D, Stack& S, const Tracer& tr) {
    for (int step = 0; step < kMaxSteps; ++step) {
        // ---- majorant and distance to the next region boundary
        float tB = kBig, mu = 0.0f;
        for (int r = 0; r < G.nRegions; ++r) {
            float t0, t1;
            slabs(G.regMin[r], G.regMax[r], ph.p, ph.inv, t0, t1);
            if (t1 <= t0 || t1 <= 0.0f || t0 >= 0.5f * kBig) continue;
            if (t0 > 0.0f) { tB = fminf(tB, t0); continue; }   // region ahead
            tB = fminf(tB, t1);                                 // inside region
            float muR;
            int m = G.regLayer[r];
            if (m >= 0) {
                if (G.shieldMode == 1) {
                    if (fabsf(ph.p.x) > G.layXh[m] || fabsf(ph.p.z) > G.layZh[m]) {
                        tr.vertex(ph.p, ph.E, kVtxKilled);
                        return;                                 // absorbed by the shield
                    }
                    // the plate's side faces bound this flight: never cross the shield unseen
                    if (ph.d.x > 1e-12f) tB = fminf(tB, (G.layXh[m] - ph.p.x) / ph.d.x);
                    else if (ph.d.x < -1e-12f) tB = fminf(tB, (-G.layXh[m] - ph.p.x) / ph.d.x);
                    if (ph.d.z > 1e-12f) tB = fminf(tB, (G.layZh[m] - ph.p.z) / ph.d.z);
                    else if (ph.d.z < -1e-12f) tB = fminf(tB, (-G.layZh[m] - ph.p.z) / ph.d.z);
                }
                int h = holeAt(G, m, ph.p.x, ph.p.y, ph.p.z);
                if (h >= 0) {
                    int hm = G.holeMat[h];
                    muR = hm >= 0 ? muTot(M, hm, ph) : 0.0f;
                    tB = fminf(tB, holeExit(G, h, ph.p, ph.d));
                } else {
                    muR = tab(M.regMaj, r * M.nE, ph);
                }
            } else {
                muR = tab(M.regMaj, r * M.nE, ph);
            }
            mu = fmaxf(mu, muR);
        }
        if (tB >= 0.5f * kBig) {                                // nothing ahead: escape
            tr.vertex(ph.p + ph.d * 30.0f, ph.E, kVtxEscape);
            return;
        }
        if (mu <= 0.0f) { ph.p = ph.p + ph.d * (tB + kPush); continue; }
        float s = -logf(rng.uniform()) / mu;
        if (s >= tB) { ph.p = ph.p + ph.d * (tB + kPush); continue; }
        ph.p = ph.p + ph.d * s;

        // ---- real or virtual collision
        int kind, idx;
        int mat = locate(G, ph.p, kind, idx);
        if (kind == 3) { tr.vertex(ph.p, ph.E, kVtxKilled); return; }
        if (mat < 0) continue;
        float muL = muTot(M, mat, ph);
        if (rng.uniform() * mu > muL) continue;

        if (kind == 1) ph.flags |= kFlagCollimator;
        else if (kind == 2 && ph.firstXtal < 0) ph.firstXtal = idx;

        int base = mat * M.nE;
        float pe = tab(M.muPE, base, ph);
        float inc = tab(M.muInc, base, ph);
        float r = rng.uniform() * muL;

        if (r <= pe) {                                          // photoelectric
            float eLocal = ph.E;
            int nf = M.nFluo[mat];
            if (C.fluo && nf > 0) {
                float u = rng.uniform(), acc = 0.0f;
                const float* pa = M.fluoP + (size_t)(base + ph.ei) * kMaxFluo;
                const float* pb = pa + kMaxFluo;
                for (int l = 0; l < nf; ++l) {
                    acc += pa[l] + ph.ef * (pb[l] - pa[l]);
                    if (u <= acc) {
                        float ef = M.fluoE[mat * kMaxFluo + l];
                        if (ef >= C.eCut && ef < eLocal && S.n < kMaxStack) {
                            Secondary& sc = S.s[S.n++];
                            sc.p = ph.p; sc.E = ef; sc.flags = ph.flags; sc.firstXtal = ph.firstXtal;
                            eLocal -= ef;
                        }
                        break;
                    }
                }
            }
            // single-scatter mode: absorption in the crystal where the photon scattered belongs to
            // no analytic term (the analytic PE matrix counts photoelectric first interactions only)
            bool sameXtal = C.singleScatter && (ph.flags & kFlagXtalScatter) && ph.firstXtal == idx;
            if (kind == 2 && !sameXtal) addDeposit(D, idx, eLocal, classOf(ph, idx));
            tr.vertex(ph.p, ph.E, kVtxPhoto);
            return;
        }
        if (C.primaryOnly) { tr.vertex(ph.p, ph.E, kVtxKilled); return; }
        // single-scatter mode: a second scatter or any Rayleigh scatter ends the history
        if (C.singleScatter && ((ph.flags & kFlagScattered) || r > pe + inc)) { tr.vertex(ph.p, ph.E, kVtxKilled); return; }

        if (r <= pe + inc) {                                    // Compton
            float cosT;
            float E1 = sampleKleinNishina(ph.E, rng, cosT);
            if (kind == 2 && !C.singleScatter) addDeposit(D, idx, ph.E - E1, classOf(ph, idx));
            ph.flags |= kFlagScattered | (kind == 2 ? kFlagXtalScatter : 0);
            setDir(ph, rotateDir(ph.d, cosT, 2.0f * kPi * rng.uniform()));
            setEnergy(M, ph, E1);
            tr.vertex(ph.p, E1, kVtxCompton);
        } else {                                                // Rayleigh
            float cosT = sampleRayleigh(M, mat, ph.E, rng);
            setDir(ph, rotateDir(ph.d, cosT, 2.0f * kPi * rng.uniform()));
            tr.vertex(ph.p, ph.E, kVtxRayleigh);
        }
        if (ph.E < C.eCut) {
            if (kind == 2 && !C.singleScatter) addDeposit(D, idx, ph.E, classOf(ph, idx));
            tr.vertex(ph.p, ph.E, kVtxCutoff);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Source: emission point in a (rotated) voxel, biased direction + weight.
//
// Direction biasing (direction_sampling = rect): a defensive mixture of two
// "projected rectangle" densities,
//   A: the rectangle that the assembly box (collimators + crystals) projects
//      onto its front plane - covers every ray that can interact at all;
//   B: the same for the box around the enabled crystals.
// With probability detBias the direction is aimed uniformly at B, otherwise
// at A; the weight is 1 / (4 pi p_mix(omega)), so the estimator stays unbiased
// (collimator scatter from anywhere on the plates is still sampled through A).
// ---------------------------------------------------------------------------
struct ProjRect { float y, xlo, xhi, zlo, zhi, area; int ok; };

MC_HD ProjRect projectBox(const float* mn, const float* mx, V3 p) {
    ProjRect r;
    r.y = mn[1];
    r.ok = p.y < r.y - 1e-3f;
    if (!r.ok) { r.xlo = r.xhi = r.zlo = r.zhi = r.area = 0.0f; return r; }
    float s = (r.y - p.y) / (mx[1] - p.y);
    r.xlo = fminf(mn[0], p.x + (mn[0] - p.x) * s);
    r.xhi = fmaxf(mx[0], p.x + (mx[0] - p.x) * s);
    r.zlo = fminf(mn[2], p.z + (mn[2] - p.z) * s);
    r.zhi = fmaxf(mx[2], p.z + (mx[2] - p.z) * s);
    r.area = (r.xhi - r.xlo) * (r.zhi - r.zlo);
    return r;
}

// solid-angle density of "aim uniformly at rectangle R" evaluated for direction d
MC_HD float rectPdf(const ProjRect& R, V3 p, V3 d) {
    if (!R.ok || d.y <= 0.0f || R.area <= 0.0f) return 0.0f;
    float t = (R.y - p.y) / d.y;
    float x = p.x + d.x * t, z = p.z + d.z * t;
    if (x < R.xlo || x > R.xhi || z < R.zlo || z > R.zhi) return 0.0f;
    return t * t / (R.area * d.y);
}

MC_HD V3 aimAt(const ProjRect& R, V3 p, Philox& rng) {
    V3 q = make3(R.xlo + rng.uniform() * (R.xhi - R.xlo), R.y, R.zlo + rng.uniform() * (R.zhi - R.zlo));
    V3 D = q - p;
    return D * (1.0f / sqrtf(dot(D, D)));
}

MC_HD V3 voxelCentre(const SimConsts& C, int voxel, int& ix, int& iy, int& iz) {
    int nxy = C.nX * C.nY;
    iz = voxel / nxy;
    int rem = voxel - iz * nxy;
    iy = rem / C.nX;
    ix = rem - iy * C.nX;
    return make3((ix - C.nX / 2.0f + 0.5f) * C.wX + C.shX,
                 (iy - C.nY / 2.0f + 0.5f) * C.wY + C.shY,
                 (iz - C.nZ / 2.0f + 0.5f) * C.wZ + C.shZ);
}

// ---------------------------------------------------------------------------
// Hole-aperture targeting. For every voxel the holes of the target layer whose
// straight channel can be traversed from (near) the voxel are listed once; a
// mixture component then aims photons uniformly at a square around the entrance
// aperture of one of them. The list depends only on (voxel, rotation), so results
// stay independent of how work is split; it only changes the sampling density,
// which enters the weight, so the estimator stays unbiased.
// ---------------------------------------------------------------------------
struct HoleTargets { int n; int id[kMaxTargets]; };

MC_HD V3 voxelWorld(const SimConsts& C, int voxel, int rot) {
    int ix, iy, iz;
    V3 v = voxelCentre(C, voxel, ix, iy, iz);
    float a = rot * C.angPerRot, ca = cosf(a), sa = sinf(a);
    return make3(v.x * ca - v.y * sa, v.x * sa + v.y * ca, v.z);
}

MC_HD void buildHoleTargets(const Geometry& G, const SimConsts& C, int voxel, int rot, HoleTargets& T) {
    T.n = 0;
    const int m = G.tgtLayer;
    if (!C.dirRect || C.holeBias <= 0.0f || m < 0) return;
    V3 V = voxelWorld(C, voxel, rot);
    float depth = G.layY1[m] - V.y;
    if (depth <= 0.0f) return;
    float slack = 0.5f * sqrtf(C.wX * C.wX + C.wY * C.wY + C.wZ * C.wZ);
    float Rb = depth * G.tgtTan + G.tgtRmax + slack;   // bound on |hole centre - voxel| (x-z)
    float inv = G.hgInvCell[m];
    int x0 = max(0, (int)floorf((V.x - Rb - G.hgOx[m]) * inv)), x1 = min(G.hgNx[m] - 1, (int)floorf((V.x + Rb - G.hgOx[m]) * inv));
    int z0 = max(0, (int)floorf((V.z - Rb - G.hgOz[m]) * inv)), z1 = min(G.hgNz[m] - 1, (int)floorf((V.z + Rb - G.hgOz[m]) * inv));
    if (x1 < x0 || z1 < z0) return;
    float score[kMaxTargets];
    auto consider = [&](int h) {
        F4 a = G.hole0[h];
        float r = G.hole1[h].y;
        float len = fminf(G.hole1[h].x, G.layY2[m]) - fmaxf(a.w, G.layY1[m]);
        float tanAcc = len > 1e-3f ? fminf(G.tgtTan, G.tgtAcc * 2.0f * r / len) : G.tgtTan;
        float dx = a.x - V.x, dz = a.y - V.z;
        float lim = (fmaxf(a.w, G.layY1[m]) - V.y) * tanAcc + r + slack;
        float d = sqrtf(dx * dx + dz * dz) / lim;           // < 1: channel reachable
        if (d >= 1.0f) return;
        if (T.n < kMaxTargets) { T.id[T.n] = h; score[T.n] = d; ++T.n; return; }
        int worst = 0;                                      // keep the best aligned holes
        for (int i = 1; i < kMaxTargets; ++i) if (score[i] > score[worst]) worst = i;
        if (d < score[worst] || (d == score[worst] && h < T.id[worst])) { T.id[worst] = h; score[worst] = d; }
    };
    long long nCells = (long long)(x1 - x0 + 1) * (z1 - z0 + 1);
    if (nCells > G.tgtHoleCount) {                          // wide acceptance (thin plate / pinholes)
        if (G.tgtHoleCount > 65536) return;
        for (int h = G.tgtHoleStart; h < G.tgtHoleStart + G.tgtHoleCount; ++h) consider(h);
        return;
    }
    for (int cx = x0; cx <= x1; ++cx)
        for (int cz = z0; cz <= z1; ++cz) {
            int cell = G.hgBase[m] + cx * G.hgNz[m] + cz;
            for (int k = G.hgCellStart[cell]; k < G.hgCellStart[cell + 1]; ++k) {
                int h = G.hgItems[k];
                F4 a = G.hole0[h];
                // count each hole once: in the cell that contains its centre
                if ((int)floorf((a.x - G.hgOx[m]) * inv) != cx || (int)floorf((a.y - G.hgOz[m]) * inv) != cz) continue;
                consider(h);
            }
        }
}

MC_HD void holeSquare(const Geometry& G, const SimConsts& C, int h, float& y, float& cx, float& cz, float& half) {
    F4 a = G.hole0[h];
    y = fmaxf(a.w, G.layY1[G.tgtLayer]);
    cx = a.x; cz = a.y;
    half = G.hole1[h].y * (1.0f + C.tgtMargin);
}

MC_HD float holesPdf(const Geometry& G, const SimConsts& C, const HoleTargets& T, V3 p, V3 d) {
    if (d.y <= 0.0f) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < T.n; ++i) {
        float y, cx, cz, hf;
        holeSquare(G, C, T.id[i], y, cx, cz, hf);
        float t = (y - p.y) / d.y;
        if (t <= 0.0f) continue;
        if (fabsf(p.x + d.x * t - cx) <= hf && fabsf(p.z + d.z * t - cz) <= hf) sum += t * t / (4.0f * hf * hf * d.y);
    }
    return sum / (float)T.n;
}

MC_HD float sampleSource(const Geometry& G, const SimConsts& C, int voxel, int rot, const HoleTargets* T,
                         Philox& rng, V3& pos, V3& dir) {
    int ix, iy, iz;
    V3 v = voxelCentre(C, voxel, ix, iy, iz);
    if (C.srcUniform) {
        v.x += (rng.uniform() - 0.5f) * C.wX;
        v.y += (rng.uniform() - 0.5f) * C.wY;
        v.z += (rng.uniform() - 0.5f) * C.wZ;
    }
    float a = rot * C.angPerRot;
    float ca = cosf(a), sa = sinf(a);
    pos = make3(v.x * ca - v.y * sa, v.x * sa + v.y * ca, v.z);

    ProjRect A = projectBox(G.asmMin, G.asmMax, pos);
    if (!C.dirRect || !A.ok) {
        dir = isotropic(rng);
        return 1.0f;
    }
    ProjRect B = projectBox(G.detMin, G.detMax, pos);
    // mixture weights: holes (if any are reachable), crystal block, whole assembly (defensive)
    float wH = (T && T->n > 0 && G.tgtLayer >= 0 && pos.y < G.layY1[G.tgtLayer]) ? C.holeBias : 0.0f;
    float wB = B.ok ? C.detBias : 0.0f;
    float u = rng.uniform();
    if (u < wH) {
        int k = min((int)(rng.uniform() * T->n), T->n - 1);
        float y, cx, cz, hf;
        holeSquare(G, C, T->id[k], y, cx, cz, hf);
        V3 q = make3(cx + (2.0f * rng.uniform() - 1.0f) * hf, y, cz + (2.0f * rng.uniform() - 1.0f) * hf);
        V3 D = q - pos;
        dir = D * (1.0f / sqrtf(dot(D, D)));
    } else if (u < wH + wB) {
        dir = aimAt(B, pos, rng);
    } else {
        dir = aimAt(A, pos, rng);
    }
    float pdf = (1.0f - wH - wB) * rectPdf(A, pos, dir) + wB * rectPdf(B, pos, dir);
    if (wH > 0.0f) pdf += wH * holesPdf(G, C, *T, pos, dir);
    return pdf > 0.0f ? 1.0f / (4.0f * kPi * pdf) : 0.0f;
}

// One full history (primary + fluorescence secondaries). Returns the weight.
MC_HD float runHistory(const Geometry& G, const Materials& M, const SimConsts& C, const HoleTargets* T,
                       int voxel, int rot, Philox& rng, Deposits& D, Tracer& tr) {
    D.n = 0;
    Stack S;
    S.n = 0;
    Photon ph;
    V3 pos, dir;
    float w = sampleSource(G, C, voxel, rot, T, rng, pos, dir);
    ph.p = pos;
    setDir(ph, dir);
    setEnergy(M, ph, C.E0);
    ph.flags = 0;
    ph.firstXtal = -1;
    tr.vertex(pos, C.E0, kVtxEmit);
    transport(G, M, C, ph, rng, D, S, tr);
    int sec = 0;
    while (S.n > 0) {
        Secondary s = S.s[--S.n];
        tr.track += 1; ++sec;
        Photon q;
        q.p = s.p;
        setDir(q, isotropic(rng));
        setEnergy(M, q, s.E);
        q.flags = s.flags;
        q.firstXtal = s.firstXtal;
        tr.vertex(q.p, q.E, kVtxFluoEmit);
        transport(G, M, C, q, rng, D, S, tr);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Detector response: probability that a deposit E in crystal c falls in window k
// ---------------------------------------------------------------------------
MC_HD float eSigma(const SimConsts& C, float R0, float E) {
    // relative FWHM R(E) = R0 (E0/E)^a  ->  sigma = R0 E0^a E^(1-a) / 2.35482
    return R0 * powf(C.E0 / E, C.eresExp) * E / 2.35482f;
}

MC_HD float windowProb(const SimConsts& C, float R0, int k, float E) {
    if (C.winMode == kWinNone) return E > 0.0f ? 1.0f : 0.0f;
    float lo = C.winLo[k], hi = C.winHi[k];
    if (C.winAuto[k]) { lo = (1.0f - 0.5f * R0) * C.E0; hi = (1.0f + 0.5f * R0) * C.E0; }
    if (C.winMode == kWinIdeal || R0 <= 0.0f) return (E >= lo && E <= hi) ? 1.0f : 0.0f;
    float s = eSigma(C, R0, E) * 1.41421356f;
    return 0.5f * (erff((hi - E) / s) - erff((lo - E) / s));
}

// Tally accumulation. The GPU kernel adds into float buffers with atomics and the
// host flushes them into double accumulators after every launch (the number of
// histories per voxel and launch is capped so float sums never reach the regime
// where increments are lost); the CPU backend accumulates in double directly.
MC_HD void addF(float* a, float v) {
#if defined(__CUDA_ARCH__)
    atomicAdd(a, v);
#else
    *a += v;
#endif
}
MC_HD void addF(double* a, float v) { *a += (double)v; }  // CPU only (thread-private buffers)

// Score one history into sysmat tallies and the energy spectrum.
//   sum  [(t*nDet + det)*stride + col]   t = window*(1 or 3) + class
//   spec [(blur*3 + class)*nSpecBins + bin]  (blur = 0 raw deposit, 1 blurred)
template <class T>
MC_HD void scoreHistory(const Geometry& G, const SimConsts& C, const Deposits& D, float w,
                        Philox& rng, T* sum, T* sum2, size_t stride, int col, T* spec) {
    int nCls = C.components ? kNumClasses : 1;
    for (int i = 0; i < D.n; ++i) {
        float E = D.e[i];
        if (E <= 0.0f) continue;
        int c = D.det[i], cls = D.cls[i];
        float R0 = G.xtalEres[c];
        for (int k = 0; k < C.nWin; ++k) {
            float p = windowProb(C, R0, k, E);
            if (p <= 0.0f) continue;
            float v = w * p;
            int t = k * nCls + (C.components ? cls : 0);
            size_t at = ((size_t)t * G.nDet + c) * stride + col;
            addF(sum + at, v);
            if (sum2) addF(sum2 + at, v * v);
        }
        if (spec) {
            int b = (int)(E / C.specBinW);
            if (b < C.nSpecBins) addF(spec + cls * C.nSpecBins + b, w);
            float Eb = E;
            if (C.winMode == kWinGauss && R0 > 0.0f) Eb += eSigma(C, R0, E) * rng.normal();
            int bb = (int)floorf(Eb / C.specBinW);
            if (bb >= 0 && bb < C.nSpecBins) addF(spec + (kNumClasses + cls) * C.nSpecBins + bb, w);
        }
    }
}

}  // namespace mc
