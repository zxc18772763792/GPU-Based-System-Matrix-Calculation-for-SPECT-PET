// CPU run of the ScatterGen_RayTracing_CircularHole kernels (built by build.py from scatter.cu):
//   model v2 (default): crystalScatterV2, collimatorScatterShellV2, collimatorScatterBulkV2
//   model legacy:       crystalScatterSysMatCuda, collimatorScatterSysMatCuda
// with the bitmaps of geometryRelationShip_Crystal2Crystal / _Collimator2Crystal.  The host logic of
// scatter() / main() is replicated (per rotation, PE matrix of the same rotation).
//
//   scatter_cpu <params_dir> <PE.sysmat> [out_dir] [-model v2|legacy] [-skip crystal|shell|bulk|collimator] [-holes a:b]
//   writes the same files as "ScatterGen_CircularHole -components" into out_dir (default params_dir):
//   Scatter_SysMat_shift_<x>_<y>_<z>.sysmat (sum), Scatter_Crystal_SysMat_..., Scatter_Collimator_SysMat_...
#include <chrono>
#include "prelude.h"
#include "scatter_kernels.inc"   // verbatim copy of scatter.cu up to the host function
#undef max
#undef min

static double seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    int model = 2;
    bool skipXtal = false, skipShell = false, skipBulk = false;     // for validation runs
    int hole0 = 0, hole1 = 1 << 30;                                 // -holes a:b (shell term of holes a..b-1 only)
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "-model" && a + 1 < argc) {
            std::string m = argv[++a];
            if (m == "legacy") model = 1;
            else if (m == "v2") model = 2;
            else { fprintf(stderr, "-model must be v2 or legacy\n"); return 1; }
        } else if (s == "-holes" && a + 1 < argc) {
            if (sscanf(argv[++a], "%d:%d", &hole0, &hole1) != 2) { fprintf(stderr, "-holes a:b\n"); return 1; }
        } else if (s == "-skip" && a + 1 < argc) {
            std::string m = argv[++a];
            if (m == "crystal") skipXtal = true;
            else if (m == "shell") skipShell = true;
            else if (m == "bulk") skipBulk = true;
            else if (m == "collimator") skipShell = skipBulk = true;
            else { fprintf(stderr, "-skip crystal|shell|bulk|collimator\n"); return 1; }
        } else pos.push_back(s);
    }
    if (pos.size() < 2) { fprintf(stderr, "usage: %s <params_dir> <PE.sysmat> [out_dir] [-model v2|legacy]\n", argv[0]); return 1; }
    std::string d = pos[0];
    auto col = cpu::readFloats(d + "/Params_Collimator.dat", 80000);
    auto det = cpu::readFloats(d + "/Params_Detector.dat", 80000);
    auto img = cpu::readFloats(d + "/Params_Image.dat", 100);
    auto phy = cpu::readFloats(d + "/Params_Physics.dat", 100);
    int nVX = (int)floor(img[0] + 0.001f), nVY = (int)floor(img[1] + 0.001f), nVZ = (int)floor(img[2] + 0.001f);
    int nP = (int)floor(det[0] + 0.0001f), nI = nVX * nVY * nVZ;
    int nRot = (int)floor(img[6] + 0.001f);
    int nHoles0 = (int)floor(col[10] + 0.01f);          // the legacy kernels use the holes of layer 0
    size_t perRot = (size_t)nP * nI;
    auto pe = cpu::readFloats(pos[1], perRot * nRot, true);
    auto t0 = std::chrono::steady_clock::now();

    // ---- bitmaps (always computed here; the GPU program can also read them from file)
    const int bpw = 32;
    const long long chunk = 4096;
    long long pairs = (long long)nP * (nP + 1) / 2;
    std::vector<unsigned> bitC((pairs * nP + bpw - 1) / bpw, 0u), bitH;
    long long nC = (long long)nP * nP * nP;
    cpu::parallelFor((nC + chunk - 1) / chunk, [&](long long c) {
        for (long long i = c * chunk; i < std::min(nC, (c + 1) * chunk); ++i) {
            blockIdx.x = (unsigned)i; threadIdx.x = 0;
            geometryRelationShip_Crystal2Crystal(bitC.data(), det.data());
        }
    });
    if (model == 1) {
        long long nH = (long long)nHoles0 * nP * nP;
        bitH.assign((nH + bpw - 1) / bpw, 0u);
        cpu::parallelFor((nH + chunk - 1) / chunk, [&](long long c) {
            for (long long i = c * chunk; i < std::min(nH, (c + 1) * chunk); ++i) {
                blockIdx.x = (unsigned)i; threadIdx.x = 0;
                geometryRelationShip_Collimator2Crystal(bitH.data(), det.data(), col.data());
            }
        });
    }

    // ---- v2 host preparation (same functions as scatter())
    std::vector<float> xi;
    std::vector<int> hashIdx, hashCnt, nb;
    S2Col layers;
    S2Hash hash;
    if (model == 2) {
        s2_crystalInfo(det.data(), img[11], xi);
        layers = s2_layers(col.data(), img[11]);
        s2_holeHash(col.data(), layers, hashIdx, hashCnt, hash);
        s2_neighbours(xi, nP, nb);
    }

    std::vector<float> xtal(perRot * nRot, 0.0f), colm(perRot * nRot, 0.0f);
    double tX = 0, tC = 0;
    for (int r = 0; r < nRot; ++r) {
        img[20] = (float)r;
        float* dX = xtal.data() + r * perRot;
        float* dC = colm.data() + r * perRot;
        float* peR = pe.data() + r * perRot;
        float peMin = model == 2 ? s2_peMin(peR, perRot) : 0.0f;
        // one CPU task per detector row: every dst element belongs to exactly one task
        auto ta = std::chrono::steady_clock::now();
        if (!skipXtal) cpu::parallelFor(nP, [&](long long row) {
            threadIdx.x = threadIdx.y = threadIdx.z = 0;
            for (int v = 0; v < nI; ++v)
                for (int s = 0; s < nP; ++s) {
                    blockIdx.x = (unsigned)row; blockIdx.y = (unsigned)v; blockIdx.z = (unsigned)s;
                    if (model == 2) crystalScatterV2(dX, xi.data(), det.data(), img.data(), phy.data(), peR, bitC.data(), nP, nI, peMin);
                    else crystalScatterSysMatCuda(dX, det.data(), img.data(), phy.data(), peR, bitC.data(), nP, nI);
                }
        });
        tX += seconds(ta);
        ta = std::chrono::steady_clock::now();
        cpu::parallelFor(nP, [&](long long row) {
            threadIdx.x = threadIdx.y = threadIdx.z = 0;
            for (int v = 0; v < nI; ++v) {
                if (model == 2) {
                    for (int h = hole0; h < (skipShell ? 0 : std::min(hole1, layers.totalHoles)); ++h) {
                        blockIdx.x = (unsigned)row; blockIdx.y = (unsigned)v; blockIdx.z = (unsigned)h;
                        collimatorScatterShellV2(dC, col.data(), xi.data(), det.data(), img.data(), phy.data(), layers, hash,
                                                 nb.data(), nP, nI, 0, layers.totalHoles);
                    }
                    for (int m = 0; m < (skipBulk ? 0 : layers.n); ++m) {
                        blockIdx.x = (unsigned)row; blockIdx.y = (unsigned)v; blockIdx.z = (unsigned)m;
                        collimatorScatterBulkV2(dC, col.data(), xi.data(), det.data(), img.data(), phy.data(), layers, hash,
                                                nb.data(), nP, nI);
                    }
                } else {
                    for (int h = 0; h < nHoles0; ++h) {
                        blockIdx.x = (unsigned)row; blockIdx.y = (unsigned)v; blockIdx.z = (unsigned)h;
                        collimatorScatterSysMatCuda(dC, col.data(), det.data(), img.data(), phy.data(), bitH.data(), nP, nI);
                    }
                }
            }
        });
        tC += seconds(ta);
    }

    std::string out = pos.size() > 2 ? pos[2] : d;
    char shift[256];
    snprintf(shift, sizeof shift, "shift_%f_%f_%f.sysmat", img[8], img[9], img[10]);
    std::vector<float> tot(xtal.size());
    double sx = 0, sc = 0;
    for (size_t i = 0; i < tot.size(); ++i) { tot[i] = xtal[i] + colm[i]; sx += xtal[i]; sc += colm[i]; }
    cpu::writeFloats(out + "/Scatter_SysMat_" + shift, tot);
    cpu::writeFloats(out + "/Scatter_Crystal_SysMat_" + shift, xtal);
    cpu::writeFloats(out + "/Scatter_Collimator_SysMat_" + shift, colm);
    double norm = (double)nI * nRot;
    printf("analytic scatter (CPU, model %s): %d detectors x %d voxels x %d rotations (%.1f s: crystal %.1f s, collimator %.1f s)\n",
           model == 2 ? "v2" : "legacy", nP, nI, nRot, seconds(t0), tX, tC);
    printf("  mean sensitivity: crystal scatter %.6e, collimator scatter %.6e -> %s/Scatter_[Crystal_|Collimator_]SysMat_%s\n",
           sx / norm, sc / norm, out.c_str(), shift);
    return 0;
}
