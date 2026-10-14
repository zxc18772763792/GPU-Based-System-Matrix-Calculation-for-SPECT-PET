// CPU run of the ORIGINAL ScatterGen_RayTracing_CircularHole kernels:
//   geometryRelationShip_Crystal2Crystal / _Collimator2Crystal (bitmaps),
//   crystalScatterSysMatCuda (image -> scatter crystal -> detector crystal),
//   collimatorScatterSysMatCuda (image -> collimator hole -> detector crystal).
// The host logic of scatter() / main() is replicated (per rotation, PE matrix of the same
// rotation).  Built by build.py.
//
//   scatter_cpu <params_dir> <PE.sysmat> [out_dir]
//   writes the same files as "ScatterGen_CircularHole -components" into out_dir (default params_dir):
//   Scatter_SysMat_shift_<x>_<y>_<z>.sysmat (sum), Scatter_Crystal_SysMat_..., Scatter_Collimator_SysMat_...
#include "prelude.h"
#include "scatter_kernels.inc"   // verbatim copy of scatter.cu up to the host function
#undef max
#undef min

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <params_dir> <PE.sysmat> [out_dir]\n", argv[0]); return 1; }
    std::string d = argv[1];
    auto col = cpu::readFloats(d + "/Params_Collimator.dat", 80000);
    auto det = cpu::readFloats(d + "/Params_Detector.dat", 80000);
    auto img = cpu::readFloats(d + "/Params_Image.dat", 100);
    auto phy = cpu::readFloats(d + "/Params_Physics.dat", 100);
    int nVX = (int)floor(img[0] + 0.001f), nVY = (int)floor(img[1] + 0.001f), nVZ = (int)floor(img[2] + 0.001f);
    int nP = (int)floor(det[0] + 0.0001f), nI = nVX * nVY * nVZ;
    int nRot = (int)floor(img[6] + 0.001f);
    int nHoles = (int)floor(col[10] + 0.01f);           // the kernels use the holes of layer 0
    size_t perRot = (size_t)nP * nI;
    auto pe = cpu::readFloats(argv[2], perRot * nRot, true);

    // ---- bitmaps (always computed here; the GPU program can also read them from file)
    const int bpw = 32;
    long long pairs = (long long)nP * (nP + 1) / 2;
    std::vector<unsigned> bitC((pairs * nP + bpw - 1) / bpw, 0u);
    std::vector<unsigned> bitH(((long long)nHoles * nP * nP + bpw - 1) / bpw, 0u);
    const long long chunk = 4096;
    long long nC = (long long)nP * nP * nP, nH = (long long)nHoles * nP * nP;
    cpu::parallelFor((nC + chunk - 1) / chunk, [&](long long c) {
        for (long long i = c * chunk; i < std::min(nC, (c + 1) * chunk); ++i) {
            blockIdx.x = (unsigned)i; threadIdx.x = 0;
            geometryRelationShip_Crystal2Crystal(bitC.data(), det.data());
        }
    });
    cpu::parallelFor((nH + chunk - 1) / chunk, [&](long long c) {
        for (long long i = c * chunk; i < std::min(nH, (c + 1) * chunk); ++i) {
            blockIdx.x = (unsigned)i; threadIdx.x = 0;
            geometryRelationShip_Collimator2Crystal(bitH.data(), det.data(), col.data());
        }
    });

    std::vector<float> xtal(perRot * nRot, 0.0f), colm(perRot * nRot, 0.0f);
    for (int r = 0; r < nRot; ++r) {
        img[20] = (float)r;
        float* dX = xtal.data() + r * perRot;
        float* dC = colm.data() + r * perRot;
        float* peR = pe.data() + r * perRot;
        // one CPU task per detector row: every dst element belongs to exactly one task
        cpu::parallelFor(nP, [&](long long row) {
            threadIdx.x = threadIdx.y = threadIdx.z = 0;
            for (int v = 0; v < nI; ++v) {
                for (int s = 0; s < nP; ++s) {
                    blockIdx.x = (unsigned)row; blockIdx.y = (unsigned)v; blockIdx.z = (unsigned)s;
                    crystalScatterSysMatCuda(dX, det.data(), img.data(), phy.data(), peR, bitC.data(), nP, nI);
                }
                for (int h = 0; h < nHoles; ++h) {
                    blockIdx.x = (unsigned)row; blockIdx.y = (unsigned)v; blockIdx.z = (unsigned)h;
                    collimatorScatterSysMatCuda(dC, col.data(), det.data(), img.data(), phy.data(), bitH.data(), nP, nI);
                }
            }
        });
    }

    std::string out = argc > 3 ? argv[3] : d;
    char shift[256];
    snprintf(shift, sizeof shift, "shift_%f_%f_%f.sysmat", img[8], img[9], img[10]);
    std::vector<float> tot(xtal.size());
    double sx = 0, sc = 0;
    for (size_t i = 0; i < tot.size(); ++i) { tot[i] = xtal[i] + colm[i]; sx += xtal[i]; sc += colm[i]; }
    cpu::writeFloats(out + "/Scatter_SysMat_" + shift, tot);
    cpu::writeFloats(out + "/Scatter_Crystal_SysMat_" + shift, xtal);
    cpu::writeFloats(out + "/Scatter_Collimator_SysMat_" + shift, colm);
    double norm = (double)nI * nRot;
    printf("analytic scatter (CPU): %d detectors x %d voxels x %d rotations, %d holes\n", nP, nI, nRot, nHoles);
    printf("  mean sensitivity: crystal scatter %.6e, collimator scatter %.6e -> %s/Scatter_[Crystal_|Collimator_]SysMat_%s\n",
           sx / norm, sc / norm, out.c_str(), shift);
    return 0;
}
