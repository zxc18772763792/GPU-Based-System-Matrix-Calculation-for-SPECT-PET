// CPU run of the ORIGINAL PEGen_RayTracing_CircularHole kernel (photodetectorCudaMe).
// The host part of PESysMatGen() (spatial hash of the holes, pe_buildHoleHash) is shared, then every
// CUDA thread of the launch is executed on the CPU.  Built by build.py.
//
//   pe_cpu <params_dir> [output.sysmat]
//   default output: <params_dir>/PE_SysMat_shift_<x>_<y>_<z>_v3.sysmat  (same name as PEGen)
#include "prelude.h"
#include "pe_kernels.inc"   // verbatim copy of PESysMatGen.cu up to the host function

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <params_dir> [output.sysmat]\n", argv[0]); return 1; }
    std::string d = argv[1];
    auto col = cpu::readFloats(d + "/Params_Collimator.dat", 80000);
    auto det = cpu::readFloats(d + "/Params_Detector.dat", 80000);
    auto img = cpu::readFloats(d + "/Params_Image.dat", 100);
    int nVX = (int)img[0], nVY = (int)img[1], nVZ = (int)img[2];
    long long nP = (long long)floorf(det[0] + 0.0001f), nI = (long long)nVX * nVY * nVZ;
    int nRot = (int)floor(img[6] + 0.001);

    // ---- spatial hash: the same function as PESysMatGen()
    PEHoleHash g = pe_buildHoleHash(col.data());

    std::vector<float> out((size_t)nRot * nP * nI, 0.0f);
    const long long total = nP * nI, chunk = 256;
    for (int r = 0; r < nRot; ++r) {
        img[20] = (float)r;
        float* dst = out.data() + (size_t)r * nP * nI;
        cpu::parallelFor((total + chunk - 1) / chunk, [&](long long c) {
            for (long long tid = c * chunk; tid < std::min(total, (c + 1) * chunk); ++tid) {
                blockIdx.x = (unsigned)tid; threadIdx.x = 0;
                photodetectorCudaMe(dst, col.data(), det.data(), img.data(), g.idx.data(), g.cnt.data(),
                                    g.originX, g.originZ, g.cellSize, g.nx, g.nz, g.maxPerCell, g.maxR, nP, nI);
            }
        });
    }

    std::string path;
    if (argc > 2) path = argv[2];
    else {
        char b[512];
        snprintf(b, sizeof b, "/PE_SysMat_shift_%f_%f_%f_v3.sysmat", img[8], img[9], img[10]);
        path = d + b;
    }
    cpu::writeFloats(path, out);
    double s = 0;
    for (float v : out) s += v;
    printf("analytic PE (CPU): %lld detectors x %lld voxels x %d rotations, mean sensitivity %.6e -> %s\n",
           nP, nI, nRot, s / ((double)nI * nRot), path.c_str());
    return 0;
}
