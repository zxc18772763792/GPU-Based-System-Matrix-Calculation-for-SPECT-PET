// CPU run of the ORIGINAL PEGen_RayTracing_CircularHole kernel (photodetectorCudaMe).
// The host part of PESysMatGen() (spatial hash of the holes) is replicated 1:1, then every
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

    // ---- spatial hash, identical to PESysMatGen()
    int nLayers = (int)floorf(col[0] + 0.001f), totalHoles = 0;
    for (int l = 0; l < nLayers; l++) totalHoles += (int)col[(l + 1) * 10];
    float colW = col[11], colH = col[13], maxR = 0.0f;
    for (int h = 0; h < totalHoles; h++) maxR = std::max(maxR, col[h * 9 + 104]);
    float cellSize = fmaxf(2.0f * maxR + 0.5f, 2.0f);
    float originX = -colW / 2.0f - cellSize, originZ = -colH / 2.0f - cellSize;
    int gridNX = (int)ceilf((colW + 2 * cellSize) / cellSize) + 1;
    int gridNZ = (int)ceilf((colH + 2 * cellSize) / cellSize) + 1;
    int maxPerCell = 16;
    std::vector<int> hIdx(gridNX * gridNZ * maxPerCell, 0), hCnt(gridNX * gridNZ, 0);
    for (int h = 0; h < totalHoles; h++) {
        float hx = col[h * 9 + 100], hz = col[h * 9 + 103], hr = col[h * 9 + 104];
        int cx0 = (int)floorf((hx - hr - originX) / cellSize), cx1 = (int)floorf((hx + hr - originX) / cellSize);
        int cz0 = (int)floorf((hz - hr - originZ) / cellSize), cz1 = (int)floorf((hz + hr - originZ) / cellSize);
        for (int cx = cx0; cx <= cx1; cx++)
            for (int cz = cz0; cz <= cz1; cz++) {
                if (cx < 0 || cx >= gridNX || cz < 0 || cz >= gridNZ) continue;
                int cid = cx * gridNZ + cz, c = hCnt[cid];
                if (c < maxPerCell) { hIdx[cid * maxPerCell + c] = h; hCnt[cid] = c + 1; }
            }
    }

    std::vector<float> out((size_t)nRot * nP * nI, 0.0f);
    const long long total = nP * nI, chunk = 256;
    for (int r = 0; r < nRot; ++r) {
        img[20] = (float)r;
        float* dst = out.data() + (size_t)r * nP * nI;
        cpu::parallelFor((total + chunk - 1) / chunk, [&](long long c) {
            for (long long tid = c * chunk; tid < std::min(total, (c + 1) * chunk); ++tid) {
                blockIdx.x = (unsigned)tid; threadIdx.x = 0;
                photodetectorCudaMe(dst, col.data(), det.data(), img.data(), hIdx.data(), hCnt.data(),
                                    originX, originZ, cellSize, gridNX, gridNZ, maxPerCell, nP, nI);
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
