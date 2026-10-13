// mc_cpu.cpp - CPU backend: identical physics to the CUDA kernel, one voxel per task.
#include <algorithm>
#include <thread>

#include "mc_backend.h"

namespace mc {

void runRotationCPU(const Simulation& S, int rot, RotationOutput& out, Progress& prog) {
    int nThreads = S.cpuThreads > 0 ? S.cpuThreads : (int)std::max(1u, std::thread::hardware_concurrency());
    const int nb = 2 * kNumClasses * S.C.nSpecBins;
    std::atomic<int> nextVox(0);
    std::mutex mu;
    float* sum = out.sum.data();
    float* sum2 = out.sum2.empty() ? nullptr : out.sum2.data();
    const long long N = S.histPerVoxel;

    const size_t nAcc = (size_t)S.nTally * S.numDet;
    auto worker = [&]() {
        // double precision, thread private: one voxel's column of every tally
        std::vector<double> acc(nAcc, 0.0), acc2(sum2 ? nAcc : 0, 0.0);
        std::vector<double> specAcc(nb, 0.0);
        std::vector<float> traceBuf;
        int traceCount = 0;
        int v;
        while ((v = nextVox.fetch_add(1)) < S.numVox) {
            std::fill(acc.begin(), acc.end(), 0.0);
            std::fill(acc2.begin(), acc2.end(), 0.0);
            bool tracing = v == S.traceVoxel && rot == S.traceRot;
            HoleTargets targets;
            buildHoleTargets(S.hg.g, S.C, v, rot, targets);
            if (tracing) traceBuf.assign((size_t)S.traceCapacity * 6, 0.0f);
            for (long long h = 0; h < N; ++h) {
                Philox rng;
                rng.init(S.C.seed, (uint32_t)h, (uint32_t)v, (uint32_t)rot);
                Tracer tr;
                tr.active = tracing && h < S.traceHist;
                tr.track = (int)h * 8;
                tr.buf = traceBuf.data();
                tr.count = &traceCount;
                tr.capacity = S.traceCapacity;
                Deposits D;
                float w = runHistory(S.hg.g, S.M, S.C, &targets, v, rot, rng, D, tr);
                scoreHistory(S.hg.g, S.C, D, w, rng, acc.data(), sum2 ? acc2.data() : (double*)nullptr, (size_t)1, 0,
                             specAcc.data());
            }
            for (size_t i = 0; i < nAcc; ++i) {
                if (acc[i] != 0.0) sum[i * S.numVox + v] = (float)acc[i];
                if (sum2 && acc2[i] != 0.0) sum2[i * S.numVox + v] = (float)acc2[i];
            }
            if (tracing) {
                std::lock_guard<std::mutex> lk(mu);
                int n = std::min(traceCount, S.traceCapacity);
                out.trace.insert(out.trace.end(), traceBuf.begin(), traceBuf.begin() + (size_t)n * 6);
            }
            prog.add(N);
        }
        std::lock_guard<std::mutex> lk(mu);
        for (int i = 0; i < nb; ++i) out.spec[i] += specAcc[i];
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
}

}  // namespace mc
