// mc_backend.h - interface between the driver and the CPU / multi-GPU backends
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mc_host.h"

namespace mc {

// Tallies of one rotation (raw sums, normalised by the driver)
struct RotationOutput {
    std::vector<float> sum;    // [nTally][nDet][nVox]
    std::vector<float> sum2;   // same layout, sum of squared history contributions (optional)
    std::vector<double> spec;  // [2*3][nSpecBins] accumulated over all rotations
    std::vector<float> trace;  // 6 floats per vertex (x y z E type track)
};

class Progress {
public:
    void start(long long total) {
        total_ = total;
        done_ = 0;
        t0_ = last_ = std::chrono::steady_clock::now();
    }
    void add(long long n) {
        long long d = (done_ += n);
        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mu_);
        if (std::chrono::duration<double>(now - last_).count() < 5.0 && d < total_) return;
        last_ = now;
        double el = std::chrono::duration<double>(now - t0_).count();
        double rate = el > 0 ? d / el : 0;
        double eta = rate > 0 ? (total_ - d) / rate : 0;
        printf("  progress %6.2f%%  %.3e histories  %.3e hist/s  elapsed %.0f s  ETA %.0f s\n",
               100.0 * d / (double)total_, (double)d, rate, el, eta);
        fflush(stdout);
    }
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }
    long long done() const { return done_; }
private:
    std::atomic<long long> done_{0};
    long long total_ = 1;
    std::chrono::steady_clock::time_point t0_, last_;
    std::mutex mu_;
};

// CPU backend (std::thread). Same physics code as the GPU kernel.
void runRotationCPU(const Simulation& S, int rot, RotationOutput& out, Progress& prog);

// Multi-GPU backend. Implemented in mc_gpu.cu; only available when the
// program is built with MC_WITH_CUDA.
class GpuRunner {
public:
    GpuRunner();
    ~GpuRunner();
    bool init(const Simulation& S, std::string& err);
    bool runRotation(const Simulation& S, int rot, RotationOutput& out, Progress& prog, std::string& err);
    void collectSpectrum(RotationOutput& out);
    std::vector<std::string> report() const;       // one line per device
    std::string devicesJson() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mc
