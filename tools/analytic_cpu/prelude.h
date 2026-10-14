// CPU stand-ins for the few CUDA names used by the analytic kernels, so that the
// ORIGINAL __global__ / __device__ code of PEGen and ScatterGen compiles with g++ and
// can be run "thread by thread" on the CPU (build.py extracts the kernels verbatim).
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __restrict__

struct CpuDim3 { unsigned x, y, z; };
static thread_local CpuDim3 blockIdx, threadIdx;      // set by the harness before each "thread"
static const CpuDim3 blockDim = {1, 1, 1};
using std::max;
using std::min;

typedef int cudaError_t;
static const int cudaSuccess = 0;
static inline const char* cudaGetErrorString(int) { return ""; }

// The harnesses give every output element to exactly one CPU thread, so a plain add is enough.
static inline float atomicAdd(float* a, float v) { float o = *a; *a = o + v; return o; }
static inline unsigned atomicOr(unsigned* a, unsigned v) { return __atomic_fetch_or(a, v, __ATOMIC_RELAXED); }

namespace cpu {
inline std::vector<float> readFloats(const std::string& path, size_t n, bool exact = false) {
    std::vector<float> v(n, 0.0f);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { perror(path.c_str()); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (exact && got != n) { fprintf(stderr, "%s: expected %zu floats, got %zu\n", path.c_str(), n, got); exit(1); }
    return v;
}
inline void writeFloats(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror(path.c_str()); exit(1); }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}
// Run body(i) for i in [0, n) on all cores, dynamic scheduling.
template <class F> void parallelFor(long long n, F body) {
    std::atomic<long long> next(0);
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&]() { for (long long i; (i = next.fetch_add(1)) < n;) body(i); });
    for (auto& th : pool) th.join();
}
}  // namespace cpu
