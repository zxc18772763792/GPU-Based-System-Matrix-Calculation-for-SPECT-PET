// mc_gpu.cu - multi-GPU backend of MCGen_MultiGPU
//
// Work decomposition: for every rotation the voxels are cut into chunks that
// GPU worker threads (one std::thread per device) pull from a shared queue,
// so faster cards simply process more chunks. Each chunk keeps its own
// [tally][detector][voxel-in-chunk] buffer on the device; finished chunks are
// gathered into the host [tally][detector][voxel] array with one strided
// cudaMemcpy2D. Kernel launches are sized adaptively (launch_target_ms) so
// display GPUs stay responsive and progress is reported regularly.
//
// Every history uses its own Philox stream keyed by (seed, history, voxel,
// rotation): results are independent of the number of GPUs, chunk sizes and
// launch splitting (up to float summation order).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <thread>

#include "mc_backend.h"

namespace mc {

#define MC_CUDA(call)                                                                       \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_) + " (" __FILE__ ":" +  \
                  std::to_string(__LINE__) + ")";                                           \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

__device__ __forceinline__ void atomicAddDouble(double* a, double v) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(a, v);
#else
    unsigned long long* p = (unsigned long long*)a;
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// tally flush: D += F, F = 0 (after every launch) / F = (float)D (before the gather)
__global__ void flushKernel(double* D, float* F, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float f = F[i];
        if (f != 0.0f) { D[i] += (double)f; F[i] = 0.0f; }
    }
}
__global__ void toFloatKernel(float* F, const double* D, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        F[i] = (float)D[i];
}

__global__ void mcKernel(Geometry G, Materials M, SimConsts C, int rot, int vStart, int nVox, int tpv,
                         long long N, long long bBase, int bCount, float* sum, float* sum2,
                         double* specOut, Tracer trBase, int traceVoxel, int traceRot, int traceHist) {
    extern __shared__ float sspec[];
    const int nb = 2 * kNumClasses * C.nSpecBins;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) sspec[i] = 0.0f;
    __syncthreads();

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int vl = (int)(tid / tpv);
    int sub = (int)(tid - (long long)vl * tpv);
    if (vl < nVox) {
        int voxel = vStart + vl;
        for (int j = 0; j < bCount; ++j) {
            long long h = (bBase + j) * tpv + sub;
            if (h >= N) break;
            Philox rng;
            rng.init(C.seed, (uint32_t)h, (uint32_t)voxel, (uint32_t)rot);
            Tracer tr = trBase;
            tr.active = trBase.active && voxel == traceVoxel && rot == traceRot && h < traceHist;
            tr.track = (int)h * 8;
            Deposits D;
            float w = runHistory(G, M, C, voxel, rot, rng, D, tr);
            scoreHistory(G, C, D, w, rng, sum, sum2, (size_t)nVox, vl, sspec);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nb; i += blockDim.x)
        if (sspec[i] != 0.0f) atomicAddDouble(&specOut[i], (double)sspec[i]);
}

struct DeviceCtx {
    int dev = 0;
    std::string name;
    int sms = 0;
    size_t memTotal = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    std::vector<void*> allocs;
    Geometry G{};
    Materials M{};
    float* tally = nullptr;            // per-launch float tallies (atomics)
    float* tally2 = nullptr;
    double* tallyD = nullptr;          // chunk accumulators
    double* tally2D = nullptr;
    double* spec = nullptr;
    float* traceBuf = nullptr;
    int* traceCount = nullptr;
    int bCount = 1;
    int tpv = 1;
    long long histories = 0;
    double kernelSeconds = 0;
};

struct GpuRunner::Impl {
    std::vector<DeviceCtx> ctx;
    long long chunk = 0;
};

GpuRunner::GpuRunner() : impl_(new Impl) {}
GpuRunner::~GpuRunner() {
    for (auto& c : impl_->ctx) {
        cudaSetDevice(c.dev);
        for (void* p : c.allocs) cudaFree(p);
        if (c.ev0) cudaEventDestroy(c.ev0);
        if (c.ev1) cudaEventDestroy(c.ev1);
        if (c.stream) cudaStreamDestroy(c.stream);
    }
}

template <class T>
static bool upload(DeviceCtx& c, const std::vector<T>& v, const T*& dst, std::string& err) {
    void* p = nullptr;
    size_t bytes = std::max<size_t>(v.size(), 1) * sizeof(T);
    MC_CUDA(cudaMalloc(&p, bytes));
    c.allocs.push_back(p);
    if (!v.empty()) MC_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    dst = (const T*)p;
    return true;
}

template <class T>
static bool alloc(DeviceCtx& c, size_t n, T*& dst, std::string& err) {
    void* p = nullptr;
    MC_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    c.allocs.push_back(p);
    dst = (T*)p;
    return true;
}

bool GpuRunner::init(const Simulation& S, std::string& err) {
    int count = 0;
    MC_CUDA(cudaGetDeviceCount(&count));
    if (count == 0) { err = "no CUDA device found"; return false; }
    std::vector<int> ids = S.gpus;
    if (ids.empty())
        for (int i = 0; i < count; ++i) ids.push_back(i);
    for (int id : ids)
        if (id < 0 || id >= count) { err = "GPU id " + std::to_string(id) + " out of range"; return false; }

    const HostGeometry& hg = S.hg;
    const HostMaterials& hm = S.hm;
    size_t specBytes = (size_t)2 * kNumClasses * S.C.nSpecBins;
    size_t minFree = (size_t)-1;
    for (int id : ids) {
        DeviceCtx c;
        c.dev = id;
        MC_CUDA(cudaSetDevice(id));
        cudaDeviceProp prop;
        MC_CUDA(cudaGetDeviceProperties(&prop, id));
        c.name = prop.name;
        c.sms = prop.multiProcessorCount;
        c.memTotal = prop.totalGlobalMem;
        MC_CUDA(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        MC_CUDA(cudaEventCreate(&c.ev0));
        MC_CUDA(cudaEventCreate(&c.ev1));

        c.G = hg.g;
        if (!upload(c, hg.hgCellStart, c.G.hgCellStart, err) || !upload(c, hg.hgItems, c.G.hgItems, err) ||
            !upload(c, hg.hole0, c.G.hole0, err) || !upload(c, hg.hole1, c.G.hole1, err) ||
            !upload(c, hg.holeMat, c.G.holeMat, err) || !upload(c, hg.xtal0, c.G.xtal0, err) ||
            !upload(c, hg.xtal1, c.G.xtal1, err) || !upload(c, hg.xtalMat, c.G.xtalMat, err) ||
            !upload(c, hg.xtalEres, c.G.xtalEres, err) || !upload(c, hg.cgCellStart, c.G.cgCellStart, err) ||
            !upload(c, hg.cgItems, c.G.cgItems, err))
            return false;
        c.M = S.M;
        if (!upload(c, hm.muPE, c.M.muPE, err) || !upload(c, hm.muInc, c.M.muInc, err) ||
            !upload(c, hm.muCoh, c.M.muCoh, err) || !upload(c, hm.muTot, c.M.muTot, err) ||
            !upload(c, hm.regMaj, c.M.regMaj, err) || !upload(c, hm.nFluo, c.M.nFluo, err) ||
            !upload(c, hm.fluoE, c.M.fluoE, err) || !upload(c, hm.fluoP, c.M.fluoP, err) ||
            !upload(c, hm.ffX2, c.M.ffX2, err) || !upload(c, hm.ffCum, c.M.ffCum, err) ||
            !upload(c, hm.hasFF, c.M.hasFF, err))
            return false;
        if (!alloc(c, specBytes, c.spec, err)) return false;
        MC_CUDA(cudaMemset(c.spec, 0, specBytes * sizeof(double)));
        if (S.traceCapacity > 0) {
            if (!alloc(c, (size_t)S.traceCapacity * 6, c.traceBuf, err) || !alloc(c, 1, c.traceCount, err)) return false;
            MC_CUDA(cudaMemset(c.traceCount, 0, sizeof(int)));
        }
        size_t fr = 0, tot = 0;
        MC_CUDA(cudaMemGetInfo(&fr, &tot));
        minFree = std::min(minFree, fr);
        impl_->ctx.push_back(c);
    }

    // chunk size: memory budget and load balance
    size_t bytesPerVox = (size_t)S.nTally * S.numDet * (sizeof(float) + sizeof(double)) * (S.saveRelErr ? 2 : 1);
    long long memChunk = (long long)std::max<size_t>(1, (size_t)(0.6 * minFree) / bytesPerVox);
    long long balance = (S.numVox + 8LL * ids.size() - 1) / (8LL * ids.size());
    long long chunk = S.voxelChunk > 0 ? S.voxelChunk : std::max(1LL, balance);
    chunk = std::min(std::min(chunk, memChunk), (long long)S.numVox);
    if (chunk < 1) { err = "not enough GPU memory for even one voxel of tallies"; return false; }
    impl_->chunk = chunk;
    for (auto& c : impl_->ctx) {
        MC_CUDA(cudaSetDevice(c.dev));
        size_t n = (size_t)chunk * S.nTally * S.numDet;
        if (!alloc(c, n, c.tally, err) || !alloc(c, n, c.tallyD, err)) return false;
        if (S.saveRelErr && (!alloc(c, n, c.tally2, err) || !alloc(c, n, c.tally2D, err))) return false;
        MC_CUDA(cudaMemset(c.tally, 0, n * sizeof(float)));
        if (c.tally2) MC_CUDA(cudaMemset(c.tally2, 0, n * sizeof(float)));
        long long target = (long long)c.sms * 2048 * 2;
        long long tpv = S.threadsPerVoxel > 0 ? S.threadsPerVoxel : (target + chunk - 1) / chunk;
        c.tpv = (int)std::max(1LL, std::min(std::min(tpv, 1024LL), S.histPerVoxel));
    }
    return true;
}

bool GpuRunner::runRotation(const Simulation& S, int rot, RotationOutput& out, Progress& prog, std::string& errOut) {
    const long long chunk = impl_->chunk;
    const long long nChunks = (S.numVox + chunk - 1) / chunk;
    std::atomic<long long> next(0);
    std::atomic<bool> failed(false);
    std::mutex mu;
    const size_t shmem = sizeof(float) * 2 * kNumClasses * S.C.nSpecBins;

    auto worker = [&](DeviceCtx& c, std::string& err) -> bool {
        MC_CUDA(cudaSetDevice(c.dev));
        long long ci;
        while (!failed && (ci = next.fetch_add(1)) < nChunks) {
            int vStart = (int)(ci * chunk);
            int nv = (int)std::min(chunk, (long long)S.numVox - vStart);
            size_t n = (size_t)nv * S.nTally * S.numDet;
            MC_CUDA(cudaMemsetAsync(c.tallyD, 0, n * sizeof(double), c.stream));
            if (c.tally2D) MC_CUDA(cudaMemsetAsync(c.tally2D, 0, n * sizeof(double), c.stream));
            const unsigned flushGrid = (unsigned)std::min<size_t>((n + 255) / 256, (size_t)c.sms * 32);
            Tracer tr;
            tr.active = c.traceBuf != nullptr;
            tr.track = 0;
            tr.buf = c.traceBuf;
            tr.count = c.traceCount;
            tr.capacity = S.traceCapacity;
            const long long N = S.histPerVoxel;
            const long long bTotal = (N + c.tpv - 1) / c.tpv;
            const long long threads = (long long)nv * c.tpv;
            const int block = 128;
            const long long grid = (threads + block - 1) / block;
            // float tallies only ever hold <= 16384 histories per voxel (and each block's spectrum
            // <= 128 x 128) before they are flushed to double, so no increment is lost to rounding
            const int bcMax = std::max(1, std::min(128, 16384 / c.tpv));
            for (long long b = 0; b < bTotal;) {
                int bc = (int)std::min<long long>(std::min(c.bCount, bcMax), bTotal - b);
                MC_CUDA(cudaEventRecord(c.ev0, c.stream));
                mcKernel<<<(unsigned)grid, block, shmem, c.stream>>>(
                    c.G, c.M, S.C, rot, vStart, nv, c.tpv, N, b, bc, c.tally, c.tally2, c.spec, tr,
                    S.traceVoxel, S.traceRot, S.traceHist);
                MC_CUDA(cudaGetLastError());
                flushKernel<<<flushGrid, 256, 0, c.stream>>>(c.tallyD, c.tally, n);
                if (c.tally2) flushKernel<<<flushGrid, 256, 0, c.stream>>>(c.tally2D, c.tally2, n);
                MC_CUDA(cudaGetLastError());
                MC_CUDA(cudaEventRecord(c.ev1, c.stream));
                MC_CUDA(cudaEventSynchronize(c.ev1));
                float ms = 0;
                MC_CUDA(cudaEventElapsedTime(&ms, c.ev0, c.ev1));
                long long h = (long long)nv * (std::min(N, (b + bc) * c.tpv) - std::min(N, b * c.tpv));
                c.histories += h;
                c.kernelSeconds += ms * 1e-3;
                prog.add(h);
                b += bc;
                double scale = ms > 0 ? S.launchTargetMs / ms : 2.0;
                scale = std::min(std::max(scale, 0.25), 4.0);
                c.bCount = (int)std::max(1.0, std::min((double)bcMax, bc * scale));
            }
            // double accumulators -> float (reuse the per-launch buffers) for the gather
            toFloatKernel<<<flushGrid, 256, 0, c.stream>>>(c.tally, c.tallyD, n);
            if (c.tally2) toFloatKernel<<<flushGrid, 256, 0, c.stream>>>(c.tally2, c.tally2D, n);
            MC_CUDA(cudaGetLastError());
            // gather [tally*det][voxel-in-chunk] -> host [tally*det][voxel]
            MC_CUDA(cudaMemcpy2DAsync(out.sum.data() + vStart, (size_t)S.numVox * sizeof(float), c.tally,
                                      (size_t)nv * sizeof(float), (size_t)nv * sizeof(float),
                                      (size_t)S.nTally * S.numDet, cudaMemcpyDeviceToHost, c.stream));
            if (c.tally2)
                MC_CUDA(cudaMemcpy2DAsync(out.sum2.data() + vStart, (size_t)S.numVox * sizeof(float), c.tally2,
                                          (size_t)nv * sizeof(float), (size_t)nv * sizeof(float),
                                          (size_t)S.nTally * S.numDet, cudaMemcpyDeviceToHost, c.stream));
            MC_CUDA(cudaMemsetAsync(c.tally, 0, n * sizeof(float), c.stream));
            if (c.tally2) MC_CUDA(cudaMemsetAsync(c.tally2, 0, n * sizeof(float), c.stream));
            MC_CUDA(cudaStreamSynchronize(c.stream));
        }
        if (c.traceBuf) {
            int cnt = 0;
            MC_CUDA(cudaMemcpy(&cnt, c.traceCount, sizeof(int), cudaMemcpyDeviceToHost));
            cnt = std::min(cnt, S.traceCapacity);
            if (cnt > 0) {
                std::vector<float> tmp((size_t)cnt * 6);
                MC_CUDA(cudaMemcpy(tmp.data(), c.traceBuf, tmp.size() * sizeof(float), cudaMemcpyDeviceToHost));
                std::lock_guard<std::mutex> lk(mu);
                out.trace.insert(out.trace.end(), tmp.begin(), tmp.end());
            }
            MC_CUDA(cudaMemset(c.traceCount, 0, sizeof(int)));
        }
        return true;
    };

    std::vector<std::thread> pool;
    std::vector<std::string> errs(impl_->ctx.size());
    for (size_t i = 0; i < impl_->ctx.size(); ++i)
        pool.emplace_back([&, i]() {
            if (!worker(impl_->ctx[i], errs[i])) {
                failed = true;
                errs[i] = "GPU " + std::to_string(impl_->ctx[i].dev) + ": " + errs[i];
            }
        });
    for (auto& t : pool) t.join();
    if (failed) {
        errOut.clear();
        for (auto& e : errs)
            if (!e.empty()) errOut += e + "\n";
        return false;
    }
    return true;
}

void GpuRunner::collectSpectrum(RotationOutput& out) {
    size_t nb = out.spec.size();
    std::vector<double> tmp(nb);
    for (auto& c : impl_->ctx) {
        cudaSetDevice(c.dev);
        if (cudaMemcpy(tmp.data(), c.spec, nb * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess)
            for (size_t i = 0; i < nb; ++i) out.spec[i] += tmp[i];
    }
}

std::vector<std::string> GpuRunner::report() const {
    std::vector<std::string> r;
    for (auto& c : impl_->ctx) {
        char b[256];
        snprintf(b, sizeof(b), "GPU %d %s: %d SMs, %.1f GB, %lld histories, %.3e hist/s (kernel time)", c.dev,
                 c.name.c_str(), c.sms, c.memTotal / 1e9, c.histories,
                 c.kernelSeconds > 0 ? c.histories / c.kernelSeconds : 0.0);
        r.push_back(b);
    }
    char b[128];
    snprintf(b, sizeof(b), "voxel chunk = %lld", impl_->chunk);
    r.push_back(b);
    return r;
}

std::string GpuRunner::devicesJson() const {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < impl_->ctx.size(); ++i) {
        auto& c = impl_->ctx[i];
        o << (i ? "," : "") << "{\"id\":" << c.dev << ",\"name\":\"" << jsonEscape(c.name) << "\",\"sms\":" << c.sms
          << ",\"histories\":" << c.histories << ",\"kernel_seconds\":" << c.kernelSeconds
          << ",\"threads_per_voxel\":" << c.tpv << "}";
    }
    o << "]";
    return o.str();
}

}  // namespace mc
