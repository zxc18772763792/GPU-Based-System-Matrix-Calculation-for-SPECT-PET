#define _CRT_SECURE_NO_WARNINGS

// MCGen_MultiGPU.cpp
// Description:
//   Multi-GPU Monte Carlo photon transport for SPECT/PET system matrices,
//   tuned for 140.5 keV (Tc-99m) and 511 keV photons. Reads the same four
//   parameter files as the analytic modules and writes system matrices in the
//   same float32 [rotation][detector][voxel] layout.
//
// Usage:
//   ./MCGen_MultiGPU [config.txt] [-key value ...]
//   ./MCGen_MultiGPU MC_Config_Tc99m.txt -gpus 0,1,2,3 -histories_per_voxel 1e6
//   ./MCGen_MultiGPU MC_Config_511keV.txt -cuda 0          (single GPU, -cuda = -gpus)
//   ./MCGen_MultiGPU MC_Config_Tc99m.txt -backend cpu      (no GPU required)
//
// See ReadMe.md for all configuration keys.

#if defined(__linux__)
#include <unistd.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mc_backend.h"

using namespace mc;

static void usage(const char* exe) {
    printf("Usage: %s [config.txt] [-key value ...]\n", exe);
    printf("  common keys: preset (tc99m|511), energy_keV, histories_per_voxel, gpus (all|0,1,..),\n");
    printf("               backend (gpu|cpu), collimator_material, hole_material, crystal_material,\n");
    printf("               windows, save_components, save_relerr, trace_voxel, params_dir, output_dir\n");
    printf("  -cuda N is accepted as an alias of -gpus N; -cpu selects the CPU backend.\n");
}

static std::string dirOf(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? "." : p.substr(0, s);
}

// directory of the running executable (argv[0] has no directory when it was found through PATH)
static std::string exeDir(const char* argv0) {
#if defined(__linux__)
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) { buf[n] = 0; return dirOf(buf); }
#endif
    return dirOf(argv0);
}

static bool writeAll(FILE* f, const float* p, size_t n) { return fwrite(p, sizeof(float), n, f) == n; }

int main(int argc, char* argv[]) {
    Simulation S;
    S.cfg.set("_exe_dir", exeDir(argv[0]));
    std::string err;
    // pass 1: configuration files (positional arguments)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        if (a == "-cpu" || a == "--cpu") continue;
        if (a[0] == '-') {
            if (i + 1 >= argc) { fprintf(stderr, "error: missing value for %s\n", a.c_str()); usage(argv[0]); return 1; }
            ++i;
            continue;
        }
        if (a.find('=') != std::string::npos) continue;
        if (!S.cfg.loadFile(a, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        S.cfg.set("_config_file", a);
    }
    // pass 2: command line keys override the files
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-cpu" || a == "--cpu") { S.cfg.set("backend", "cpu"); continue; }
        if (a[0] == '-') {
            std::string k = a.substr(a[1] == '-' ? 2 : 1);
            if (k == "cuda") k = "gpus";
            S.cfg.set(k, argv[++i]);
        } else if (a.find('=') != std::string::npos) {
            S.cfg.set(a.substr(0, a.find('=')), a.substr(a.find('=') + 1));
        }
    }

    printf("########################################################\n");
    printf(" MCGen_MultiGPU - Monte Carlo system matrix generator\n");
    printf("########################################################\n");
    if (!S.setup(err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const SimConsts& C = S.C;
    printf("Photon energy        : %.3f keV\n", C.E0);
    for (int k = 0; k < C.nWin; ++k) {
        if (C.winAuto[k]) printf("Energy window %d      : per crystal (1 -/+ R/2) x E0\n", k);
        else printf("Energy window %d      : %.2f - %.2f keV\n", k, C.winLo[k], C.winHi[k]);
    }
    printf("Window model         : %s (resolution ~ E^-%.2f)\n",
           C.winMode == kWinGauss ? "gaussian" : C.winMode == kWinIdeal ? "ideal" : "none", C.eresExp);
    printf("Physics              : %s, Rayleigh %s, fluorescence %s, cut %.1f keV\n",
           C.primaryOnly ? "primary_pe (analytic-equivalent)" : C.singleScatter ? "single_scatter (analytic first-order scatter)" : "full",
           S.cfg.flag("rayleigh", true) ? "on" : "off", C.fluo ? "on" : "off", C.eCut);
    if (C.dirRect) {
        printf("Source               : %s in voxel, importance-sampled directions (detector_bias %.2f",
               C.srcUniform ? "uniform" : "centre", C.detBias);
        if (S.hg.g.tgtLayer >= 0 && C.holeBias > 0)
            printf(", hole_bias %.2f on layer %d holes", C.holeBias, S.hg.g.tgtLayer);
        printf(")\n");
    }
    else
        printf("Source               : %s in voxel, isotropic directions\n", C.srcUniform ? "uniform" : "centre");
    printf("Image                : %d x %d x %d voxels (%.3g x %.3g x %.3g mm), rotations %d..%d of %d\n", C.nX, C.nY,
           C.nZ, C.wX, C.wY, C.wZ, S.rotBegin, S.rotEnd - 1, S.numRot);
    printf("Detector bins        : %d (%d enabled crystals), holes %d, layers %d\n", S.numDet,
           S.hg.numEnabledXtals, S.hg.numHoles, S.hg.g.nLayers);
    for (auto& m : S.materialReport) printf("  %s\n", m.c_str());
    printf("Histories per voxel  : %lld  (total %.3e)\n", S.histPerVoxel,
           (double)S.histPerVoxel * S.numVox * (S.rotEnd - S.rotBegin));
    double gb = (double)S.nTally * S.numDet * S.numVox * 4.0 * (S.saveRelErr ? 2 : 1) / 1e9;
    printf("Host tally memory    : %.2f GB per rotation\n", gb);
    for (auto& w : S.warnings) printf("WARNING: %s\n", w.c_str());
    for (auto& k : S.cfg.unusedKeys()) printf("WARNING: unknown configuration key '%s'\n", k.c_str());

    // ---------------------------------------------------------------- backend
    RotationOutput out;
    out.sum.assign((size_t)S.nTally * S.numDet * S.numVox, 0.0f);
    if (S.saveRelErr) out.sum2.assign(out.sum.size(), 0.0f);
    out.spec.assign((size_t)2 * kNumClasses * C.nSpecBins, 0.0);

#ifdef MC_WITH_CUDA
    GpuRunner gpu;
    if (S.backend == "gpu") {
        if (!gpu.init(S, err)) {
            fprintf(stderr, "error: GPU initialisation failed: %s\n", err.c_str());
            return 1;
        }
        for (auto& l : gpu.report()) printf("  %s\n", l.c_str());
    }
#else
    if (S.backend == "gpu") {
        fprintf(stderr, "error: this binary was built without CUDA; rebuild with ./bd or run with -backend cpu\n");
        return 1;
    }
#endif
    if (S.backend == "cpu") printf("Backend              : CPU, %d threads\n", S.cpuThreads > 0 ? S.cpuThreads : (int)std::thread::hardware_concurrency());

    // ---------------------------------------------------------------- outputs
    const int nCls = C.components ? kNumClasses : 1;
    struct OutFile { std::string path; FILE* f; };
    std::vector<OutFile> files;
    auto open = [&](const std::string& suffix) -> int {
        std::string p = S.outPrefix + "_" + suffix + ".sysmat";
        FILE* f = fopen(p.c_str(), "wb");
        if (!f) { fprintf(stderr, "error: cannot create %s\n", p.c_str()); exit(1); }
        files.push_back({p, f});
        return (int)files.size() - 1;
    };
    std::vector<int> fTotal(C.nWin), fRel(C.nWin, -1);
    std::vector<std::vector<int>> fCls(C.nWin), fClsRel(C.nWin);
    for (int k = 0; k < C.nWin; ++k) {
        std::string w = "win" + std::to_string(k);
        fTotal[k] = open(C.components ? w + "_total" : w);
        if (C.components) {
            static const char* names[] = {"primary", "colscatter", "xtalscatter"};
            for (int c = 0; c < kNumClasses; ++c) {
                fCls[k].push_back(open(w + "_" + names[c]));
                // each component's own statistical error (the total's error mostly measures the primary)
                if (S.saveRelErr) fClsRel[k].push_back(open(w + "_" + names[c] + "_relerr"));
            }
        }
        if (S.saveRelErr) fRel[k] = open(w + "_relerr");
    }

    const size_t slab = (size_t)S.numDet * S.numVox;
    const double N = (double)S.histPerVoxel;
    const float scale = (float)(S.yield / N);
    std::vector<float> buf(slab);
    std::vector<double> sens((size_t)C.nWin * (nCls + 1), 0.0);  // per window: total, classes
    long long totalHist = (long long)S.histPerVoxel * S.numVox * (S.rotEnd - S.rotBegin);

    Progress prog;
    prog.start(totalHist);
    auto tStart = std::chrono::steady_clock::now();
    for (int rot = S.rotBegin; rot < S.rotEnd; ++rot) {
        printf("Rotation %d ...\n", rot);
        fflush(stdout);
        std::fill(out.sum.begin(), out.sum.end(), 0.0f);
        if (S.saveRelErr) std::fill(out.sum2.begin(), out.sum2.end(), 0.0f);
#ifdef MC_WITH_CUDA
        if (S.backend == "gpu") {
            if (!gpu.runRotation(S, rot, out, prog, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        } else
#endif
            runRotationCPU(S, rot, out, prog);

        for (int k = 0; k < C.nWin; ++k) {
            // total = sum over classes
            const float* base = out.sum.data() + (size_t)k * nCls * slab;
            for (size_t i = 0; i < slab; ++i) {
                float t = 0;
                for (int c = 0; c < nCls; ++c) t += base[c * slab + i];
                buf[i] = t;
            }
            std::vector<float> rel;
            if (S.saveRelErr) {
                rel.assign(slab, 0.0f);
                const float* b2 = out.sum2.data() + (size_t)k * nCls * slab;
                for (size_t i = 0; i < slab; ++i) {
                    double s1 = buf[i], s2 = 0;
                    for (int c = 0; c < nCls; ++c) s2 += b2[c * slab + i];
                    rel[i] = s1 > 0 ? (float)std::sqrt(std::max(0.0, s2 / (s1 * s1) - 1.0 / N)) : 0.0f;
                }
            }
            double st = 0;
            for (size_t i = 0; i < slab; ++i) { buf[i] *= scale; st += buf[i]; }
            sens[k * (nCls + 1)] += st;
            if (!writeAll(files[fTotal[k]].f, buf.data(), slab)) { fprintf(stderr, "error: write failed\n"); return 1; }
            if (C.components)
                for (int c = 0; c < kNumClasses; ++c) {
                    double sc = 0;
                    for (size_t i = 0; i < slab; ++i) { buf[i] = base[c * slab + i] * scale; sc += buf[i]; }
                    sens[k * (nCls + 1) + 1 + c] += sc;
                    if (!writeAll(files[fCls[k][c]].f, buf.data(), slab)) { fprintf(stderr, "error: write failed\n"); return 1; }
                    if (S.saveRelErr) {
                        const float* b1 = base + c * slab;
                        const float* b2 = out.sum2.data() + ((size_t)k * nCls + c) * slab;
                        for (size_t i = 0; i < slab; ++i) {
                            double s1 = b1[i];
                            buf[i] = s1 > 0 ? (float)std::sqrt(std::max(0.0, b2[i] / (s1 * s1) - 1.0 / N)) : 0.0f;
                        }
                        if (!writeAll(files[fClsRel[k][c]].f, buf.data(), slab)) { fprintf(stderr, "error: write failed\n"); return 1; }
                    }
                }
            if (S.saveRelErr && !writeAll(files[fRel[k]].f, rel.data(), slab)) { fprintf(stderr, "error: write failed\n"); return 1; }
        }
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
    for (auto& f : files) fclose(f.f);
#ifdef MC_WITH_CUDA
    if (S.backend == "gpu") {
        gpu.collectSpectrum(out);
        for (auto& l : gpu.report()) printf("  %s\n", l.c_str());
    }
#endif
    printf("Done: %.3e histories in %.1f s (%.3e hist/s)\n", (double)totalHist, wall, totalHist / std::max(wall, 1e-9));

    // ---------------------------------------------------------------- spectrum
    std::string specPath = S.outPrefix + "_spectrum.csv";
    {
        FILE* f = fopen(specPath.c_str(), "w");
        if (f) {
            fprintf(f, "E_low_keV,E_high_keV,raw_primary,raw_colscatter,raw_xtalscatter,blur_primary,blur_colscatter,blur_xtalscatter\n");
            double norm = S.yield / (double)totalHist;
            for (int b = 0; b < C.nSpecBins; ++b) {
                fprintf(f, "%.4f,%.4f", b * C.specBinW, (b + 1) * C.specBinW);
                for (int s = 0; s < 2 * kNumClasses; ++s) fprintf(f, ",%.6e", out.spec[(size_t)s * C.nSpecBins + b] * norm);
                fprintf(f, "\n");
            }
            fclose(f);
        }
    }

    // ---------------------------------------------------------------- tracks
    std::string trackPath;
    if (!out.trace.empty()) {
        trackPath = S.outPrefix + "_tracks.json";
        std::map<int, std::vector<int>> tracks;   // track id -> vertex indices (in recorded order)
        size_t nv = out.trace.size() / 6;
        for (size_t i = 0; i < nv; ++i) tracks[(int)out.trace[i * 6 + 5]].push_back((int)i);
        FILE* f = fopen(trackPath.c_str(), "w");
        if (f) {
            fprintf(f, "{\"voxel\":%d,\"rotation\":%d,\"E0\":%.4f,\"vertex_types\":[\"emit\",\"photo\",\"compton\",\"rayleigh\",\"escape\",\"fluo_emit\",\"cutoff\",\"killed\"],\"tracks\":[\n",
                    S.traceVoxel, S.traceRot, C.E0);
            bool first = true;
            for (auto& t : tracks) {
                fprintf(f, "%s{\"id\":%d,\"v\":[", first ? "" : ",\n", t.first);
                for (size_t j = 0; j < t.second.size(); ++j) {
                    const float* v = &out.trace[(size_t)t.second[j] * 6];
                    fprintf(f, "%s[%.3f,%.3f,%.3f,%.2f,%d]", j ? "," : "", v[0], v[1], v[2], v[3], (int)v[4]);
                }
                fprintf(f, "]}");
                first = false;
            }
            fprintf(f, "\n]}\n");
            fclose(f);
        }
    }

    // ---------------------------------------------------------------- summary
    std::string sumPath = S.outPrefix + "_summary.json";
    FILE* f = fopen(sumPath.c_str(), "w");
    if (f) {
        int nr = S.rotEnd - S.rotBegin;
        fprintf(f, "{\n  \"program\": \"MCGen_MultiGPU\",\n");
        fprintf(f, "  \"energy_keV\": %.4f,\n", C.E0);
        fprintf(f, "  \"windows\": [");
        for (int k = 0; k < C.nWin; ++k)
            fprintf(f, "%s{\"lo\":%.4f,\"hi\":%.4f,\"per_crystal\":%s}", k ? "," : "", C.winLo[k], C.winHi[k], C.winAuto[k] ? "true" : "false");
        fprintf(f, "],\n  \"window_mode\": \"%s\",\n", C.winMode == kWinGauss ? "gauss" : C.winMode == kWinIdeal ? "ideal" : "none");
        fprintf(f, "  \"physics\": \"%s\",\n  \"rayleigh\": %s,\n  \"fluorescence\": %s,\n",
                C.primaryOnly ? "primary_pe" : C.singleScatter ? "single_scatter" : "full", S.cfg.flag("rayleigh", true) ? "true" : "false", C.fluo ? "true" : "false");
        fprintf(f, "  \"direction_sampling\": \"%s\",\n  \"detector_bias\": %.3f,\n  \"source_sampling\": \"%s\",\n",
                C.dirRect ? "rect" : "iso", C.detBias, C.srcUniform ? "uniform" : "center");
        fprintf(f, "  \"hole_bias\": %.3f,\n  \"hole_target_layer\": %d,\n", C.holeBias, S.hg.g.tgtLayer);
        fprintf(f, "  \"histories_per_voxel\": %lld,\n  \"total_histories\": %lld,\n", S.histPerVoxel, totalHist);
        fprintf(f, "  \"yield\": %.6g,\n  \"seed\": %llu,\n", S.yield, (unsigned long long)C.seed);
        fprintf(f, "  \"image\": [%d,%d,%d],\n  \"num_detectors\": %d,\n  \"num_voxels\": %d,\n", C.nX, C.nY, C.nZ, S.numDet, S.numVox);
        fprintf(f, "  \"rotations\": [%d,%d],\n  \"num_rotations_total\": %d,\n", S.rotBegin, S.rotEnd, S.numRot);
        fprintf(f, "  \"backend\": \"%s\",\n", S.backend.c_str());
#ifdef MC_WITH_CUDA
        if (S.backend == "gpu") fprintf(f, "  \"devices\": %s,\n", gpu.devicesJson().c_str());
#endif
        fprintf(f, "  \"wall_seconds\": %.3f,\n", wall);
        fprintf(f, "  \"materials\": [");
        for (size_t i = 0; i < S.materialReport.size(); ++i) fprintf(f, "%s\"%s\"", i ? "," : "", jsonEscape(S.materialReport[i]).c_str());
        fprintf(f, "],\n  \"warnings\": [");
        for (size_t i = 0; i < S.warnings.size(); ++i) fprintf(f, "%s\"%s\"", i ? "," : "", jsonEscape(S.warnings[i]).c_str());
        fprintf(f, "],\n  \"mean_sensitivity\": [");
        // mean over voxels and rotations of sum over detectors (detection probability per emitted photon)
        for (int k = 0; k < C.nWin; ++k) {
            double d = (double)S.numVox * nr;
            fprintf(f, "%s{\"window\":%d,\"total\":%.6e", k ? "," : "", k, sens[k * (nCls + 1)] / d);
            if (C.components)
                fprintf(f, ",\"primary\":%.6e,\"colscatter\":%.6e,\"xtalscatter\":%.6e", sens[k * (nCls + 1) + 1] / d,
                        sens[k * (nCls + 1) + 2] / d, sens[k * (nCls + 1) + 3] / d);
            fprintf(f, "}");
        }
        fprintf(f, "],\n  \"files\": {\"spectrum\": \"%s\"", jsonEscape(specPath).c_str());
        if (!trackPath.empty()) fprintf(f, ", \"tracks\": \"%s\"", jsonEscape(trackPath).c_str());
        fprintf(f, ", \"sysmat\": [");
        for (size_t i = 0; i < files.size(); ++i) fprintf(f, "%s\"%s\"", i ? "," : "", jsonEscape(files[i].path).c_str());
        fprintf(f, "]}\n}\n");
        fclose(f);
    }
    printf("Outputs:\n");
    for (auto& o : files) printf("  %s\n", o.path.c_str());
    printf("  %s\n", specPath.c_str());
    if (!trackPath.empty()) printf("  %s\n", trackPath.c_str());
    printf("  %s\n", sumPath.c_str());
    for (int k = 0; k < C.nWin; ++k)
        printf("Mean sensitivity (window %d): %.6e counts per emitted photon\n", k,
               sens[k * (nCls + 1)] / ((double)S.numVox * (S.rotEnd - S.rotBegin)));
    return 0;
}
