// mc_host.h
// Host-side setup for MCGen_MultiGPU: configuration, parameter files,
// material database, acceleration structures and output helpers.
#pragma once
#include <map>
#include <string>
#include <vector>
#include <cstdint>

#include "mc_core.h"

namespace mc {

// ------------------------------------------------------------------ config
class Config {
public:
    bool loadFile(const std::string& path, std::string& err);
    void set(const std::string& key, const std::string& value);
    bool has(const std::string& key) const;
    std::string str(const std::string& key, const std::string& def = "") const;
    double num(const std::string& key, double def) const;
    long long integer(const std::string& key, long long def) const;
    bool flag(const std::string& key, bool def) const;     // 1/0, on/off, true/false, yes/no
    std::vector<std::string> keysWithPrefix(const std::string& prefix) const;
    const std::map<std::string, std::string>& all() const { return kv_; }
    void markUsed(const std::string& key) const;
    std::vector<std::string> unusedKeys() const;
private:
    std::map<std::string, std::string> kv_;
    mutable std::map<std::string, bool> used_;
};

// ------------------------------------------------------- material database
struct DbMaterial {
    std::string name;
    double density = 0;                       // g/cm^3
    std::vector<float> pe, inc, coh;          // cm^2/g on the DB energy grid
    std::vector<float> fluoE;                 // line energies
    std::vector<std::string> fluoLabel;
    std::vector<int> fluoZ;
    std::vector<std::vector<float>> fluoP;    // per line, on the energy grid
    std::vector<float> ffCum;
};

struct MaterialDb {
    int nE = 0;
    double eMin = 1.0, eMax = 1100.0;
    int nFF = 0;
    double x2Min = 1e-4, x2Max = 1e4;
    std::vector<DbMaterial> mats;
    bool load(const std::string& path, std::string& err);
    const DbMaterial* find(const std::string& name) const;
};

// ------------------------------------------------- host side tables/geometry
struct HostMaterials {
    std::vector<std::string> names;           // material index -> description
    int nE = 1024;
    double eMin = 1.0, eMax = 1100.0;
    std::vector<float> muPE, muInc, muCoh, muTot, regMaj;
    std::vector<int> nFluo;
    std::vector<float> fluoE, fluoP;
    int nFF = 2;
    double x2Min = 1e-4, x2Max = 1e4;
    std::vector<float> ffX2, ffCum;
    std::vector<int> hasFF;
};

struct HostGeometry {
    Geometry g{};
    std::vector<int> hgCellStart, hgItems, holeMat, xtalMat, cgCellStart, cgItems;
    std::vector<F4> hole0, hole1, xtal0, xtal1;
    std::vector<float> xtalEres;
    std::vector<int> holeLayer;
    int numHoles = 0, numEnabledXtals = 0;
};

struct ParamFiles {
    std::vector<float> col, det, img, phy;    // zero-padded to the sizes the analytic modules read
    size_t nCol = 0, nDet = 0, nImg = 0, nPhy = 0;   // floats actually in the files
    std::string dir;
};

struct Simulation {
    Config cfg;
    ParamFiles params;
    MaterialDb db;
    bool dbLoaded = false;
    HostMaterials hm;
    HostGeometry hg;
    SimConsts C{};
    Materials M{};                            // host pointers into hm

    int numDet = 0, numVox = 0, numRot = 0;
    int rotBegin = 0, rotEnd = 0;
    long long histPerVoxel = 0;
    int nTally = 0;
    bool saveRelErr = false;
    float yield = 1.0f;
    float fov2col = 0.0f;

    int traceVoxel = -1, traceRot = 0, traceHist = 0, traceCapacity = 0;

    std::string backend;                      // "gpu" | "cpu"
    std::vector<int> gpus;
    int cpuThreads = 0;
    int threadsPerVoxel = 0;                  // 0 = auto
    long long voxelChunk = 0;                 // 0 = auto
    double launchTargetMs = 300.0;

    std::string outPrefix;
    std::vector<std::string> warnings;
    std::vector<std::string> materialReport;  // human readable assignment summary

    bool setup(std::string& err);             // everything after cfg is filled
    void bindPointers();                      // host pointers into M / hg.g
    std::string tallyName(int t) const;       // e.g. "win0_primary"
};

bool readParamFile(const std::string& path, std::vector<float>& out, size_t minCount, std::string& err, size_t* nRead = nullptr);
std::string jsonEscape(const std::string& s);

}  // namespace mc
