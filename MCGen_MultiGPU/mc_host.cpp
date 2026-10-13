// mc_host.cpp - host-side setup for MCGen_MultiGPU (see mc_host.h)
#define _CRT_SECURE_NO_WARNINGS
#include "mc_host.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace mc {

// =================================================================== utils
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
static std::vector<std::string> split(const std::string& s, const std::string& seps) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (seps.find(c) != std::string::npos) {
            if (!trim(cur).empty()) out.push_back(trim(cur));
            cur.clear();
        } else cur += c;
    }
    if (!trim(cur).empty()) out.push_back(trim(cur));
    return out;
}
static std::string fmt(const char* f, double v) {
    char b[64];
    snprintf(b, sizeof(b), f, v);
    return b;
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

// ================================================================== config
bool Config::loadFile(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open config file " + path; return false; }
    std::string line;
    int ln = 0;
    while (std::getline(f, line)) {
        ++ln;
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        line = trim(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            err = path + ":" + std::to_string(ln) + ": expected key = value";
            return false;
        }
        set(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
    }
    return true;
}
void Config::set(const std::string& key, const std::string& value) { kv_[lower(key)] = value; }
bool Config::has(const std::string& key) const {
    markUsed(key);
    return kv_.count(lower(key)) > 0;
}
std::string Config::str(const std::string& key, const std::string& def) const {
    markUsed(key);
    auto it = kv_.find(lower(key));
    return it == kv_.end() ? def : it->second;
}
double Config::num(const std::string& key, double def) const {
    std::string s = str(key);
    return s.empty() ? def : atof(s.c_str());
}
long long Config::integer(const std::string& key, long long def) const {
    std::string s = str(key);
    return s.empty() ? def : (long long)llround(atof(s.c_str()));
}
std::vector<std::string> Config::keysWithPrefix(const std::string& prefix) const {
    std::vector<std::string> out;
    std::string p = lower(prefix);
    for (auto& kv : kv_)
        if (kv.first.compare(0, p.size(), p) == 0) { out.push_back(kv.first); markUsed(kv.first); }
    return out;
}
void Config::markUsed(const std::string& key) const { used_[lower(key)] = true; }
std::vector<std::string> Config::unusedKeys() const {
    std::vector<std::string> out;
    for (auto& kv : kv_)
        if (!used_.count(kv.first) && kv.first[0] != '_') out.push_back(kv.first);
    return out;
}

// ============================================================ param files
bool readParamFile(const std::string& path, std::vector<float>& out, size_t minCount, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    long long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = (size_t)(bytes / 4);
    out.assign(std::max(n, minCount), 0.0f);
    size_t got = n ? fread(out.data(), sizeof(float), n, f) : 0;
    fclose(f);
    if (got != n) { err = "short read on " + path; return false; }
    return true;
}

// ============================================================= material db
bool MaterialDb::load(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open material database " + path; return false; }
    std::string line;
    DbMaterial* cur = nullptr;
    auto readVec = [](std::istringstream& is, std::vector<float>& v) {
        float x;
        while (is >> x) v.push_back(x);
    };
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "MCDB") continue;
        if (tag == "EGRID") { is >> nE >> eMin >> eMax; }
        else if (tag == "FFGRID") { is >> nFF >> x2Min >> x2Max; }
        else if (tag == "MATERIAL") {
            mats.emplace_back();
            cur = &mats.back();
            is >> cur->name;
        } else if (!cur) {
            continue;
        } else if (tag == "DENSITY") { is >> cur->density; }
        else if (tag == "MU_PE") readVec(is, cur->pe);
        else if (tag == "MU_INC") readVec(is, cur->inc);
        else if (tag == "MU_COH") readVec(is, cur->coh);
        else if (tag == "FLUO") {
            std::string label; int z; float e;
            is >> label >> z >> e;
            cur->fluoLabel.push_back(label);
            cur->fluoZ.push_back(z);
            cur->fluoE.push_back(e);
            cur->fluoP.emplace_back();
            readVec(is, cur->fluoP.back());
        } else if (tag == "FF_CUM") readVec(is, cur->ffCum);
        else if (tag == "END") cur = nullptr;
    }
    if (nE < 2 || mats.empty()) { err = "material database " + path + " is empty or malformed"; return false; }
    for (auto& m : mats) {
        bool ok = (int)m.pe.size() == nE && (int)m.inc.size() == nE && (int)m.coh.size() == nE &&
                  (int)m.ffCum.size() == nFF && m.density > 0;
        for (auto& p : m.fluoP) ok = ok && (int)p.size() == nE;
        if (!ok) { err = "material " + m.name + " in " + path + " is malformed"; return false; }
    }
    return true;
}

const DbMaterial* MaterialDb::find(const std::string& name) const {
    for (auto& m : mats)
        if (lower(m.name) == lower(name)) return &m;
    return nullptr;
}

// ============================================================ setup pieces
namespace {

double kleinNishinaTotal(double E) {  // relative units
    double k = E / kElectronMass;
    double l = std::log(1.0 + 2.0 * k);
    return (1.0 + k) / (k * k) * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - l / k) + l / (2.0 * k) -
           (1.0 + 3.0 * k) / ((1.0 + 2.0 * k) * (1.0 + 2.0 * k));
}

struct MaterialRegistry {
    Simulation& S;
    std::map<std::string, int> index;   // key -> material index
    explicit MaterialRegistry(Simulation& s) : S(s) {}

    double gridE(int i) const {
        return S.hm.eMin * std::exp(std::log(S.hm.eMax / S.hm.eMin) * i / (S.hm.nE - 1));
    }
    int addRaw(const std::string& key, const std::string& name) {
        int id = (int)S.hm.names.size();
        S.hm.names.push_back(name);
        size_t nE = S.hm.nE;
        S.hm.muPE.resize((id + 1) * nE);
        S.hm.muInc.resize((id + 1) * nE);
        S.hm.muCoh.resize((id + 1) * nE);
        S.hm.nFluo.push_back(0);
        S.hm.fluoE.resize((id + 1) * kMaxFluo, 0.0f);
        S.hm.fluoP.resize((id + 1) * nE * kMaxFluo, 0.0f);
        S.hm.ffCum.resize((id + 1) * S.hm.nFF, 0.0f);
        S.hm.hasFF.push_back(0);
        index[key] = id;
        return id;
    }
    // legacy: coefficients given at E0 in the parameter files (1/mm)
    int fromParams(float tot, float pe, float c) {
        if (tot <= 0.0f && pe <= 0.0f && c <= 0.0f) return -1;
        char key[128];
        snprintf(key, sizeof(key), "params:%.6g/%.6g/%.6g", tot, pe, c);
        auto it = index.find(key);
        if (it != index.end()) return it->second;
        int id = addRaw(key, std::string("params(tot=") + fmt("%.4g", tot) + ",pe=" + fmt("%.4g", pe) +
                                 ",compton=" + fmt("%.4g", c) + "/mm @E0)");
        double E0 = S.C.E0;
        double kn0 = kleinNishinaTotal(E0);
        double rem = tot - pe - c;
        for (int i = 0; i < S.hm.nE; ++i) {
            double E = gridE(i), r = E0 / E;
            double mpe, minc, mcoh = 0;
            if (pe + c <= 0.0f) { mpe = tot * r * r * r; minc = 0; }
            else {
                mpe = pe * r * r * r;
                minc = c * kleinNishinaTotal(E) / kn0;
                if (rem > 1e-6 * tot) mcoh = rem * r * r;
            }
            S.hm.muPE[id * S.hm.nE + i] = (float)mpe;
            S.hm.muInc[id * S.hm.nE + i] = (float)minc;
            S.hm.muCoh[id * S.hm.nE + i] = (float)mcoh;
        }
        return id;
    }
    int fromDb(const std::string& name, std::string& err) {
        std::string key = "db:" + lower(name);
        auto it = index.find(key);
        if (it != index.end()) return it->second;
        if (!S.dbLoaded) { err = "material '" + name + "' needs the material database (material_db)"; return -2; }
        const DbMaterial* m = S.db.find(name);
        if (!m) { err = "material '" + name + "' not found in material database"; return -2; }
        double rho = S.cfg.num("density_" + lower(name), m->density);
        int id = addRaw(key, m->name + " (rho=" + fmt("%.4g", rho) + " g/cm3)");
        double s = rho / 10.0;  // cm^2/g * g/cm^3 -> 1/cm -> 1/mm
        for (int i = 0; i < S.hm.nE; ++i) {
            S.hm.muPE[id * S.hm.nE + i] = (float)(m->pe[i] * s);
            S.hm.muInc[id * S.hm.nE + i] = (float)(m->inc[i] * s);
            S.hm.muCoh[id * S.hm.nE + i] = (float)(m->coh[i] * s);
        }
        int nl = std::min((int)m->fluoE.size(), kMaxFluo);
        if ((int)m->fluoE.size() > kMaxFluo)
            S.warnings.push_back("material " + m->name + ": only the first " + std::to_string(kMaxFluo) + " K lines are used");
        S.hm.nFluo[id] = nl;
        for (int l = 0; l < nl; ++l) {
            S.hm.fluoE[id * kMaxFluo + l] = m->fluoE[l];
            for (int i = 0; i < S.hm.nE; ++i)
                S.hm.fluoP[((size_t)id * S.hm.nE + i) * kMaxFluo + l] = m->fluoP[l][i];
        }
        for (int k = 0; k < S.hm.nFF; ++k) S.hm.ffCum[id * S.hm.nFF + k] = m->ffCum[k];
        S.hm.hasFF[id] = 1;
        return id;
    }
    // spec: "params" | "vacuum" | <db material name>
    int resolve(const std::string& spec, float tot, float pe, float c, std::string& err) {
        std::string s = lower(trim(spec));
        if (s.empty() || s == "params") return fromParams(tot, pe, c);
        if (s == "vacuum" || s == "none") return -1;
        int id = fromDb(spec, err);
        if (id >= 0 && tot > 0.0f) {  // consistency check against the parameter file
            int nE = S.hm.nE;
            double t = (std::log(S.C.E0) - std::log(S.hm.eMin)) / std::log(S.hm.eMax / S.hm.eMin) * (nE - 1);
            int i = std::min(std::max((int)t, 0), nE - 2);
            double f = t - i;
            auto at = [&](const std::vector<float>& v) { return v[id * nE + i] * (1 - f) + v[id * nE + i + 1] * f; };
            double dbTot = at(S.hm.muPE) + at(S.hm.muInc);  // parameter files exclude Rayleigh
            if (std::fabs(dbTot - tot) > 0.15 * tot) {
                std::string w = "material " + spec + ": mu(PE+Compton) at " + fmt("%.1f", S.C.E0) + " keV is " +
                                fmt("%.4g", dbTot) + "/mm but the parameter file says " + fmt("%.4g", tot) + "/mm";
                if (std::find(S.warnings.begin(), S.warnings.end(), w) == S.warnings.end()) S.warnings.push_back(w);
            }
        }
        return id;
    }
};

// CSR builder for uniform grids
struct CsrBuilder {
    std::vector<int> count;
    std::vector<std::pair<int, int>> pairs;  // (cell, item)
    void add(int cell, int item) { pairs.push_back({cell, item}); }
    void build(int cellBase, int nCells, std::vector<int>& start, std::vector<int>& items) {
        // start has cellBase entries already; append nCells entries + keep a trailing sentinel
        std::vector<int> cnt(nCells, 0);
        for (auto& p : pairs) cnt[p.first]++;
        if (start.empty()) start.push_back(0);
        int base = start.back();
        std::vector<int> off(nCells + 1, 0);
        for (int i = 0; i < nCells; ++i) off[i + 1] = off[i] + cnt[i];
        for (int i = 1; i <= nCells; ++i) start.push_back(base + off[i]);
        items.resize(base + off[nCells]);
        std::vector<int> fill(nCells, 0);
        std::stable_sort(pairs.begin(), pairs.end());
        for (auto& p : pairs) items[base + off[p.first] + fill[p.first]++] = p.second;
        (void)cellBase;
    }
};

}  // namespace

// ================================================================== setup
std::string Simulation::tallyName(int t) const {
    int nCls = C.components ? kNumClasses : 1;
    int w = t / nCls, c = t % nCls;
    static const char* cls[] = {"primary", "colscatter", "xtalscatter"};
    std::string s = "win" + std::to_string(w);
    if (C.components) s += std::string("_") + cls[c];
    return s;
}

bool Simulation::setup(std::string& err) {
    // ------------------------------------------------ parameter files
    std::string dir = cfg.str("params_dir", ".");
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += "/";
    params.dir = dir;
    auto loadP = [&](const char* name, std::vector<float>& v, size_t n) {
        std::string e1, e2;
        if (readParamFile(dir + "Params_" + name + ".dat", v, n, e1)) return true;
        if (readParamFile(dir + "Param_" + name + ".dat", v, n, e2)) return true;
        err = e1;
        return false;
    };
    if (!loadP("Collimator", params.col, 80000) || !loadP("Detector", params.det, 80000) ||
        !loadP("Image", params.img, 100) || !loadP("Physics", params.phy, 100))
        return false;
    const auto& P = params;

    // ------------------------------------------------ energy / preset / windows
    std::string preset = lower(cfg.str("preset", "none"));
    double presetE = 0;
    std::string presetWin;
    if (preset == "tc99m" || preset == "140" || preset == "140kev") { presetE = 140.511; presetWin = "126.46-154.56"; }
    else if (preset == "511" || preset == "f18" || preset == "511kev" || preset == "pet") { presetE = 511.0; presetWin = "459.9-562.1"; }
    else if (preset != "none" && !preset.empty()) { err = "unknown preset '" + preset + "' (tc99m | 511 | none)"; return false; }
    C.E0 = (float)cfg.num("energy_kev", presetE > 0 ? presetE : P.phy[7]);
    if (C.E0 <= 0.0f) { err = "photon energy unknown: set energy_keV or preset, or Param_Physics[7]"; return false; }
    if (C.E0 > 1000.0f) { err = "energy above 1 MeV is outside the validity range of this code"; return false; }

    std::string winSpec = cfg.str("windows", presetWin.empty() ? "params" : presetWin);
    C.nWin = 0;
    for (auto& tok : split(winSpec, ",;")) {
        if (C.nWin >= kMaxWindows) { err = "at most 4 energy windows"; return false; }
        int k = C.nWin;
        C.winAuto[k] = 0;
        std::string t = lower(tok);
        if (t == "auto") { C.winAuto[k] = 1; C.winLo[k] = C.winHi[k] = 0; }
        else if (t == "params") {
            if (P.phy[4] > 0.5f && P.phy[6] > P.phy[5]) { C.winLo[k] = P.phy[5]; C.winHi[k] = P.phy[6]; }
            else C.winAuto[k] = 1;
        } else {
            auto lh = split(t, "-:");
            if (lh.size() != 2) { err = "bad energy window '" + tok + "' (use lo-hi in keV)"; return false; }
            C.winLo[k] = (float)atof(lh[0].c_str());
            C.winHi[k] = (float)atof(lh[1].c_str());
            if (!(C.winHi[k] > C.winLo[k])) { err = "bad energy window '" + tok + "'"; return false; }
        }
        ++C.nWin;
    }
    if (C.nWin == 0) { err = "no energy window"; return false; }
    std::string wm = lower(cfg.str("ewin_mode", "gauss"));
    C.winMode = wm == "ideal" ? kWinIdeal : wm == "none" ? kWinNone : kWinGauss;
    if (wm != "gauss" && wm != "ideal" && wm != "none") { err = "ewin_mode must be gauss | ideal | none"; return false; }
    C.eresExp = (float)cfg.num("eres_exponent", 0.5);

    // ------------------------------------------------ image / rotations
    C.nX = (int)std::floor(P.img[0] + 0.001f);
    C.nY = (int)std::floor(P.img[1] + 0.001f);
    C.nZ = (int)std::floor(P.img[2] + 0.001f);
    C.wX = P.img[3]; C.wY = P.img[4]; C.wZ = P.img[5];
    numRot = std::max(1, (int)std::floor(P.img[6] + 0.001f));
    C.angPerRot = P.img[7];
    C.shX = P.img[8]; C.shY = P.img[9]; C.shZ = P.img[10];
    fov2col = P.img[11];
    numVox = C.nX * C.nY * C.nZ;
    if (numVox <= 0) { err = "image has no voxels (Param_Image[0..2])"; return false; }
    rotBegin = (int)cfg.integer("rot_begin", 0);
    rotEnd = (int)cfg.integer("rot_end", numRot);
    if (rotBegin < 0 || rotEnd > numRot || rotBegin >= rotEnd) { err = "bad rotation range"; return false; }

    // ------------------------------------------------ run control
    histPerVoxel = cfg.integer("histories_per_voxel", 100000);
    if (histPerVoxel <= 0 || histPerVoxel > 4000000000LL) { err = "histories_per_voxel must be in [1, 4e9]"; return false; }
    C.seed = (uint64_t)cfg.integer("seed", 20241221);
    std::string ss = lower(cfg.str("source_sampling", "uniform"));
    C.srcUniform = ss != "center" && ss != "centre";
    std::string ds = lower(cfg.str("direction_sampling", "rect"));
    C.dirRect = ds != "iso" && ds != "isotropic";
    // hole_bias / detector_bias: "auto" (default) is resolved once the collimator is known
    std::string hbSpec = lower(cfg.str("hole_bias", "auto")), dbSpec = lower(cfg.str("detector_bias", "auto"));
    C.tgtMargin = (float)cfg.num("hole_target_margin", 0.25);
    std::string phys = lower(cfg.str("physics", "full"));
    if (phys != "full" && phys != "primary_pe") { err = "physics must be full | primary_pe"; return false; }
    C.primaryOnly = phys == "primary_pe";
    C.fluo = !C.primaryOnly && cfg.integer("fluorescence", 1) != 0;
    bool rayleigh = cfg.integer("rayleigh", 1) != 0;
    C.eCut = (float)cfg.num("energy_cutoff_kev", 10.0);
    C.components = cfg.integer("save_components", 0) != 0;
    saveRelErr = cfg.integer("save_relerr", 0) != 0;
    yield = (float)cfg.num("yield", 1.0);
    C.nSpecBins = (int)cfg.integer("spectrum_bins", 1024);
    C.nSpecBins = std::min(std::max(C.nSpecBins, 16), 1536);
    C.specBinW = (float)(cfg.num("spectrum_emax_kev", 1.25 * C.E0) / C.nSpecBins);
    nTally = C.nWin * (C.components ? kNumClasses : 1);

    // ------------------------------------------------ materials
    std::string dbPath = cfg.str("material_db", "");
    if (dbPath.empty()) {
        // default: data/materials.db next to the executable, then in the cwd
        std::string exeDir = cfg.str("_exe_dir", ".");
        for (std::string cand : {exeDir + "/data/materials.db", std::string("data/materials.db")}) {
            std::ifstream t(cand);
            if (t) { dbPath = cand; break; }
        }
    }
    if (!dbPath.empty() && lower(dbPath) != "none") {
        std::string e;
        if (!db.load(dbPath, e)) {
            if (cfg.has("material_db")) { err = e; return false; }
            warnings.push_back(e);
        } else dbLoaded = true;
    }
    if (dbLoaded) {
        hm.nE = db.nE; hm.eMin = db.eMin; hm.eMax = db.eMax;
        hm.nFF = db.nFF; hm.x2Min = db.x2Min; hm.x2Max = db.x2Max;
    } else {
        hm.nE = 1024; hm.eMin = 1.0; hm.eMax = 1100.0;
        hm.nFF = 2; hm.x2Min = 1e-4; hm.x2Max = 1e4;
    }
    hm.ffX2.resize(hm.nFF);
    for (int k = 0; k < hm.nFF; ++k)
        hm.ffX2[k] = (float)(hm.x2Min * std::pow(hm.x2Max / hm.x2Min, (double)k / (hm.nFF - 1)));
    MaterialRegistry reg(*this);

    // ------------------------------------------------ collimator layers + holes
    Geometry& g = hg.g;
    g.nLayers = (int)std::floor(P.col[0] + 0.001f);
    if (g.nLayers < 0 || g.nLayers > kMaxLayers) { err = "numCollimatorLayers must be 0..10"; return false; }
    std::string shield = lower(cfg.str("shield_outside_collimator", "absorber"));
    if (shield == "1" || shield == "absorber") g.shieldMode = 1;
    else if (shield == "2" || shield == "slab") g.shieldMode = 2;
    else if (shield == "0" || shield == "none") g.shieldMode = 0;
    else { err = "shield_outside_collimator must be absorber | slab | none"; return false; }
    int holeStart = 0;
    std::string colDefault = cfg.str("collimator_material", "params");
    std::string holeSpec = cfg.str("hole_material", "params");
    for (int m = 0; m < g.nLayers; ++m) {
        int b = (m + 1) * 10;
        int nh = (int)std::floor(P.col[b] + 0.001f);
        float W = P.col[b + 1], T = P.col[b + 2], H = P.col[b + 3], off = P.col[b + 4];
        g.layXh[m] = W / 2; g.layZh[m] = H / 2;
        g.layY1[m] = fov2col + off - T / 2;
        g.layY2[m] = fov2col + off + T / 2;
        std::string spec = cfg.str("collimator_material_layer" + std::to_string(m), colDefault);
        g.layMat[m] = reg.resolve(spec, P.col[b + 5], P.col[b + 6], P.col[b + 7], err);
        if (g.layMat[m] < -1) return false;
        materialReport.push_back("collimator layer " + std::to_string(m) + ": " +
                                 (g.layMat[m] >= 0 ? hm.names[g.layMat[m]] : std::string("vacuum")) + ", " +
                                 std::to_string(nh) + " holes, plate " + fmt("%.3g", W) + " x " + fmt("%.3g", T) +
                                 " x " + fmt("%.3g", H) + " mm at y=[" + fmt("%.4g", g.layY1[m]) + "," + fmt("%.4g", g.layY2[m]) + "]");
        // holes
        size_t need = 100 + (size_t)(holeStart + nh) * 9;
        if (need > P.col.size()) { err = "Params_Collimator is too short for the declared number of holes"; return false; }
        float maxR = 0, xmin = -W / 2, xmax = W / 2, zmin = -H / 2, zmax = H / 2;
        for (int h = holeStart; h < holeStart + nh; ++h) {
            const float* q = &P.col[100 + (size_t)h * 9];
            float y1 = q[1] + fov2col, y2 = q[2] + fov2col;
            if (y1 > y2) std::swap(y1, y2);
            float r = std::fabs(q[4]);
            hg.hole0.push_back({q[0], q[3], r * r, y1});
            hg.hole1.push_back({y2, r, 0.0f, 0.0f});
            int hm_ = reg.resolve(holeSpec, q[5], q[6], q[7], err);
            if (hm_ < -1) return false;
            hg.holeMat.push_back(hm_);
            hg.holeLayer.push_back(m);
            maxR = std::max(maxR, r);
            xmin = std::min(xmin, q[0] - r); xmax = std::max(xmax, q[0] + r);
            zmin = std::min(zmin, q[3] - r); zmax = std::max(zmax, q[3] + r);
        }
        // hole grid for this layer
        float cell = std::max(2.0f * maxR, 0.25f);
        if (nh == 0) cell = std::max(xmax - xmin, zmax - zmin) + 1.0f;  // single empty cell
        int nx, nz;
        for (;;) {
            nx = (int)std::ceil((xmax - xmin) / cell) + 2;
            nz = (int)std::ceil((zmax - zmin) / cell) + 2;
            if ((long long)nx * nz <= 4000000LL) break;
            cell *= 1.5f;
        }
        g.hgOx[m] = xmin - cell; g.hgOz[m] = zmin - cell;
        g.hgInvCell[m] = 1.0f / cell;
        g.hgNx[m] = nx; g.hgNz[m] = nz;
        g.hgBase[m] = hg.hgCellStart.empty() ? 0 : (int)hg.hgCellStart.size() - 1;
        CsrBuilder cb;
        for (int h = holeStart; h < holeStart + nh; ++h) {
            F4 a = hg.hole0[h];
            float r = hg.hole1[h].y;
            int x0 = (int)std::floor((a.x - r - g.hgOx[m]) / cell), x1 = (int)std::floor((a.x + r - g.hgOx[m]) / cell);
            int z0 = (int)std::floor((a.y - r - g.hgOz[m]) / cell), z1 = (int)std::floor((a.y + r - g.hgOz[m]) / cell);
            for (int cx = std::max(0, x0); cx <= std::min(nx - 1, x1); ++cx)
                for (int cz = std::max(0, z0); cz <= std::min(nz - 1, z1); ++cz) cb.add(cx * nz + cz, h);
        }
        cb.build(g.hgBase[m], nx * nz, hg.hgCellStart, hg.hgItems);
        holeStart += nh;
    }
    hg.numHoles = holeStart;
    if (hg.hgCellStart.empty()) hg.hgCellStart.push_back(0);
    if (P.col.size() > 80000 && holeStart * 9 + 100 > 80000)
        warnings.push_back("more holes than the analytic modules can read (80000 floats)");

    // ------------------------------------------------ crystals
    numDet = (int)std::floor(P.det[0] + 0.001f);
    g.nDet = numDet;
    if (numDet <= 0) { err = "no detector bins (Param_Detector[0])"; return false; }
    if ((size_t)numDet * 12 + 13 > P.det.size()) { err = "Params_Detector is too short for the declared number of detectors"; return false; }
    std::string xtalDefault = cfg.str("crystal_material", "params");
    double eresOverride = cfg.num("eres", -1.0);
    float bmin[3] = {kBig, kBig, kBig}, bmax[3] = {-kBig, -kBig, -kBig};
    std::vector<float> minDim;
    std::map<std::string, int> matCount;
    for (int i = 0; i < numDet; ++i) {
        const float* q = &P.det[(size_t)12 * i];
        float hw = q[4] / 2, ht = q[5] / 2, hh = q[6] / 2, rot = q[11];
        hg.xtal0.push_back({q[1], q[2] + fov2col, q[3], std::cos(rot)});
        hg.xtal1.push_back({hw, ht, hh, std::sin(rot)});
        hg.xtalEres.push_back(eresOverride >= 0 ? (float)eresOverride : q[10]);
        int mat = -1;
        if (q[7] > 0.01f) {  // same "enabled" rule as the analytic code
            int flag = (int)std::lround(q[12]);
            std::string spec = cfg.str("crystal_material_flag" + std::to_string(flag), xtalDefault);
            mat = reg.resolve(spec, q[7], q[8], q[9], err);
            if (mat < -1) return false;
            if (mat >= 0) {
                matCount[hm.names[mat]]++;
                hg.numEnabledXtals++;
                float ex = std::fabs(std::cos(rot)) * hw + std::fabs(std::sin(rot)) * hh;
                float ez = std::fabs(std::sin(rot)) * hw + std::fabs(std::cos(rot)) * hh;
                float c[3] = {q[1], q[2] + fov2col, q[3]}, e[3] = {ex, ht, ez};
                for (int a = 0; a < 3; ++a) { bmin[a] = std::min(bmin[a], c[a] - e[a]); bmax[a] = std::max(bmax[a], c[a] + e[a]); }
                minDim.push_back(std::min(q[4], std::min(q[5], q[6])));
            }
        }
        hg.xtalMat.push_back(mat);
    }
    for (auto& kv : matCount) materialReport.push_back("crystals: " + std::to_string(kv.second) + " x " + kv.first);
    if (C.winMode == kWinGauss) {
        int noRes = 0;
        for (int i = 0; i < numDet; ++i) noRes += hg.xtalMat[i] >= 0 && !(hg.xtalEres[i] > 0.0f);
        bool autoWin = false;
        for (int k = 0; k < C.nWin; ++k) autoWin = autoWin || C.winAuto[k];
        if (noRes)
            warnings.push_back(std::to_string(noRes) + " enabled crystal(s) have zero energy resolution: " +
                               (autoWin ? "their per-crystal window collapses to E0 (set eres)" : "an ideal window is used for them"));
    }
    if (hg.numEnabledXtals == 0) { err = "no enabled crystals (total attenuation > 0.01/mm required)"; return false; }

    // crystal grid
    std::sort(minDim.begin(), minDim.end());
    float cell = std::min(std::max(minDim[minDim.size() / 2], 0.25f), 50.0f);
    int gx, gy, gz;
    for (;;) {
        gx = (int)std::ceil((bmax[0] - bmin[0]) / cell) + 2;
        gy = (int)std::ceil((bmax[1] - bmin[1]) / cell) + 2;
        gz = (int)std::ceil((bmax[2] - bmin[2]) / cell) + 2;
        if ((long long)gx * gy * gz <= 16000000LL) break;
        cell *= 1.3f;
    }
    g.cgOx = bmin[0] - cell; g.cgOy = bmin[1] - cell; g.cgOz = bmin[2] - cell;
    g.cgInvCell = 1.0f / cell;
    g.cgNx = gx; g.cgNy = gy; g.cgNz = gz;
    {
        CsrBuilder cb;
        for (int i = 0; i < numDet; ++i) {
            if (hg.xtalMat[i] < 0) continue;
            F4 a = hg.xtal0[i], b = hg.xtal1[i];
            float ex = std::fabs(a.w) * b.x + std::fabs(b.w) * b.z;
            float ez = std::fabs(b.w) * b.x + std::fabs(a.w) * b.z;
            int x0 = (int)std::floor((a.x - ex - g.cgOx) / cell), x1 = (int)std::floor((a.x + ex - g.cgOx) / cell);
            int y0 = (int)std::floor((a.y - b.y - g.cgOy) / cell), y1 = (int)std::floor((a.y + b.y - g.cgOy) / cell);
            int z0 = (int)std::floor((a.z - ez - g.cgOz) / cell), z1 = (int)std::floor((a.z + ez - g.cgOz) / cell);
            for (int cx = std::max(0, x0); cx <= std::min(gx - 1, x1); ++cx)
                for (int cy = std::max(0, y0); cy <= std::min(gy - 1, y1); ++cy)
                    for (int cz = std::max(0, z0); cz <= std::min(gz - 1, z1); ++cz) cb.add((cx * gy + cy) * gz + cz, i);
        }
        cb.build(0, gx * gy * gz, hg.cgCellStart, hg.cgItems);
    }

    // ------------------------------------------------ regions + majorants
    g.nRegions = 0;
    std::vector<std::vector<int>> regMats;
    // slab-mode frame: by default it covers the crystal block laterally plus a margin
    double frame = cfg.num("shield_frame_margin_mm", 20.0);
    for (int m = 0; m < g.nLayers; ++m) {
        float fx = (float)std::max<double>(std::max(std::fabs(bmin[0]), std::fabs(bmax[0])) + frame, g.layXh[m]);
        float fz = (float)std::max<double>(std::max(std::fabs(bmin[2]), std::fabs(bmax[2])) + frame, g.layZh[m]);
        g.shXh[m] = (float)cfg.num("shield_frame_halfwidth_mm", fx);
        g.shZh[m] = (float)cfg.num("shield_frame_halfheight_mm", fz);
        float ex = g.shieldMode == 1 ? 1e6f : g.shieldMode == 2 ? g.shXh[m] : g.layXh[m];
        float ez = g.shieldMode == 1 ? 1e6f : g.shieldMode == 2 ? g.shZh[m] : g.layZh[m];
        int r = g.nRegions++;
        g.regMin[r][0] = -ex; g.regMax[r][0] = ex;
        g.regMin[r][1] = g.layY1[m]; g.regMax[r][1] = g.layY2[m];
        g.regMin[r][2] = -ez; g.regMax[r][2] = ez;
        g.regLayer[r] = m;
        std::vector<int> mats = {g.layMat[m]};
        for (int h = 0; h < hg.numHoles; ++h)
            if (hg.holeLayer[h] == m) mats.push_back(hg.holeMat[h]);
        regMats.push_back(mats);
    }
    {   // detector regions: cluster enabled crystals along y
        struct Iv { float y0, y1; int i; };
        std::vector<Iv> iv;
        for (int i = 0; i < numDet; ++i)
            if (hg.xtalMat[i] >= 0) iv.push_back({hg.xtal0[i].y - hg.xtal1[i].y, hg.xtal0[i].y + hg.xtal1[i].y, i});
        std::sort(iv.begin(), iv.end(), [](const Iv& a, const Iv& b) { return a.y0 < b.y0; });
        std::vector<std::vector<int>> clusters;
        float yEnd = -kBig;
        for (auto& v : iv) {
            if (clusters.empty() || v.y0 > yEnd + 2.0f) clusters.emplace_back();
            clusters.back().push_back(v.i);
            yEnd = std::max(yEnd, v.y1);
        }
        if ((int)clusters.size() > kMaxRegions - g.nRegions) {
            std::vector<int> all;
            for (auto& c : clusters) all.insert(all.end(), c.begin(), c.end());
            clusters.assign(1, all);
        }
        for (auto& cl : clusters) {
            int r = g.nRegions++;
            for (int a = 0; a < 3; ++a) { g.regMin[r][a] = kBig; g.regMax[r][a] = -kBig; }
            std::vector<int> mats;
            for (int i : cl) {
                F4 a = hg.xtal0[i], b = hg.xtal1[i];
                float ex = std::fabs(a.w) * b.x + std::fabs(b.w) * b.z;
                float ez = std::fabs(b.w) * b.x + std::fabs(a.w) * b.z;
                float c[3] = {a.x, a.y, a.z}, e[3] = {ex, b.y, ez};
                for (int k = 0; k < 3; ++k) {
                    g.regMin[r][k] = std::min(g.regMin[r][k], c[k] - e[k] - 1e-3f);
                    g.regMax[r][k] = std::max(g.regMax[r][k], c[k] + e[k] + 1e-3f);
                }
                mats.push_back(hg.xtalMat[i]);
            }
            g.regLayer[r] = -1;
            regMats.push_back(mats);
        }
    }

    // finish material tables (Rayleigh switch, totals, majorants)
    size_t nM = hm.names.size();
    if (!rayleigh) std::fill(hm.muCoh.begin(), hm.muCoh.end(), 0.0f);
    hm.muTot.resize(nM * hm.nE);
    for (size_t i = 0; i < nM * hm.nE; ++i) hm.muTot[i] = hm.muPE[i] + hm.muInc[i] + hm.muCoh[i];
    hm.regMaj.assign((size_t)g.nRegions * hm.nE, 0.0f);
    for (int r = 0; r < g.nRegions; ++r) {
        std::sort(regMats[r].begin(), regMats[r].end());
        regMats[r].erase(std::unique(regMats[r].begin(), regMats[r].end()), regMats[r].end());
        for (int m : regMats[r]) {
            if (m < 0) continue;
            for (int i = 0; i < hm.nE; ++i)
                hm.regMaj[(size_t)r * hm.nE + i] = std::max(hm.regMaj[(size_t)r * hm.nE + i], hm.muTot[(size_t)m * hm.nE + i]);
        }
    }
    if (nM == 0) { err = "no materials (all attenuation coefficients are zero?)"; return false; }

    // hole-aperture targeting: the front-most layer with holes (or hole_target_layer)
    g.tgtLayer = -1;
    g.tgtTan = 0.0f; g.tgtRmax = 0.0f; g.tgtAcc = 1.5f; g.tgtHoleStart = 0; g.tgtHoleCount = 0;
    {
        int want = (int)cfg.integer("hole_target_layer", -1);
        float bestY = kBig;
        for (int m = 0; m < g.nLayers; ++m) {
            bool hasHoles = false;
            for (int h = 0; h < hg.numHoles; ++h) hasHoles = hasHoles || hg.holeLayer[h] == m;
            if (!hasHoles) continue;
            if (want >= 0 ? m == want : g.layY1[m] < bestY) { g.tgtLayer = m; bestY = g.layY1[m]; }
        }
        if (g.tgtLayer >= 0) {
            int m = g.tgtLayer;
            float acc = (float)cfg.num("hole_target_acceptance", 1.5);  // x geometric channel acceptance
            g.tgtAcc = acc;
            g.tgtHoleStart = -1;
            for (int h = 0; h < hg.numHoles; ++h) {
                if (hg.holeLayer[h] != m) continue;
                if (g.tgtHoleStart < 0) g.tgtHoleStart = h;
                g.tgtHoleCount++;
                float r = hg.hole1[h].y;
                float len = std::min(hg.hole1[h].x, g.layY2[m]) - std::max(hg.hole0[h].w, g.layY1[m]);
                float tanA = len > 1e-3f ? acc * 2.0f * r / len : 10.0f;
                g.tgtTan = std::max(g.tgtTan, std::min(tanA, 10.0f));
                g.tgtRmax = std::max(g.tgtRmax, r);
            }
        }
    }

    // Sampling mixture defaults (figure-of-merit study, see ReadMe): aiming at hole apertures
    // gains 5-10x when the target plate is opaque (mu*T >= 5, e.g. W at 140 keV) but loses ~2x
    // when septal penetration dominates (W at 511 keV, thin plates).
    {
        double opacity = 0.0;
        if (g.tgtLayer >= 0 && g.layMat[g.tgtLayer] >= 0) {
            int m = g.tgtLayer, mat = g.layMat[m];
            double t = (std::log(C.E0) - std::log(hm.eMin)) / std::log(hm.eMax / hm.eMin) * (hm.nE - 1);
            int i = std::min(std::max((int)t, 0), hm.nE - 2);
            double f = t - i;
            double mu = hm.muTot[(size_t)mat * hm.nE + i] * (1 - f) + hm.muTot[(size_t)mat * hm.nE + i + 1] * f;
            opacity = mu * (g.layY2[m] - g.layY1[m]);
        }
        C.holeBias = hbSpec == "auto" ? (opacity >= 5.0 ? 0.5f : 0.0f) : (float)atof(hbSpec.c_str());
        C.detBias = dbSpec == "auto" ? (C.holeBias > 0.0f ? 0.3f : 0.8f) : (float)atof(dbSpec.c_str());
        if (C.detBias < 0.0f || C.holeBias < 0.0f || C.detBias + C.holeBias > 0.95f) {
            err = "detector_bias and hole_bias must be >= 0 with detector_bias + hole_bias <= 0.95";
            return false;
        }
        if (g.tgtLayer >= 0)
            materialReport.push_back("hole targeting: layer " + std::to_string(g.tgtLayer) + ", mu*T = " + fmt("%.3g", opacity) +
                                     (hbSpec == "auto" ? (C.holeBias > 0 ? " (opaque -> on)" : " (penetrable -> off)") : " (set by hole_bias)"));
    }

    // assembly box (finite plates + enabled crystals) and crystal box
    for (int a = 0; a < 3; ++a) {
        g.asmMin[a] = g.detMin[a] = bmin[a];
        g.asmMax[a] = g.detMax[a] = bmax[a];
    }
    for (int m = 0; m < g.nLayers; ++m) {
        float ex = g.shieldMode == 2 ? g.shXh[m] : g.layXh[m], ez = g.shieldMode == 2 ? g.shZh[m] : g.layZh[m];
        g.asmMin[0] = std::min(g.asmMin[0], -ex); g.asmMax[0] = std::max(g.asmMax[0], ex);
        g.asmMin[1] = std::min(g.asmMin[1], g.layY1[m]);  g.asmMax[1] = std::max(g.asmMax[1], g.layY2[m]);
        g.asmMin[2] = std::min(g.asmMin[2], -ez); g.asmMax[2] = std::max(g.asmMax[2], ez);
    }
    if (g.shieldMode == 2)
        for (int m = 0; m < g.nLayers; ++m)
            materialReport.push_back("collimator layer " + std::to_string(m) + ": frame of the same material out to |x| <= " +
                                     fmt("%.4g", g.shXh[m]) + ", |z| <= " + fmt("%.4g", g.shZh[m]) + " mm");

    // ------------------------------------------------ backend / tracing / output
    backend = lower(cfg.str("backend", "gpu"));
    if (backend != "gpu" && backend != "cpu") { err = "backend must be gpu | cpu"; return false; }
    std::string gl = lower(cfg.str("gpus", "all"));
    if (gl != "all")
        for (auto& t : split(gl, ", ")) gpus.push_back(atoi(t.c_str()));
    cpuThreads = (int)cfg.integer("cpu_threads", 0);
    threadsPerVoxel = (int)cfg.integer("threads_per_voxel", 0);
    voxelChunk = cfg.integer("voxel_chunk", 0);
    launchTargetMs = cfg.num("launch_target_ms", 300.0);

    traceVoxel = (int)cfg.integer("trace_voxel", -1);
    if (traceVoxel == -2)  // convenience: central voxel
        traceVoxel = (C.nZ / 2) * C.nX * C.nY + (C.nY / 2) * C.nX + C.nX / 2;
    traceRot = (int)cfg.integer("trace_rotation", rotBegin);
    traceHist = (int)cfg.integer("trace_histories", 200);
    traceCapacity = traceVoxel >= 0 ? (int)cfg.integer("trace_max_vertices", 200000) : 0;
    if (traceVoxel >= numVox) { err = "trace_voxel out of range"; return false; }

    std::string od = cfg.str("output_dir", ".");
    if (!od.empty() && od.back() != '/' && od.back() != '\\') od += "/";
    char buf[512];
    snprintf(buf, sizeof(buf), "MC_SysMat_%.1fkeV_shift_%f_%f_%f", C.E0, C.shX, C.shY, C.shZ);
    outPrefix = od + cfg.str("output_prefix", buf);
    if (rotBegin != 0 || rotEnd != numRot)
        outPrefix += "_rot" + std::to_string(rotBegin) + "-" + std::to_string(rotEnd);

    bindPointers();
    return true;
}

void Simulation::bindPointers() {
    Geometry& g = hg.g;
    g.hgCellStart = hg.hgCellStart.data();
    g.hgItems = hg.hgItems.data();
    g.hole0 = hg.hole0.data();
    g.hole1 = hg.hole1.data();
    g.holeMat = hg.holeMat.data();
    g.xtal0 = hg.xtal0.data();
    g.xtal1 = hg.xtal1.data();
    g.xtalMat = hg.xtalMat.data();
    g.xtalEres = hg.xtalEres.data();
    g.cgCellStart = hg.cgCellStart.data();
    g.cgItems = hg.cgItems.data();

    M.nMat = (int)hm.names.size();
    M.nE = hm.nE;
    M.logEmin = (float)std::log(hm.eMin);
    M.invDlogE = (float)((hm.nE - 1) / std::log(hm.eMax / hm.eMin));
    M.muPE = hm.muPE.data();
    M.muInc = hm.muInc.data();
    M.muCoh = hm.muCoh.data();
    M.muTot = hm.muTot.data();
    M.regMaj = hm.regMaj.data();
    M.nFluo = hm.nFluo.data();
    M.fluoE = hm.fluoE.data();
    M.fluoP = hm.fluoP.data();
    M.nFF = hm.nFF;
    M.ffX2min = (float)hm.x2Min;
    M.ffLogX2min = (float)std::log(hm.x2Min);
    M.ffInvDlog = (float)((hm.nFF - 1) / std::log(hm.x2Max / hm.x2Min));
    M.ffX2 = hm.ffX2.data();
    M.ffCum = hm.ffCum.data();
    M.hasFF = hm.hasFF.data();
}

}  // namespace mc
