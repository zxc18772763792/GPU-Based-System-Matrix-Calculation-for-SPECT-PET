// tests/test_physics.cpp - self tests of the shared MC core (runs on the CPU, no GPU needed)
//   g++ -O2 -std=c++17 -I.. test_physics.cpp ../mc_host.cpp -o test_physics && ./test_physics ../data/materials.db
//
//  1. Philox4x32-10 known-answer vectors (Random123)
//  2. Klein-Nishina sampling vs the analytic differential cross section (140.5, 511 keV)
//  3. Rayleigh sampling vs (1+cos^2) F^2(x) built from the tabulated form factors (W, NaI)
//  4. Photoelectric K-fluorescence probabilities (Pb, W at 140.5 keV)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../mc_host.h"

using namespace mc;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++failures;
}

static double knDiff(double E, double c) {  // dsigma/dOmega up to a constant
    double k = E / kElectronMass;
    double r = 1.0 / (1.0 + k * (1.0 - c));
    return r * r * (r + 1.0 / r - (1.0 - c * c));
}

// chi^2 per degree of freedom of a cos(theta) histogram against a density
template <class F>
static double chi2(const std::vector<double>& h, long long n, F pdf, int bins) {
    std::vector<double> e(bins);
    double tot = 0;
    for (int b = 0; b < bins; ++b) {
        double s = 0;
        for (int k = 0; k < 20; ++k) s += pdf(-1.0 + (b + (k + 0.5) / 20.0) * 2.0 / bins);
        e[b] = s;
        tot += s;
    }
    double x2 = 0;
    int dof = 0;
    for (int b = 0; b < bins; ++b) {
        double ex = e[b] / tot * n;
        if (ex < 20) continue;
        x2 += (h[b] - ex) * (h[b] - ex) / ex;
        ++dof;
    }
    return x2 / std::max(1, dof - 1);
}

int main(int argc, char** argv) {
    printf("Philox4x32-10 known answers\n");
    {
        uint32_t o[4];
        Philox::block(0, 0, 0, 0, 0, 0, o);
        check(o[0] == 0x6627e8d5u && o[1] == 0xe169c58du && o[2] == 0xbc57ac4cu && o[3] == 0x9b00dbd8u, "zero counter / key");
        Philox::block(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, o);
        check(o[0] == 0x408f276du && o[1] == 0x41c83b0eu && o[2] == 0xa20bc7c6u && o[3] == 0x6d5451fdu, "all-ones counter / key");
        Philox::block(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xa4093822u, 0x299f31d0u, o);
        check(o[0] == 0xd16cfe09u && o[1] == 0x94fdccebu && o[2] == 0x5001e420u && o[3] == 0x24126ea1u, "pi digits");
    }

    const int bins = 100;
    const long long n = 4000000;
    printf("Klein-Nishina sampling (%lld samples, %d bins)\n", n, bins);
    for (double E : {140.511, 511.0}) {
        Philox rng;
        rng.init(1234, 1, 2, 3);
        std::vector<double> h(bins, 0);
        double eSum = 0;
        for (long long i = 0; i < n; ++i) {
            float c;
            float e1 = sampleKleinNishina((float)E, rng, c);
            int b = std::min(bins - 1, (int)((c + 1.0f) * 0.5f * bins));
            h[b]++;
            double expect = E / (1.0 + E / kElectronMass * (1.0 - c));
            eSum += std::fabs(e1 - expect) / expect;
        }
        double x2 = chi2(h, n, [&](double c) { return knDiff(E, c); }, bins);
        char msg[160];
        snprintf(msg, sizeof(msg), "E=%.1f keV: chi2/dof = %.3f, Compton formula error %.1e", E, x2, eSum / n);
        check(x2 < 1.5 && eSum / n < 1e-5, msg);
    }

    if (argc < 2) {
        printf("(pass the material database path to run the Rayleigh / fluorescence tests)\n");
        return failures ? 1 : 0;
    }
    // minimal simulation with only materials, built through the normal setup path
    Simulation S;
    std::string err;
    if (!S.db.load(argv[1], err)) { printf("%s\n", err.c_str()); return 1; }
    S.dbLoaded = true;
    S.hm.nE = S.db.nE; S.hm.eMin = S.db.eMin; S.hm.eMax = S.db.eMax;
    S.hm.nFF = S.db.nFF; S.hm.x2Min = S.db.x2Min; S.hm.x2Max = S.db.x2Max;
    S.hm.ffX2.resize(S.hm.nFF);
    for (int k = 0; k < S.hm.nFF; ++k)
        S.hm.ffX2[k] = (float)(S.hm.x2Min * std::pow(S.hm.x2Max / S.hm.x2Min, (double)k / (S.hm.nFF - 1)));
    const char* names[] = {"W", "NaI", "Pb"};
    for (const char* nm : names) {
        const DbMaterial* m = S.db.find(nm);
        if (!m) { printf("material %s missing\n", nm); return 1; }
        int id = (int)S.hm.names.size();
        S.hm.names.push_back(nm);
        for (int i = 0; i < S.hm.nE; ++i) {
            float s = (float)(m->density / 10.0);
            S.hm.muPE.push_back(m->pe[i] * s);
            S.hm.muInc.push_back(m->inc[i] * s);
            S.hm.muCoh.push_back(m->coh[i] * s);
            S.hm.muTot.push_back((m->pe[i] + m->inc[i] + m->coh[i]) * s);
        }
        S.hm.nFluo.push_back((int)std::min<size_t>(m->fluoE.size(), kMaxFluo));
        S.hm.fluoE.resize((id + 1) * kMaxFluo, 0.0f);
        S.hm.fluoP.resize((size_t)(id + 1) * S.hm.nE * kMaxFluo, 0.0f);
        for (int l = 0; l < S.hm.nFluo[id]; ++l) {
            S.hm.fluoE[id * kMaxFluo + l] = m->fluoE[l];
            for (int i = 0; i < S.hm.nE; ++i) S.hm.fluoP[((size_t)id * S.hm.nE + i) * kMaxFluo + l] = m->fluoP[l][i];
        }
        S.hm.ffCum.insert(S.hm.ffCum.end(), m->ffCum.begin(), m->ffCum.end());
        S.hm.hasFF.push_back(1);
    }
    S.hm.regMaj.assign(S.hm.nE, 0.0f);
    S.bindPointers();
    const Materials& M = S.M;

    printf("Rayleigh sampling vs tabulated form factors (%lld samples)\n", n);
    for (int m = 0; m < 2; ++m)
        for (double E : {140.511, 511.0}) {
            Philox rng;
            rng.init(99, m, (uint32_t)E, 0);
            std::vector<double> h(bins, 0);
            for (long long i = 0; i < n; ++i) {
                float c = sampleRayleigh(M, m, (float)E, rng);
                h[std::min(bins - 1, (int)((c + 1.0f) * 0.5f * bins))]++;
            }
            double xmax2 = std::pow(E / kHcKeVA, 2);
            // dsigma/dcos ~ (1+c^2) dA/d(x^2), with A the tabulated cumulative
            auto pdf = [&](double c) {
                double x2 = xmax2 * (1.0 - c) / 2.0, d = 1e-3 * x2 + 1e-9;
                double dA = (ffCumAt(M, m, (float)(x2 + d)) - ffCumAt(M, m, (float)std::max(0.0, x2 - d))) / (x2 + d - std::max(0.0, x2 - d));
                return (1.0 + c * c) * dA;
            };
            // the forward peak is extremely narrow: compare on cos < 0.9 only
            std::vector<double> hh(h.begin(), h.begin() + 95);
            long long nn = 0;
            for (double v : hh) nn += (long long)v;
            std::vector<double> e(95);
            double tot = 0;
            for (int b = 0; b < 95; ++b) {
                double s = 0;
                for (int k = 0; k < 20; ++k) s += pdf(-1.0 + (b + (k + 0.5) / 20.0) * 2.0 / bins);
                e[b] = s;
                tot += s;
            }
            double chi = 0;
            int dof = 0;
            for (int b = 0; b < 95; ++b) {
                double ex = e[b] / tot * nn;
                if (ex < 20) continue;
                chi += (hh[b] - ex) * (hh[b] - ex) / ex;
                ++dof;
            }
            double meanC = 0;
            for (int b = 0; b < bins; ++b) meanC += h[b] * (-1.0 + (b + 0.5) * 2.0 / bins);
            char msg[200];
            snprintf(msg, sizeof(msg), "%s E=%.1f keV: chi2/dof(cos<0.9) = %.3f, <cos theta> = %.4f", S.hm.names[m].c_str(), E,
                     dof > 1 ? chi / (dof - 1) : 0.0, meanC / n);
            check(dof > 1 && chi / (dof - 1) < 1.5, msg);
        }

    printf("K fluorescence per photo-absorption at 140.5 keV\n");
    for (int m : {0, 2}) {
        Photon ph;
        setEnergy(M, ph, 140.511f);
        double p = 0;
        int nf = M.nFluo[m];
        for (int l = 0; l < nf; ++l) {
            const float* pa = M.fluoP + ((size_t)m * M.nE + ph.ei) * kMaxFluo;
            p += pa[l] + ph.ef * (pa[l + kMaxFluo] - pa[l]);
        }
        char msg[160];
        // tau_K/tau ~ 0.79-0.80 and omega_K ~ 0.95-0.96 for W and Pb
        snprintf(msg, sizeof(msg), "%s: P(K x-ray) = %.3f (expected ~0.75-0.80)", S.hm.names[m].c_str(), p);
        check(p > 0.70 && p < 0.80, msg);
    }
    printf(failures ? "%d test(s) FAILED\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
