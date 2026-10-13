#!/usr/bin/env python3
"""
Generate the photon cross-section database used by MCGen_MultiGPU.

Data source: xraylib (EPDL97 / Scofield photoionisation, Hubbell incoherent
and coherent cross sections, Hubbell form factors, Krause fluorescence yields).
    pip install xraylib

For every material the file stores, on a log-spaced energy grid
(default 1 keV .. 1100 keV, 1024 points):
  * mass attenuation coefficients (cm^2/g) for photoelectric absorption,
    incoherent (Compton) and coherent (Rayleigh) scattering,
  * K-shell fluorescence lines (Z >= 30) and, per grid energy, the probability
    that a photoelectric absorption emits each line,
  * the cumulative integral A(x^2) = int_0^{x^2} F^2(x') d(x'^2) of the squared
    (independent-atom) form factor, used to sample Rayleigh scattering angles.

Usage:
    python3 gen_material_db.py                      # default material set
    python3 gen_material_db.py --add "LuAG:Lu3Al5O12:6.73" --out my.db
    python3 gen_material_db.py --list

Material definitions are FORMULA:DENSITY (compound formula, fractional
stoichiometry allowed) or MIX(Sym:massfrac,Sym:massfrac,...):DENSITY.
"""

import argparse
import math
import sys

try:
    import xraylib as xrl
except ImportError:  # pragma: no cover
    sys.exit("xraylib is required:  pip install xraylib")

# name -> (definition, density g/cm^3, comment)
DEFAULT_MATERIALS = [
    ("Air",        "MIX(C:0.000124,N:0.755268,O:0.231781,Ar:0.012827)", 0.001205, "dry air, sea level"),
    ("Water",      "H2O",            1.000, ""),
    ("PMMA",       "C5H8O2",         1.190, "acrylic"),
    ("Al",         "Al",             2.699, "housing / entrance window"),
    ("Fe",         "Fe",             7.874, ""),
    ("Cu",         "Cu",             8.960, ""),
    # collimator / shielding
    ("W",          "W",             19.300, "pure tungsten"),
    ("W90Ni7Fe3",  "MIX(W:0.90,Ni:0.07,Fe:0.03)", 17.000, "tungsten heavy alloy"),
    ("W95Ni3.5Fe1.5", "MIX(W:0.95,Ni:0.035,Fe:0.015)", 18.000, "tungsten heavy alloy"),
    ("Pb",         "Pb",            11.350, "lead"),
    ("Ta",         "Ta",            16.650, ""),
    ("Au",         "Au",            19.320, "pinhole inserts"),
    ("Pt",         "Pt",            21.450, "pinhole inserts"),
    # scintillators / semiconductors
    ("NaI",        "NaI",            3.670, "NaI(Tl)"),
    ("CsI",        "CsI",            4.510, "CsI(Tl/Na)"),
    ("LYSO",       "Lu1.8Y0.2SiO5",  7.100, "LYSO(Ce)"),
    ("LSO",        "Lu2SiO5",        7.400, "LSO(Ce)"),
    ("BGO",        "Bi4Ge3O12",      7.130, ""),
    ("GAGG",       "Gd3Al2Ga3O12",   6.630, "GAGG(Ce)"),
    ("LaBr3",      "LaBr3",          5.080, "LaBr3(Ce)"),
    ("CeBr3",      "CeBr3",          5.100, ""),
    ("CZT",        "Cd0.9Zn0.1Te",   5.780, ""),
    ("CdTe",       "CdTe",           5.850, ""),
    ("Ge",         "Ge",             5.323, "HPGe"),
    ("Si",         "Si",             2.330, ""),
]

K_GROUPS = [  # (label, xraylib lines merged into one emission line)
    ("KA1", ("KA1_LINE",)),
    ("KA2", ("KA2_LINE",)),
    ("KB13", ("KB1_LINE", "KB3_LINE", "KB5_LINE")),
    ("KB2", ("KB2_LINE", "KB4_LINE")),
]
FLUO_MIN_Z = 30
HC_KEV_ANGSTROM = 12.398419843320026  # h*c in keV*Angstrom


def parse_definition(defn):
    """Return list of (Z, mass_fraction)."""
    if defn.startswith("MIX(") and defn.endswith(")"):
        items = []
        for tok in defn[4:-1].split(","):
            sym, frac = tok.split(":")
            items.append((xrl.SymbolToAtomicNumber(sym.strip()), float(frac)))
        s = sum(f for _, f in items)
        return [(z, f / s) for z, f in items]
    comp = xrl.CompoundParser(defn)
    return list(zip(comp["Elements"], comp["massFractions"]))


def safe(fn, *args):
    try:
        return fn(*args)
    except ValueError:
        return 0.0


def k_lines(z):
    """Merged K lines of element z: list of (label, energy keV, relative rate)."""
    out = []
    total = 0.0
    for label, names in K_GROUPS:
        e_sum = r_sum = 0.0
        for n in names:
            line = getattr(xrl, n)
            r = safe(xrl.RadRate, z, line)
            e = safe(xrl.LineEnergy, z, line)
            if r > 0 and e > 0:
                e_sum += e * r
                r_sum += r
        if r_sum > 0:
            out.append((label, e_sum / r_sum, r_sum))
            total += r_sum
    return [(l, e, r / total) for l, e, r in out]  # renormalised to sum 1


def build_material(name, defn, density, egrid, x2grid, fluo):
    comp = parse_definition(defn)
    pe = [sum(w * safe(xrl.CS_Photo, z, e) for z, w in comp) for e in egrid]
    inc = [sum(w * safe(xrl.CS_Compt, z, e) for z, w in comp) for e in egrid]
    coh = [sum(w * safe(xrl.CS_Rayl, z, e) for z, w in comp) for e in egrid]

    lines = []          # (label, Z, E, P(E) list)
    if fluo:
        for z, w in comp:
            if z < FLUO_MIN_Z:
                continue
            omega = safe(xrl.FluorYield, z, xrl.K_SHELL)
            for label, eline, rate in k_lines(z):
                probs = []
                for i, e in enumerate(egrid):
                    tau_k = safe(xrl.CS_Photo_Partial, z, xrl.K_SHELL, e)
                    p = w * tau_k * omega * rate / pe[i] if pe[i] > 0 else 0.0
                    probs.append(p)
                if max(probs) > 1e-6:
                    lines.append((label, z, eline, probs))

    # Rayleigh: cumulative of F_mix^2 in x^2 (x = sin(theta/2)/lambda, 1/Angstrom)
    # F_mix^2 = sum_i (w_i/A_i) F_i(x)^2 (atoms per gram, arbitrary normalisation)
    na = [(z, w / xrl.AtomicWeight(z)) for z, w in comp]
    fine_n = 6000
    lo, hi = math.log(1e-6), math.log(x2grid[-1] * 1.0001)
    fine = [0.0] + [math.exp(lo + (hi - lo) * k / (fine_n - 1)) for k in range(fine_n)]

    def f2(x2):
        x = math.sqrt(x2)
        return sum(n * xrl.FF_Rayl(z, x) ** 2 for z, n in na)

    vals = [f2(v) for v in fine]
    cum = [0.0]
    for k in range(1, len(fine)):
        cum.append(cum[-1] + 0.5 * (vals[k] + vals[k - 1]) * (fine[k] - fine[k - 1]))
    # resample at x2grid (linear interpolation on fine grid)
    ff = []
    j = 1
    for v in x2grid:
        while j < len(fine) - 1 and fine[j] < v:
            j += 1
        t = (v - fine[j - 1]) / (fine[j] - fine[j - 1])
        ff.append(cum[j - 1] + t * (cum[j] - cum[j - 1]))
    norm = ff[-1]
    ff = [v / norm for v in ff]
    return comp, pe, inc, coh, lines, ff


def fmt(values):
    return " ".join("0" if v == 0 else f"{v:.5g}" for v in values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="materials.db")
    ap.add_argument("--emin", type=float, default=1.0)
    ap.add_argument("--emax", type=float, default=1100.0)
    ap.add_argument("--ne", type=int, default=1024)
    ap.add_argument("--nff", type=int, default=400)
    ap.add_argument("--only", default="", help="comma separated subset of default material names")
    ap.add_argument("--add", action="append", default=[], help="NAME:DEFINITION:DENSITY (repeatable)")
    ap.add_argument("--no-fluo", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    mats = list(DEFAULT_MATERIALS)
    if args.list:
        for n, d, rho, c in mats:
            print(f"{n:16s} {d:40s} {rho:8.4f}  {c}")
        return
    if args.only:
        keep = set(s.strip() for s in args.only.split(","))
        mats = [m for m in mats if m[0] in keep]
    for spec in args.add:
        name, rest = spec.split(":", 1)       # MIX(...) definitions contain ':'
        defn, rho = rest.rsplit(":", 1)
        mats.append((name, defn, float(rho), "user"))

    ne = args.ne
    lr = math.log(args.emax / args.emin)
    egrid = [args.emin * math.exp(lr * i / (ne - 1)) for i in range(ne)]
    x2min, x2max = 1e-4, 1e4
    x2grid = [x2min * (x2max / x2min) ** (k / (args.nff - 1)) for k in range(args.nff)]

    with open(args.out, "w") as f:
        f.write("# Photon cross-section database for MCGen_MultiGPU\n")
        f.write(f"# generated with xraylib {getattr(xrl, '__version__', '?')} by tools/gen_material_db.py\n")
        f.write("# MU_* : mass attenuation coefficients in cm^2/g on EGRID (log-spaced, keV)\n")
        f.write("# FLUO : K line (label Z energy_keV) + probability per photoelectric absorption on EGRID\n")
        f.write("# FF_CUM : normalised cumulative int F^2 d(x^2) on FFGRID (x = sin(theta/2)/lambda [1/A], log-spaced in x^2)\n")
        f.write("MCDB 1\n")
        f.write(f"EGRID {ne} {args.emin:.6g} {args.emax:.6g}\n")
        f.write(f"FFGRID {args.nff} {x2min:.6g} {x2max:.6g}\n")
        for name, defn, rho, comment in mats:
            comp, pe, inc, coh, lines, ff = build_material(name, defn, rho, egrid, x2grid, not args.no_fluo)
            i140 = min(range(ne), key=lambda i: abs(egrid[i] - 140.5))
            i511 = min(range(ne), key=lambda i: abs(egrid[i] - 511.0))
            mu = lambda i: (pe[i] + inc[i] + coh[i]) * rho / 10.0
            print(f"{name:16s} rho={rho:7.4f}  mu(140.5)={mu(i140):8.4f}/mm  mu(511)={mu(i511):8.4f}/mm  K-lines={len(lines)}")
            f.write(f"MATERIAL {name}\n")
            if comment:
                f.write(f"# {comment}: {defn}\n")
            f.write(f"DENSITY {rho:.6g}\n")
            f.write("COMPOSITION " + " ".join(f"{z}:{w:.6g}" for z, w in comp) + "\n")
            f.write("MU_PE " + fmt(pe) + "\n")
            f.write("MU_INC " + fmt(inc) + "\n")
            f.write("MU_COH " + fmt(coh) + "\n")
            f.write(f"NFLUO {len(lines)}\n")
            for label, z, eline, probs in lines:
                f.write(f"FLUO {label} {z} {eline:.5f} " + fmt(probs) + "\n")
            f.write("FF_CUM " + fmt(ff) + "\n")
            f.write("END\n")
    print(f"wrote {args.out} ({len(mats)} materials)")


if __name__ == "__main__":
    main()
