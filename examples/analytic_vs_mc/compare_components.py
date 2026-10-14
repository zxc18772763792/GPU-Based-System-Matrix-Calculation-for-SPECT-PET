#!/usr/bin/env python3
"""
Component-by-component comparison of the analytic system matrix (PEGen + ScatterGen) with
MCGen_MultiGPU, for the example produced by run_example.sh.

Expected files in <run_dir>:
    Params_*.dat
    PE_SysMat_shift_*_v3.sysmat                     analytic photoelectric matrix (PEGen / pe_cpu)
    Scatter_Crystal_SysMat_shift_*.sysmat           analytic crystal scatter   (ScatterGen -components)
    Scatter_Collimator_SysMat_shift_*.sysmat        analytic collimator scatter
    mc_pe_win0.sysmat                               MC, physics=primary_pe, ideal window
    <run>_win<k>_{total,primary,colscatter,xtalscatter,relerr}.sysmat   MC full-physics runs
optional --wide <dir>: analytic scatter computed with the wide window (MC window 1).

Metrics for every pair analytic A / Monte Carlo M (same window, same event class):
    sum ratio      sum(A) / sum(M)                (+- MC statistical error of sum(M))
    voxel ratio    median and 5-95 % range of the per-voxel sensitivity ratio
    shape TVD      0.5 * sum |A/sum(A) - M/sum(M)|: 0 = same distribution, 1 = disjoint
                   (the "noise" column bounds what MC statistics alone would give)

    python3 compare_components.py <run_dir> [--wide <dir>] [--md report.md] [--svg flood.svg]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

CLASSES = ("primary", "colscatter", "xtalscatter")


def params(d):
    det = np.fromfile(os.path.join(d, "Params_Detector.dat"), np.float32)
    img = np.fromfile(os.path.join(d, "Params_Image.dat"), np.float32)
    nP = int(det[0] + 1e-4)
    nI = int(img[0] + 1e-3) * int(img[1] + 1e-3) * int(img[2] + 1e-3)
    nR = max(1, int(img[6] + 1e-3))
    xtal = det[1:1 + 12 * nP].reshape(nP, 12)
    return nP, nI, nR, xtal


def load(path, shape):
    a = np.fromfile(path, np.float32)
    if a.size != np.prod(shape):
        sys.exit(f"{path}: {a.size} values, expected {np.prod(shape)}")
    return a.reshape(shape).astype(np.float64)


def find(d, pattern):
    hits = sorted(glob.glob(os.path.join(d, pattern)))
    return hits[0] if hits else None


def compare(A, M, Merr=None):
    """A, M: (rot, det, vox) arrays. Merr: absolute 1-sigma error per element (upper bound) or None."""
    sA, sM = A.sum(), M.sum()
    r = {"analytic": sA, "mc": sM, "ratio": sA / sM if sM > 0 else float("nan")}
    if Merr is not None and sM > 0:
        # elements are nearly independent (different histories dominate different crystals)
        r["ratio_err"] = r["ratio"] * math.sqrt((Merr ** 2).sum()) / sM
        r["tvd_noise"] = 0.5 * math.sqrt(2 / math.pi) * Merr.sum() / sM
    vA, vM = A.sum(axis=1).ravel(), M.sum(axis=1).ravel()
    ok = vM > 0.05 * vM.max() if vM.max() > 0 else np.zeros_like(vM, bool)
    if ok.any():
        q = np.percentile(vA[ok] / vM[ok], [5, 50, 95])
        r["voxel_ratio"] = q.tolist()
    if sA > 0 and sM > 0:
        r["tvd"] = 0.5 * np.abs(A / sA - M / sM).sum()
    return r


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}e}"


def row(name, r):
    ratio = f"{r['ratio']:.3f}" + (f" ± {r['ratio_err']:.3f}" if "ratio_err" in r else "")
    vr = r.get("voxel_ratio")
    vr = f"{vr[1]:.3f} [{vr[0]:.3f}, {vr[2]:.3f}]" if vr else "-"
    tvd = f"{r['tvd']:.3f}" if "tvd" in r else "-"
    noise = f"≤ {r['tvd_noise']:.3f}" if "tvd_noise" in r else "-"
    return f"| {name} | {fmt(r['analytic'])} | {fmt(r['mc'])} | {ratio} | {vr} | {tvd} | {noise} |"


HEAD = ("| analytic vs MC | Σ analytic | Σ MC | ratio A/MC | per-voxel ratio: median [5 %, 95 %] | shape TVD | MC noise |\n"
        "|---|---|---|---|---|---|---|")


def svg_flood(path, xtal, panels, ncol):
    """panels: list of (title, flood[nDet] or None). Crystal rectangles at their (x, z) positions,
    each panel scaled from its own minimum to its own maximum (the sums are printed in the titles)."""
    x, z, w, h = xtal[:, 0], xtal[:, 2], xtal[:, 3], xtal[:, 5]
    x0, x1 = (x - w / 2).min(), (x + w / 2).max()
    z0, z1 = (z - h / 2).min(), (z + h / 2).max()
    S = 180.0 / max(x1 - x0, z1 - z0)
    pw, ph = (x1 - x0) * S, (z1 - z0) * S
    gap, top = 14, 34
    nrow = (len(panels) + ncol - 1) // ncol
    W, H = ncol * (pw + gap) + gap, nrow * (ph + top + gap) + gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'font-family="sans-serif" font-size="11"><rect width="100%" height="100%" fill="#fff"/>']
    for k, (title, f) in enumerate(panels):
        ox = gap + (k % ncol) * (pw + gap)
        oy = gap + (k // ncol) * (ph + top + gap)
        lines = title.split("\n")
        for j, t in enumerate(lines):
            out.append(f'<text x="{ox:.1f}" y="{oy + 12 + 12 * j:.1f}">{t}</text>')
        if f is None:
            continue
        lo, hi = float(f.min()), float(f.max())
        span = hi - lo if hi > lo else 1.0
        for i in range(len(f)):
            v = max(0.0, min(1.0, (f[i] - lo) / span)) ** 0.5  # min..max of each panel, sqrt scale
            c = int(255 * (1 - v))
            rx = ox + (x[i] - w[i] / 2 - x0) * S
            ry = oy + top + (z1 - (z[i] + h[i] / 2)) * S      # +z up
            out.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{w[i] * S:.1f}" height="{h[i] * S:.1f}" '
                       f'fill="rgb({c},{c},{255 - int(155 * v)})"/>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--wide", help="directory with the analytic scatter for MC window 1")
    ap.add_argument("--runs", default="mc_model,mc_full", help="MC output prefixes (full physics runs)")
    ap.add_argument("--md", help="write the Markdown report here")
    ap.add_argument("--json", help="write all numbers here")
    ap.add_argument("--svg", help="flood images (sum over voxels) per component")
    a = ap.parse_args()
    d = a.run_dir
    nP, nI, nR, xtal = params(d)
    shp = (nR, nP, nI)

    pe = find(d, "PE_SysMat_shift_*_v3.sysmat") or find(d, "PE_SysMat_shift_*.sysmat")
    cr = find(d, "Scatter_Crystal_SysMat_shift_*.sysmat")
    co = find(d, "Scatter_Collimator_SysMat_shift_*.sysmat")
    if not (pe and cr and co):
        sys.exit("analytic matrices missing (PE_SysMat_*, Scatter_Crystal_SysMat_*, Scatter_Collimator_SysMat_*)")
    AN = {"pe": load(pe, shp), "crystal": load(cr, shp), "collimator": load(co, shp)}
    wide = None
    if a.wide:
        wc = find(a.wide, "Scatter_Crystal_SysMat_shift_*.sysmat")
        wl = find(a.wide, "Scatter_Collimator_SysMat_shift_*.sysmat")
        if wc and wl:
            wide = {"crystal": load(wc, shp), "collimator": load(wl, shp)}

    res, md = {}, []
    md.append(f"Geometry: {nP} crystals, {nI} voxels, {nR} rotation(s).  "
              "Sensitivities are summed over all matrix elements (counts per emitted photon x voxels).\n")

    mcpe = os.path.join(d, "mc_pe_win0.sysmat")
    if os.path.exists(mcpe):
        M = load(mcpe, shp)
        rel = os.path.join(d, "mc_pe_win0_relerr.sysmat")
        err = load(rel, shp) * M if os.path.exists(rel) else None
        res["pe_vs_mc_primary_pe"] = compare(AN["pe"], M, err)
        md += ["### Photoelectric model (MC physics = primary_pe, ideal window)\n", HEAD,
               row("PE vs MC primary_pe", res["pe_vs_mc_primary_pe"]), ""]

    for run in [r for r in a.runs.split(",") if r]:
        if not os.path.exists(os.path.join(d, f"{run}_win0_total.sysmat")):
            continue
        summ = os.path.join(d, f"{run}_summary.json")
        desc, wins = "", []
        if os.path.exists(summ):
            with open(summ) as fh:
                js = json.load(fh)
            wins = js.get("windows", [])
            desc = (f"physics {js.get('physics')}, Rayleigh {js.get('rayleigh')}, fluorescence {js.get('fluorescence')}, "
                    f"{js.get('total_histories', 0):.3g} histories; " + "; ".join(js.get("materials", [])))
        wide_label = f"wide window {wins[1]['lo']:g}-{wins[1]['hi']:g} keV" if len(wins) > 1 else "window 1"
        for k, (an, label) in enumerate([(AN, "photopeak window (Param_Physics)"), (wide, wide_label)]):
            f = lambda c: os.path.join(d, f"{run}_win{k}_{c}.sysmat")
            if an is None or not os.path.exists(f("total")):
                continue
            M = {c: load(f(c), shp) for c in CLASSES + ("total",)}
            err = load(f("relerr"), shp) * M["total"] if os.path.exists(f("relerr")) else None
            pe_term = AN["pe"]
            rr = {
                "primary": compare(pe_term, M["primary"], err),
                "crystal": compare(an["crystal"], M["xtalscatter"], err),
                "collimator": compare(an["collimator"], M["colscatter"], err),
                "total": compare(pe_term + an["crystal"] + an["collimator"], M["total"], err),
            }
            sa = (an["crystal"].sum() + an["collimator"].sum()) / pe_term.sum()
            sm = (M["xtalscatter"].sum() + M["colscatter"].sum()) / M["primary"].sum()
            rr["scatter_to_primary"] = {"analytic": sa, "mc": sm}
            res[f"{run}_win{k}"] = rr
            md += [f"### {run}, {label}\n", f"{desc}\n", HEAD,
                   row("PE vs MC primary", rr["primary"]),
                   row("crystal scatter vs MC xtalscatter", rr["crystal"]),
                   row("collimator scatter vs MC colscatter", rr["collimator"]),
                   row("PE + scatter vs MC total", rr["total"]),
                   "", f"scatter / primary: analytic {sa:.4f}, MC {sm:.4f}\n"]
            if a.svg and k == 0 and run == a.runs.split(",")[-1]:
                fl = lambda X: X.sum(axis=(0, 2))
                panels = [("analytic PE", fl(AN["pe"])), ("MC primary", fl(M["primary"])),
                          ("analytic crystal scatter", fl(an["crystal"])), ("MC crystal scatter", fl(M["xtalscatter"])),
                          ("analytic collimator scatter", fl(an["collimator"])), ("MC collimator scatter", fl(M["colscatter"]))]
                panels = [(f"{t}\nsum {X.sum():.3e}", X) for t, X in panels]
                svg_flood(a.svg, xtal, panels, 2)

    text = "\n".join(md)
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
