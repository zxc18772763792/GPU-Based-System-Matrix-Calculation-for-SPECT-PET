#!/usr/bin/env python3
"""
Independent cross-check of the full photon transport (multiple Compton scattering + photoelectric
absorption) against a separate, vectorised numpy Monte Carlo.

Setup: one crystal (box) in front of a point source, no collimator, Rayleigh and fluorescence off.
Both codes use the same cross-section table (data/materials.db) but nothing else is shared: the
reference uses its own geometry, cone source sampling and Kahn's method for Klein-Nishina
(MCGen uses the Butcher-Messel method). Deposited-energy windows are compared:
photopeak, Compton-only region, intermediate region and "any deposit".

    python3 tests/reference_mc.py --energy 140.5 --material NaI
    python3 tests/reference_mc.py --energy 511 --material LYSO --thickness 15
"""
import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ME = 510.99895


def load_material(db, name):
    grid, mat, cur = None, {}, None
    with open(db) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "EGRID":
                n, e0, e1 = int(t[1]), float(t[2]), float(t[3])
                grid = e0 * (e1 / e0) ** (np.arange(n) / (n - 1))
            elif t[0] == "MATERIAL":
                cur = t[1]
            elif cur == name and t[0] in ("DENSITY", "MU_PE", "MU_INC"):
                mat[t[0]] = np.array([float(v) for v in t[1:]])
    rho = mat["DENSITY"][0] / 10.0
    return grid, mat["MU_PE"] * rho, mat["MU_INC"] * rho


def kahn(E, rng):
    """Klein-Nishina by Kahn's rejection method. Returns (E', cos theta)."""
    k = E / ME
    out_x = np.empty_like(E)
    todo = np.arange(E.size)
    while todo.size:
        kk = k[todo]
        r1, r2, r3 = rng.random(todo.size), rng.random(todo.size), rng.random(todo.size)
        b1 = r1 <= (1 + 2 * kk) / (9 + 2 * kk)
        x = np.where(b1, 1 + 2 * kk * r2, (1 + 2 * kk) / (1 + 2 * kk * r2))
        c = 1 - (x - 1) / kk
        acc = np.where(b1, r3 <= 4 * (1 / x - 1 / x ** 2), r3 <= 0.5 * (c * c + 1 / x))
        out_x[todo[acc]] = x[acc]
        todo = todo[~acc]
    cos = 1 - (out_x - 1) / k
    return E / out_x, np.clip(cos, -1, 1)


def rotate(d, cos, phi):
    sin = np.sqrt(np.maximum(0, 1 - cos ** 2))
    # orthonormal frame around d
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    u = np.cross(d, a); u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(d, u)
    n = d * cos[:, None] + (u * np.cos(phi)[:, None] + v * np.sin(phi)[:, None]) * sin[:, None]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def box_exit(p, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - p) / d
        t2 = (hi - p) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return tmin, tmax


def reference(args, grid, mpe, minc, n, seed):
    rng = np.random.default_rng(seed)
    hx, hy, hz = args.width / 2, args.thickness / 2, args.width / 2
    Y = args.distance
    lo, hi = np.array([-hx, Y - hy, -hz]), np.array([hx, Y + hy, hz])
    alpha = math.atan(math.hypot(hx, hz) / (Y - hy)) * 1.02
    wcone = (1 - math.cos(alpha)) / 2
    cos0 = 1 - rng.random(n) * (1 - math.cos(alpha))
    ph0 = rng.random(n) * 2 * math.pi
    s0 = np.sqrt(1 - cos0 ** 2)
    d = np.stack([s0 * np.cos(ph0), cos0, s0 * np.sin(ph0)], axis=1)
    p = np.zeros((n, 3))
    tmin, tmax = box_exit(p, d, lo, hi)
    hit = tmax > np.maximum(tmin, 0)
    idx = np.nonzero(hit)[0]
    p = p[idx] + d[idx] * tmin[idx, None]
    d = d[idx]
    E = np.full(idx.size, args.energy)
    dep = np.zeros(idx.size)
    alive = np.arange(idx.size)
    lg = np.log(grid)
    while alive.size:
        e = E[alive]
        le = np.log(e)
        pe = np.interp(le, lg, mpe); ic = np.interp(le, lg, minc); mu = pe + ic
        _, texit = box_exit(p[alive], d[alive], lo, hi)
        s = -np.log(rng.random(alive.size)) / mu
        esc = s >= texit
        inside = alive[~esc]
        p[inside] += d[inside] * s[~esc, None]
        pe_i, mu_i = pe[~esc], mu[~esc]
        photo = rng.random(inside.size) * mu_i < pe_i
        dep[inside[photo]] += E[inside[photo]]
        comp = inside[~photo]
        if comp.size:
            e1, c = kahn(E[comp], rng)
            dep[comp] += E[comp] - e1
            E[comp] = e1
            d[comp] = rotate(d[comp], c, rng.random(comp.size) * 2 * math.pi)
            cut = comp[E[comp] < args.cutoff]
            dep[cut] += E[cut]
            comp = comp[E[comp] >= args.cutoff]
        alive = comp
    return dep, wcone, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energy", type=float, default=140.5)
    ap.add_argument("--material", default="NaI")
    ap.add_argument("--width", type=float, default=20.0)
    ap.add_argument("--thickness", type=float, default=10.0)
    ap.add_argument("--distance", type=float, default=40.0, help="source to crystal centre (mm)")
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=4_000_000, help="reference photons")
    ap.add_argument("--histories", type=int, default=4_000_000, help="MCGen histories")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mcgen", default=os.path.join(HERE, "..", "MCGen_CPU"))
    ap.add_argument("--db", default=os.path.join(HERE, "..", "data", "materials.db"))
    args = ap.parse_args()

    E0 = args.energy
    wins = [(E0 - 1.0, E0 + 1.0), (args.cutoff * 0.5, E0 * 2 * (E0 / ME) / (1 + 2 * E0 / ME)),
            (E0 * 2 * (E0 / ME) / (1 + 2 * E0 / ME) + 1.0, E0 - 1.0), (0.5, E0 + 1.0)]
    names = ["photopeak", "single-Compton region", "between edge and peak", "any deposit"]
    grid, mpe, minc = load_material(args.db, args.material)

    # ---------------------------------------------------------------- MCGen run
    with tempfile.TemporaryDirectory() as tmp:
        f = lambda n, v: open(os.path.join(tmp, f"Params_{n}.dat"), "wb").write(struct.pack(f"<{len(v)}f", *v))
        det = [0.0] * 80000
        det[0] = 1
        det[1:13] = [0, args.distance, 0, args.width, args.thickness, args.width, 0.3, 0.2, 0.1, 0.1, 0, 1]
        img = [0.0] * 100
        img[:12] = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        f("Collimator", [0.0] * 80000); f("Detector", det); f("Image", img); f("Physics", [0.0] * 100)
        wspec = ",".join(f"{a:.4f}-{b:.4f}" for a, b in wins)
        cmd = [args.mcgen, "-cpu", "-params_dir", tmp, "-output_dir", tmp, "-energy_keV", str(E0), "-material_db", args.db,
               "-crystal_material", args.material, "-rayleigh", "0", "-fluorescence", "0", "-energy_cutoff_keV", str(args.cutoff),
               "-ewin_mode", "ideal", "-windows", wspec, "-source_sampling", "center", "-histories_per_voxel", str(args.histories),
               "-save_relerr", "1", "-seed", str(args.seed), "-output_prefix", "ref"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout + r.stderr)
        mc = []
        for k in range(len(wins)):
            v = np.fromfile(os.path.join(tmp, f"ref_win{k}.sysmat"), np.float32)[0]
            e = np.fromfile(os.path.join(tmp, f"ref_win{k}_relerr.sysmat"), np.float32)[0]
            mc.append((float(v), float(v * e)))

    # ---------------------------------------------------------------- reference
    dep, w, n = reference(args, grid, mpe, minc, args.n, args.seed + 1)
    print(f"{args.material}, {E0} keV, crystal {args.width}x{args.thickness}x{args.width} mm at {args.distance} mm "
          f"(Rayleigh and fluorescence off)")
    print(f"{'window (keV)':28s} {'MCGen':>12s} {'reference':>12s}   ratio        z")
    worst = 0.0
    for (a, b), name, (v, ev) in zip(wins, names, mc):
        k = np.count_nonzero((dep >= a) & (dep <= b))
        ref = w * k / n
        eref = w * math.sqrt(k * (1 - k / n)) / n
        z = (v - ref) / math.hypot(ev, eref)
        worst = max(worst, abs(z))
        print(f"{name + f' [{a:.1f},{b:.1f}]':28s} {v:12.5e} {ref:12.5e}   {v / ref:.4f}   {z:+.2f}")
    print("OK" if worst < 4 else "MISMATCH")
    sys.exit(0 if worst < 4 else 1)


if __name__ == "__main__":
    main()
