#!/usr/bin/env python3
"""
Write a small, self-consistent set of parameter files
(Params_Collimator.dat, Params_Detector.dat, Params_Image.dat, Params_Physics.dat)
for trying out the analytic generators, MCGen_MultiGPU and the Visualizer.

Attenuation coefficients at the photon energy are taken from the Monte Carlo
material database (MCGen_MultiGPU/data/materials.db) so that the analytic
modules and the MC code see the same materials.

Examples:
    python3 tools/make_demo_params.py --out demo_tc99m                       # 140.5 keV, W multi-pinhole, NaI
    python3 tools/make_demo_params.py --out demo_511 --energy 511 --crystal LYSO --thickness 10
    python3 tools/make_demo_params.py --layout parallel --hole-radius 0.6 --pitch 1.6
"""
import argparse
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "..", "MCGen_MultiGPU", "data", "materials.db")


def load_db(path):
    """Return grid, {name: (density, pe[], inc[], coh[])} (cm^2/g)."""
    mats, grid, cur = {}, None, None
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            t = line.split()
            if t[0] == "EGRID":
                n, e0, e1 = int(t[1]), float(t[2]), float(t[3])
                grid = [e0 * (e1 / e0) ** (i / (n - 1)) for i in range(n)]
            elif t[0] == "MATERIAL":
                cur = mats.setdefault(t[1], {})
            elif cur is not None and t[0] == "DENSITY":
                cur["rho"] = float(t[1])
            elif cur is not None and t[0] in ("MU_PE", "MU_INC", "MU_COH"):
                cur[t[0]] = [float(v) for v in t[1:]]
    return grid, mats


def coeffs(grid, mats, name, energy):
    """(total excl. Rayleigh, photoelectric, Compton) in 1/mm at `energy`."""
    m = mats[name]
    lg = [math.log(e) for e in grid]
    x = math.log(energy)
    i = max(0, min(len(grid) - 2, int((x - lg[0]) / (lg[1] - lg[0]))))
    f = (x - lg[i]) / (lg[i + 1] - lg[i])
    at = lambda v: v[i] * (1 - f) + v[i + 1] * f
    s = m["rho"] / 10.0
    pe, inc = at(m["MU_PE"]) * s, at(m["MU_INC"]) * s
    return pe + inc, pe, inc


def write(path, values, size):
    arr = list(values) + [0.0] * (size - len(values))
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{size}f", *arr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_params")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--energy", type=float, default=140.5)
    ap.add_argument("--collimator", default="W")
    ap.add_argument("--crystal", default="NaI")
    ap.add_argument("--layout", choices=["pinhole", "parallel"], default="pinhole")
    ap.add_argument("--plate", type=float, nargs=3, default=[80.0, 6.0, 80.0], metavar=("W", "T", "H"),
                    help="collimator plate width(x) thickness(y) height(z) in mm")
    ap.add_argument("--holes", type=int, default=5, help="holes per row (pinhole layout)")
    ap.add_argument("--pitch", type=float, default=12.0, help="hole pitch in mm")
    ap.add_argument("--hole-radius", type=float, default=0.75)
    ap.add_argument("--crystals", type=int, default=32, help="crystals per row")
    ap.add_argument("--crystal-pitch", type=float, default=2.0)
    ap.add_argument("--crystal-size", type=float, default=1.8)
    ap.add_argument("--thickness", type=float, default=6.0, help="crystal thickness along y (mm)")
    ap.add_argument("--det-distance", type=float, default=30.0, help="collimator centre -> crystal centre (mm)")
    ap.add_argument("--fov", type=int, nargs=3, default=[40, 40, 1], metavar=("NX", "NY", "NZ"))
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--fov2col", type=float, default=40.0)
    ap.add_argument("--rotations", type=int, default=1)
    ap.add_argument("--eres", type=float, default=0.10, help="FWHM energy resolution at E0")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                    help="fixed energy window in keV for all crystals (Param_Physics[4]=1); "
                         "default: per crystal (1 -/+ eres/2) E0")
    args = ap.parse_args()

    grid, mats = load_db(args.db)
    os.makedirs(args.out, exist_ok=True)
    E = args.energy
    if args.rotations < 1:
        sys.exit("--rotations must be >= 1")
    ct, cpe, cc = coeffs(grid, mats, args.collimator, E)
    dt, dpe, dc = coeffs(grid, mats, args.crystal, E)

    # ---------------------------------------------------------------- collimator
    W, T, H = args.plate
    holes = []
    if args.layout == "pinhole":
        n = args.holes
        for i in range(n):
            for k in range(n):
                holes.append(((i - (n - 1) / 2) * args.pitch, (k - (n - 1) / 2) * args.pitch))
    else:  # parallel hole, hexagonal packing over the plate
        p = args.pitch
        dz = p * math.sqrt(3) / 2
        nrow = int((H - p) / dz + 1e-6) + 1 if H >= p else 0
        z0 = -(nrow - 1) * dz / 2                     # rows centred on the plate
        for row in range(nrow):
            z = z0 + row * dz
            x = -W / 2 + p / 2 + (p / 2 if row % 2 else 0)
            while x <= W / 2 - p / 2 + 1e-6:
                holes.append((x, z))
                x += p
    # the analytic programs read exactly 80000 floats per parameter file (8877 holes, 6666 crystals)
    if len(holes) > 8877:
        sys.exit(f"{len(holes)} holes: the analytic programs support at most 8877 (Params_Collimator.dat = 80000 floats)")
    if args.crystals ** 2 > 6666:
        sys.exit(f"{args.crystals ** 2} crystals: the analytic programs support at most 6666 (Params_Detector.dat = 80000 floats)")
    col = [0.0] * (100 + 9 * len(holes))
    col[0] = 1
    col[10:18] = [len(holes), W, T, H, 0.0, ct, cpe, cc]
    for h, (x, z) in enumerate(holes):
        col[100 + 9 * h: 109 + 9 * h] = [x, -T / 2, T / 2, z, args.hole_radius, 0.0, 0.0, 0.0, 1.0]
    write(os.path.join(args.out, "Params_Collimator.dat"), col, 80000)

    # ---------------------------------------------------------------- detector
    n = args.crystals
    det = [float(n * n)]
    for iz in range(n):
        for ix in range(n):
            det += [(ix - (n - 1) / 2) * args.crystal_pitch, args.det_distance,
                    (iz - (n - 1) / 2) * args.crystal_pitch,
                    args.crystal_size, args.thickness, args.crystal_size,
                    dt, dpe, dc, args.eres, 0.0, 1.0]
    write(os.path.join(args.out, "Params_Detector.dat"), det, 80000)

    # ---------------------------------------------------------------- image / physics
    nx, ny, nz = args.fov
    img = [nx, ny, nz, args.voxel, args.voxel, args.voxel, args.rotations,
           2 * math.pi / args.rotations, 0.0, 0.0, 0.0, args.fov2col]
    write(os.path.join(args.out, "Params_Image.dat"), img, 100)
    lo, hi = args.window if args.window else (E * (1 - args.eres / 2), E * (1 + args.eres / 2))
    phy = [1, 1, 1, 1, 1 if args.window else 0, lo, hi, E, 1, 1]
    write(os.path.join(args.out, "Params_Physics.dat"), phy, 100)

    print(f"wrote {args.out}/Params_*.dat")
    print(f"  {len(holes)} holes (r={args.hole_radius} mm) in {args.collimator} plate {W}x{T}x{H} mm, "
          f"mu={ct:.4f}/mm (PE {cpe:.4f}, Compton {cc:.4f}) at {E} keV")
    print(f"  {n}x{n} {args.crystal} crystals {args.crystal_size}x{args.thickness}x{args.crystal_size} mm, "
          f"mu={dt:.4f}/mm (PE {dpe:.4f}, Compton {dc:.4f})")
    print(f"  FOV {nx}x{ny}x{nz} voxels of {args.voxel} mm, {args.rotations} rotation(s), FOV->collimator {args.fov2col} mm")
    print(f"  energy window {lo:.2f}-{hi:.2f} keV" + ("" if args.window else " (per crystal from eres)"))


if __name__ == "__main__":
    main()
