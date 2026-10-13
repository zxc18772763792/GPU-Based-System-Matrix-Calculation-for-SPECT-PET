#!/usr/bin/env python3
"""
Compare two system matrices with the same [rotation][detector][voxel] float32 layout,
e.g. the analytic PE matrix (PEGen_CircularHole) against MCGen_MultiGPU.

    python3 tools/compare_sysmat.py PE_SysMat_shift_..._v3.sysmat MC_..._win0.sysmat --params .
    python3 tools/compare_sysmat.py A.sysmat B.sysmat --ndet 1024 --nvox 40000 --relerr MC_..._win0_relerr.sysmat

For an apples-to-apples check of the MC against the analytic photoelectric model run the MC with
    -physics primary_pe -source_sampling center -ewin_mode none -collimator_material params
    -crystal_material params -hole_material params -save_relerr 1
and the z-scores (MC - analytic) / sigma_MC should have rms ~ 1.

Files are memory-mapped, one rotation at a time (needs numpy).
"""
import argparse
import os
import struct
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")


def dims_from_params(d):
    def first(name, n):
        for pre in ("Params_", "Param_"):
            p = os.path.join(d, f"{pre}{name}.dat")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return struct.unpack(f"<{n}f", f.read(4 * n))
        sys.exit(f"cannot find {name} parameter file in {d}")
    det = first("Detector", 1)
    img = first("Image", 3)
    return int(det[0] + 1e-3), int(img[0] + 1e-3) * int(img[1] + 1e-3) * int(img[2] + 1e-3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference matrix (e.g. analytic)")
    ap.add_argument("b", help="test matrix (e.g. Monte Carlo)")
    ap.add_argument("--params", help="directory with the parameter files (for the dimensions)")
    ap.add_argument("--ndet", type=int)
    ap.add_argument("--nvox", type=int)
    ap.add_argument("--relerr", help="relative-error matrix of b (MC) for z-scores")
    ap.add_argument("--rot", type=int, default=-1, help="only this rotation (default: all)")
    ap.add_argument("--zcut", type=float, default=0.1,
                    help="z-scores only for elements with MC relative error below this (low-count elements bias z upwards)")
    args = ap.parse_args()

    if args.params:
        ndet, nvox = dims_from_params(args.params)
    elif args.ndet and args.nvox:
        ndet, nvox = args.ndet, args.nvox
    else:
        sys.exit("give --params DIR or --ndet and --nvox")
    per = ndet * nvox
    sa, sb = os.path.getsize(args.a) // 4, os.path.getsize(args.b) // 4
    if sa % per or sb % per or sa != sb:
        sys.exit(f"sizes do not match {ndet} detectors x {nvox} voxels: {sa} vs {sb} floats")
    nrot = sa // per
    A = np.memmap(args.a, np.float32, "r", shape=(nrot, ndet, nvox))
    B = np.memmap(args.b, np.float32, "r", shape=(nrot, ndet, nvox))
    R = np.memmap(args.relerr, np.float32, "r", shape=(nrot, ndet, nvox)) if args.relerr else None

    rots = [args.rot] if args.rot >= 0 else range(nrot)
    print(f"{nrot} rotation(s) x {ndet} detectors x {nvox} voxels")
    print("rot   sum(A)        sum(B)        B/A      voxel-sens ratio (mean / std / min / max)   95%-signal wrms   z mean / rms / |z|>3 (relerr < zcut)")
    for r in rots:
        a = np.asarray(A[r], dtype=np.float64)
        b = np.asarray(B[r], dtype=np.float64)
        va, vb = a.sum(0), b.sum(0)
        ok = va > 0
        ratio = vb[ok] / va[ok]
        flat = a.ravel()
        order = np.argsort(flat)[::-1]
        c = np.cumsum(flat[order])
        k = int(np.searchsorted(c, 0.95 * c[-1])) + 1 if c[-1] > 0 else 0
        idx = order[:k]
        rel = (b.ravel()[idx] - flat[idx]) / flat[idx] if k else np.zeros(1)
        w = flat[idx] if k else np.ones(1)
        line = (f"{r:3d}  {a.sum():.5e}  {b.sum():.5e}  {b.sum() / a.sum() if a.sum() > 0 else float('nan'):7.4f}   "
                f"{ratio.mean():.4f} / {ratio.std():.4f} / {ratio.min():.4f} / {ratio.max():.4f}        "
                f"{np.sqrt(np.average(rel ** 2, weights=w)):.4f}")
        if R is not None:
            rr = np.asarray(R[r], dtype=np.float64)
            m = (a > 0) & (b > 0) & (rr > 0) & (rr < args.zcut)
            z = (b[m] - a[m]) / (rr[m] * b[m])
            line += f"        {z.mean():.3f} / {np.sqrt((z ** 2).mean()):.3f} / {100 * np.mean(np.abs(z) > 3):.2f}%"
        print(line)


if __name__ == "__main__":
    main()
