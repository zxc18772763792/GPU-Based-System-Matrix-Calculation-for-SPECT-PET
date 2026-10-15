#!/usr/bin/env python3
"""
Build CPU versions of the ORIGINAL analytic CUDA kernels, for checks on machines without a GPU.

The kernel code is copied verbatim from
    PEGen_RayTracing_CircularHole/PESysMatGen.cu          (everything before PESysMatGen())
    ScatterGen_RayTracing_CircularHole/scatter.cu         (everything before scatter())
and compiled with g++ together with prelude.h (CPU stand-ins for blockIdx, atomicAdd, ...)
and the small harnesses pe_cpu.cpp / scatter_cpu.cpp, which replicate the host code.
Any edit of the .cu kernels is therefore picked up by simply re-running this script.

    python3 tools/analytic_cpu/build.py            # -> tools/analytic_cpu/build/{pe_cpu,scatter_cpu}
    tools/analytic_cpu/build/pe_cpu <params_dir>
    tools/analytic_cpu/build/scatter_cpu <params_dir> <params_dir>/PE_SysMat_shift_..._v3.sysmat

It is slow (one CPU thread per CUDA thread) and meant for small validation examples.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SOURCES = {
    "pe": ("PEGen_RayTracing_CircularHole/PESysMatGen.cu", r"^int\s+PESysMatGen\s*\("),
    "scatter": ("ScatterGen_RayTracing_CircularHole/scatter.cu", r"^int\s+scatter\s*\("),
}
DROP = re.compile(r'^\s*#\s*include\s*[<"](cuda_runtime|device_launch_parameters|scatter|PESysMatGen)\.h[>"]')


def extract(rel, marker):
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read().replace("\r\n", "\n")
    m = re.search(marker, text, re.M)
    if not m:
        sys.exit(f"{rel}: host function marker {marker!r} not found")
    lines = [ln for ln in text[: m.start()].split("\n") if not DROP.match(ln)]
    return f'#line 1 "{rel}"\n' + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "build"))
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--flags", default="-O3 -std=c++17 -pthread")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, (rel, marker) in SOURCES.items():
        with open(os.path.join(args.out, f"{name}_kernels.inc"), "w") as f:
            f.write(extract(rel, marker))
        exe = os.path.join(args.out, f"{name}_cpu")
        cmd = [args.cxx, *args.flags.split(), "-I", HERE, "-I", args.out, "-o", exe, os.path.join(HERE, f"{name}_cpu.cpp")]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
    print(f"built {args.out}/pe_cpu and {args.out}/scatter_cpu")


if __name__ == "__main__":
    main()
