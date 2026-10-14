#!/usr/bin/env bash
# Analytic system matrix (PEGen + ScatterGen) versus MCGen_MultiGPU on one small multi-pinhole
# example, component by component (photoelectric / crystal scatter / collimator scatter).
#
#   ./run_example.sh [tc99m|511] [gpu|cpu] [histories_per_voxel]
#   SCATTER_MODEL=legacy ./run_example.sh ...   runs the old ScatterGen kernels instead of model v2
#
#   gpu (default): the original CUDA programs on ONE GPU (GPU=0 by default, e.g. GPU=1 ./run_example.sh):
#                  PEGen_CircularHole -> ScatterGen_CircularHole -components -> MCGen_MultiGPU -gpus $GPU
#                  build them first: PEGen_*/bd, ScatterGen_RayTracing_CircularHole/bd, (cd MCGen_MultiGPU && make)
#   cpu:           the same analytic kernels compiled for the CPU (tools/analytic_cpu) and MCGen_CPU
#                  (cd MCGen_MultiGPU && make cpu); ~10-20 min per case on 4 cores
#
# Results: run_<case>/report.md (tables), run_<case>/flood.svg (detector images), run_<case>/report.json
set -euo pipefail

CASE=${1:-tc99m}
MODE=${2:-gpu}
NH=${3:-4000000}
GPU=${GPU:-0}
SCATTER_MODEL=${SCATTER_MODEL:-v2}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-$HERE/run_$CASE}
SHIFT=0.000000_0.000000_0.000000

# Geometry shared by both cases: W plate with 3x3 cylindrical pinholes (r 0.75 mm, pitch 12 mm),
# 16x16 pixelated crystals (2.3 mm, pitch 2.5 mm) 30 mm behind the plate centre,
# 16x16 voxels of 2 mm in the plane 40 mm in front of it.
COMMON="--layout pinhole --holes 3 --pitch 12 --hole-radius 0.75 --crystals 16 --crystal-pitch 2.5 --crystal-size 2.3
        --det-distance 30 --fov 16 1 16 --voxel 2 --fov2col 40 --rotations 1"
case "$CASE" in
  tc99m) GEOM="--energy 140.511 --collimator W --crystal NaI  --plate 60 6 60  --thickness 6  --eres 0.10 --window 126.46 154.56"
         MATS="-collimator_material W -crystal_material NaI";  WIDE_LO=60;  WIDE_HI=200 ;;
  511)   GEOM="--energy 511     --collimator W --crystal LYSO --plate 60 10 60 --thickness 10 --eres 0.12 --window 459.9 562.1"
         MATS="-collimator_material W -crystal_material LYSO"; WIDE_LO=250; WIDE_HI=650 ;;
  *) echo "case must be tc99m or 511"; exit 1 ;;
esac

need() { [ -x "$1" ] || { echo "missing $1 - $2"; exit 1; }; }
if [ "$MODE" = gpu ]; then
  PEGEN=$ROOT/PEGen_RayTracing_CircularHole/PEGen_CircularHole
  SCGEN=$ROOT/ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole
  MC=$ROOT/MCGen_MultiGPU/MCGen_MultiGPU
  need "$PEGEN" "build with PEGen_RayTracing_CircularHole/bd"
  need "$SCGEN" "build with ScatterGen_RayTracing_CircularHole/bd"
  need "$MC" "build with: cd MCGen_MultiGPU && make"
  MCARGS="-backend gpu -gpus $GPU"
else
  python3 "$ROOT/tools/analytic_cpu/build.py" > /dev/null
  MC=$ROOT/MCGen_MultiGPU/MCGen_CPU
  need "$MC" "build with: cd MCGen_MultiGPU && make cpu"
  MCARGS="-backend cpu"
fi

echo "== parameters -> $OUT"
python3 "$ROOT/tools/make_demo_params.py" --out "$OUT" $COMMON $GEOM
# same geometry, wide energy window: analytic scatter for MC window 1
mkdir -p "$OUT/wide"
cp "$OUT"/Params_*.dat "$OUT/wide/"
python3 - "$OUT/wide/Params_Physics.dat" "$WIDE_LO" "$WIDE_HI" <<'EOF'
import struct, sys
p = sys.argv[1]
v = list(struct.unpack("<100f", open(p, "rb").read(400)))
v[4], v[5], v[6] = 1.0, float(sys.argv[2]), float(sys.argv[3])
open(p, "wb").write(struct.pack("<100f", *v))
EOF

PE=PE_SysMat_shift_${SHIFT}_v3.sysmat
echo "== analytic model ($MODE)"
if [ "$MODE" = gpu ]; then
  (cd "$OUT" && "$PEGEN" -cuda "$GPU" > pegen.log)
  (cd "$OUT" && "$SCGEN" -PE "$PE" -cuda "$GPU" -components -model "$SCATTER_MODEL" > scattergen.log)
  (cd "$OUT/wide" && "$SCGEN" -PE "../$PE" -cuda "$GPU" -components -model "$SCATTER_MODEL" > scattergen.log)
else
  B=$ROOT/tools/analytic_cpu/build
  "$B/pe_cpu" "$OUT"
  "$B/scatter_cpu" "$OUT" "$OUT/$PE" -model "$SCATTER_MODEL"
  "$B/scatter_cpu" "$OUT/wide" "$OUT/$PE" -model "$SCATTER_MODEL"
fi

echo "== Monte Carlo ($MODE, $NH histories per voxel)"
CFG=$HERE/MC_example.txt
cd "$OUT"
# 1) analytic-equivalent physics: coefficients from the parameter files (no Rayleigh), a photon is
#    counted only if its first interaction is photoelectric, ideal window -> checks the PE kernel
"$MC" "$CFG" $MCARGS -output_prefix mc_pe -collimator_material params -crystal_material params -hole_material vacuum \
      -physics primary_pe -ewin_mode ideal -windows params -save_components 0 -histories_per_voxel "$NH" > mc_pe.log
# 2) first-order scatter with the same physics as the analytic model: at most one Compton scatter,
#    only photoelectric absorption scored -> exact reference for ScatterGen model v2
"$MC" "$CFG" $MCARGS -output_prefix mc_single -collimator_material params -crystal_material params -hole_material vacuum \
      -physics single_scatter -windows "params,$WIDE_LO-$WIDE_HI" -histories_per_voxel "$NH" > mc_single.log
# 3) Compton + photoelectric only, coefficients and E^-3 / Klein-Nishina scaling from the parameter
#    files, no Rayleigh, no fluorescence: the physics the analytic scatter models assume
"$MC" "$CFG" $MCARGS -output_prefix mc_model -collimator_material params -crystal_material params -hole_material vacuum \
      -rayleigh 0 -fluorescence 0 -windows "params,$WIDE_LO-$WIDE_HI" -histories_per_voxel "$NH" > mc_model.log
# 4) full physics with the material database (Rayleigh, K fluorescence, exact cross sections)
"$MC" "$CFG" $MCARGS $MATS -output_prefix mc_full -windows "params,$WIDE_LO-$WIDE_HI" -histories_per_voxel "$NH" > mc_full.log

echo "== comparison"
python3 "$HERE/compare_components.py" "$OUT" --wide "$OUT/wide" --md "$OUT/report.md" --svg "$OUT/flood.svg" --json "$OUT/report.json"
