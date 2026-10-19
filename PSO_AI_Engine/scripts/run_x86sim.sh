#!/usr/bin/env bash
# ============================================================
# run_x86sim.sh  —  build, simulate and evaluate the PSO graph
#
# One variant at a time (default: all six, vi 0..5):
#   1. writes data/pso_input.txt from the PNG images
#   2. compiles the graph for the x86 simulator (build/x86sim_<Variant>)
#      and draws the compiled graph (report_figures/adf_graph_<Variant>.*)
#   3. runs x86simulator: the images one after another, every PSO
#      iteration spread over 1 master + PSO_WORKERS worker tiles
#   4. evaluates the detections (scripts/evaluate.py); the per-image
#      times of scripts/compare_timing.sh are merged in if present
#
# Usage, from anywhere:
#   PSO_AI_Engine/scripts/run_x86sim.sh [vi ...]        e.g.  run_x86sim.sh 0 4
# The x86sim wall time per variant is written to results/timing_x86sim.csv.
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VITIS_SETTINGS="${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2024.2/settings64.sh}"
SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
NAMES=(StandardPSO_DBSCAN StandardPSO_noDBSCAN APSO_DBSCAN APSO_noDBSCAN SPPSO_DBSCAN SPPSO_noDBSCAN)

VIS=("$@")
[ ${#VIS[@]} -gt 0 ] || VIS=(0 1 2 3 4 5)

"${PYTHON[@]}" "$PROJECT/scripts/make_input.py"

# shellcheck disable=SC1090
source "$VITIS_SETTINGS" > /dev/null

TIMES="$PROJECT/results/timing_x86sim.csv"
mkdir -p "$PROJECT/results"
[ -f "$TIMES" ] || echo "variant,images,x86sim_wall_s" > "$TIMES"

for VI in "${VIS[@]}"; do
    V="${NAMES[$VI]}"
    BUILD="$PROJECT/build/x86sim_$V"
    cmake -S "$PROJECT" -B "$BUILD" -DVITIS_TARGET=x86sim -DPSO_VARIANT=$VI -DPSO_IMAGES= > /dev/null
    make -C "$BUILD"
    "${PYTHON[@]}" "$PROJECT/scripts/plot_adf_graph.py" --work "$BUILD/Work" --name "$V"

    cd "$BUILD"
    rm -rf x86simulator_output
    start=$(date +%s)
    x86simulator --pkg-dir=./Work --input-dir=../.. --output-dir=./x86simulator_output
    wall=$(( $(date +%s) - start ))
    echo "$V x86simulator wall time: $wall s"
    grep -v "^$V," "$TIMES" > "$TIMES.tmp" || true
    mv "$TIMES.tmp" "$TIMES"
    echo "$V,1000,$wall" >> "$TIMES"

    "${PYTHON[@]}" "$PROJECT/scripts/evaluate.py" --variant "$V" --source x86sim
done
