#!/usr/bin/env bash
# ============================================================
# run_x86sim.sh  —  build, simulate and evaluate the ABC graph
#
# One algorithm at a time (default: HA, then SR):
#   1. writes data/abc_input.txt from the PNG images
#   2. compiles the graph for the x86 simulator (build/x86sim_<Variant>)
#   3. runs x86simulator: the images one after another, every phase of
#      every ABC iteration spread over ABC_WORKERS worker tiles
#   3b. draws the compiled ADF graph (report_figures/adf_graph_<Variant>.png)
#   4. evaluates the detections (scripts/evaluate.py); the per-image
#      times of scripts/compare_timing.sh are merged in if present
#
# Usage, from anywhere:
#   [IMAGES=N] ABC_AI_Engine/scripts/run_x86sim.sh [HA] [SR]
# IMAGES (default 1000) limits the run: x86sim needs about 6 s per image
# here (600 broadcast/gather rounds between 31 simulated tiles per image).
# The x86sim wall time per algorithm is written to results/timing_x86sim.csv.
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VITIS_SETTINGS="${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2024.2/settings64.sh}"
SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
VP=/tools/Xilinx/Vitis/2024.2/tps/lnx64/python-3.8.3
MPL=(env -u PYTHONPATH -u PYTHONHOME "LD_LIBRARY_PATH=$VP/lib" "${MPL_PYTHON:-$HOME/.venvs/aie_plots/bin/python}")
IMAGES="${IMAGES:-1000}"

VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(HA SR)

"${PYTHON[@]}" "$PROJECT/scripts/make_input.py"

# shellcheck disable=SC1090
source "$VITIS_SETTINGS" > /dev/null

TIMES="$PROJECT/results/timing_x86sim.csv"
mkdir -p "$PROJECT/results"
[ -f "$TIMES" ] || echo "variant,images,x86sim_wall_s" > "$TIMES"

for V in "${VARIANTS[@]}"; do
    case "$V" in
        HA) VI=0 ;;
        SR) VI=1 ;;
        *) echo "unknown variant $V (HA or SR)"; exit 1 ;;
    esac
    BUILD="$PROJECT/build/x86sim_$V"
    cmake -S "$PROJECT" -B "$BUILD" -DVITIS_TARGET=x86sim -DABC_VARIANT=$VI -DABC_IMAGES="$IMAGES" > /dev/null
    make -C "$BUILD"
    # plot of the compiled ADF graph -> report_figures/adf_graph_<Variant>.png
    "${PYTHON[@]}" "$PROJECT/scripts/plot_adf_graph.py" --work "$BUILD/Work" --name "$V"

    cd "$BUILD"
    rm -rf x86simulator_output
    start=$(date +%s)
    x86simulator --pkg-dir=./Work --input-dir=../.. --output-dir=./x86simulator_output
    wall=$(( $(date +%s) - start ))
    echo "$V x86simulator wall time: $wall s"
    grep -v "^$V," "$TIMES" > "$TIMES.tmp" || true
    mv "$TIMES.tmp" "$TIMES"
    echo "$V,$IMAGES,$wall" >> "$TIMES"

    if [ "$IMAGES" -ge 1000 ]; then
        "${MPL[@]}" "$PROJECT/scripts/evaluate.py" --variant "$V" --source x86sim
    else
        # partial run: check it against the native bench record by record
        "${PYTHON[@]}" "$PROJECT/scripts/compare_x86sim_bench.py" --variant "$V" --images "$IMAGES"
    fi
done
