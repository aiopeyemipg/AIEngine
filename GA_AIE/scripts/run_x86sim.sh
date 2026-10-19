#!/usr/bin/env bash
# ============================================================
# run_x86sim.sh  —  build, simulate and evaluate the GA graph
#
# One GA variant at a time (default: ConventionalGA, MicroGA, IslandGA):
#   1. writes data/ga_input.txt from the PNG images
#   2. compiles the graph for the x86 simulator (build/x86sim_<Variant>)
#      and draws it (report_figures/adf_graph_<Variant>.png)
#   3. runs x86simulator: the images one after another, every generation
#      spread over 1 master + GA_WORKERS worker tiles
#   4. evaluates the detections (scripts/evaluate.py); the per-image
#      times of scripts/compare_timing.sh are merged in if present
#
# Usage, from anywhere:
#   GA_AIE/scripts/run_x86sim.sh [ConventionalGA] [MicroGA] [IslandGA]
# Images per run: GA_X86SIM_IMAGES (default 1000). x86sim is a slow
# functional model: the Conventional GA takes ~13 s per image there, so
# e.g. GA_X86SIM_IMAGES=100 checks it on a subset; a subset is evaluated
# into results/x86sim_subset/ and leaves the 1000-image tables alone.
# The x86sim wall time per variant is written to results/timing_x86sim.csv.
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VITIS_SETTINGS="${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2024.2/settings64.sh}"
SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
IMAGES="${GA_X86SIM_IMAGES:-1000}"

VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(ConventionalGA MicroGA IslandGA)

"${PYTHON[@]}" "$PROJECT/scripts/make_input.py"

# shellcheck disable=SC1090
source "$VITIS_SETTINGS" > /dev/null

TIMES="$PROJECT/results/timing_x86sim.csv"
mkdir -p "$PROJECT/results"
[ -f "$TIMES" ] || echo "variant,images,x86sim_wall_s" > "$TIMES"

for V in "${VARIANTS[@]}"; do
    case "$V" in
        ConventionalGA) VI=0 ;;
        MicroGA)        VI=1 ;;
        IslandGA)       VI=2 ;;
        *) echo "unknown variant $V (ConventionalGA, MicroGA or IslandGA)"; exit 1 ;;
    esac
    BUILD="$PROJECT/build/x86sim_$V"
    cmake -S "$PROJECT" -B "$BUILD" -DVITIS_TARGET=x86sim -DGA_VARIANT=$VI -DGA_IMAGES="$IMAGES" > /dev/null
    make -C "$BUILD"
    "${PYTHON[@]}" "$PROJECT/scripts/plot_adf_graph.py" --work "$BUILD/Work" --name "$V" || true

    cd "$BUILD"
    rm -rf x86simulator_output
    start=$(date +%s)
    x86simulator --pkg-dir=./Work --input-dir=../.. --output-dir=./x86simulator_output
    wall=$(( $(date +%s) - start ))
    echo "$V x86simulator wall time: $wall s ($IMAGES images)"
    grep -v "^$V," "$TIMES" > "$TIMES.tmp" || true
    mv "$TIMES.tmp" "$TIMES"
    echo "$V,$IMAGES,$wall" >> "$TIMES"

    if [ "$IMAGES" -ge 1000 ]; then
        "${PYTHON[@]}" "$PROJECT/scripts/evaluate.py" --variant "$V" --source x86sim
    else
        mkdir -p "$PROJECT/results/x86sim_subset"
        cp -r "$PROJECT/results/timing" "$PROJECT/results/x86sim_subset/" 2> /dev/null || true
        "${PYTHON[@]}" "$PROJECT/scripts/evaluate.py" --variant "$V" --source x86sim \
            --images "$IMAGES" --results "$PROJECT/results/x86sim_subset"
    fi
done
