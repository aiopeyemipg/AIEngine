#!/usr/bin/env bash
# ============================================================
# make_figures.sh  —  every figure of the ACO AIE study
#
#   report_figures/adf_graph_<Variant>.png/.svg   compiled ADF graph
#   report_figures/timing_mean.png                 C++ vs AIE code, ms/image
#   report_figures/timing_per_image.png            per-image time distribution
#   report_figures/accuracy.png                    accuracy, C++ vs AIE code
#   results/<Variant>/figures/id_<n>_detection.png    per image
#   results/<Variant>/figures/id_<n>_convergence.png  per image
#
# Run after scripts/run_x86sim.sh and scripts/compare_timing.sh.
# matplotlib comes from ~/.venvs/aie_plots (MPL_PYTHON to override).
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VP=/tools/Xilinx/Vitis/2024.2/tps/lnx64/python-3.8.3
MPL=(env -u PYTHONPATH -u PYTHONHOME "LD_LIBRARY_PATH=$VP/lib" "${MPL_PYTHON:-$HOME/.venvs/aie_plots/bin/python}")
SYS=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH /usr/bin/python3 -I)

cd "$PROJECT"
for V in FPED StdACO; do
    [ -f "build/x86sim_$V/Work/temp/project.json" ] &&
        "${SYS[@]}" scripts/plot_adf_graph.py --work "build/x86sim_$V/Work" --name "$V"
done
"${MPL[@]}" scripts/plot_timing.py --project . --title "ACO"
"${MPL[@]}" scripts/plot_images.py --project . --images ../ACO/data/Ellipses --iteration-name Colony "$@"
