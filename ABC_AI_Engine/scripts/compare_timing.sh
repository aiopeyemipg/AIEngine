#!/usr/bin/env bash
# ============================================================
# compare_timing.sh  —  time per image: original C++ vs AIE code
#
# For each algorithm (HA = hollowness-aware, SR = standard-residual),
# over the 1000 frames:
#   1. reference C++   sequential C++ port of the Python scripts
#                      (bench/reference_cpp.cpp), one thread
#   2. AIE code serial the AIE detection code (abc_core.h) natively,
#                      one thread does every bee task of every iteration
#   3. AIE code parallel the same code, THREADS threads inside every ABC
#                      iteration (the bees of each phase), the way the
#                      AIE graph spreads them over worker tiles
# then scripts/evaluate.py writes the per-image times into
# results/<Variant>/table/<Variant>_results.csv and results/summary.csv.
#
# Run it alone (no simulator or other heavy job): it measures wall time.
#
# Usage, from anywhere:
#   ABC_AI_Engine/scripts/compare_timing.sh [THREADS]      (default 6)
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${1:-6}"
SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
# evaluate.py needs numpy + OpenCV: the environment of ~/.venvs/aie_plots
VP=/tools/Xilinx/Vitis/2024.2/tps/lnx64/python-3.8.3
MPL=(env -u PYTHONPATH -u PYTHONHOME "LD_LIBRARY_PATH=$VP/lib" "${MPL_PYTHON:-$HOME/.venvs/aie_plots/bin/python}")
# the Vitis environment puts an older libstdc++ first; link it statically
CXX=(g++ -std=c++17 -O2 -static-libstdc++ -static-libgcc)

cd "$PROJECT"
mkdir -p build/bench results/timing
[ -f data/abc_input.txt ] || "${PYTHON[@]}" scripts/make_input.py

"${CXX[@]}" bench/reference_cpp.cpp -o build/bench/reference_cpp
"${CXX[@]}" -DABC_NATIVE -pthread bench/abc_bench.cpp -o build/bench/abc_bench

for V in HA SR; do
    echo "== $V-ABC =="
    build/bench/reference_cpp --variant "$V" --out results/timing
    build/bench/abc_bench --variant "$V" --threads 1 --out results/timing
    mv results/timing/abc_results.txt "results/timing/${V}_serial.txt"
    build/bench/abc_bench --variant "$V" --threads "$THREADS" --out results/timing
    mv results/timing/abc_results.txt "results/timing/${V}_parallel.txt"
    echo "$THREADS" > "results/timing/${V}_threads.txt"
    # detections must not depend on the thread count
    "${PYTHON[@]}" - "$V" <<'EOF'
import sys
from pathlib import Path
v = sys.argv[1]
def recs(p):
    w = [t for t in Path(p).read_text().split() if t != "TLAST"]
    n = 1 + 13 + 150                                             # header + record (include.h)
    return [w[i:i + 13] + w[i + 14:i + n] for i in range(0, len(w), n)]   # time word dropped
a, b = recs(f"results/timing/{v}_serial.txt"), recs(f"results/timing/{v}_parallel.txt")
print(f"  serial vs parallel detections identical: {a == b}")
if a != b:
    sys.exit(1)
EOF
    "${MPL[@]}" scripts/evaluate.py --variant "$V" --source bench
done
echo
column -s, -t < results/summary.csv
