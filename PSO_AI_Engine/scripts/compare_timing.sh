#!/usr/bin/env bash
# ============================================================
# compare_timing.sh  —  time per image: original C++ vs AIE code
#
# For each PSO variant (default all six), over the 1000 frames:
#   1. reference C++     reference cppCode detectors, unmodified, one thread
#   2. AIE code serial   the AIE detection code (pso_core.h) natively,
#                        one thread does all the work of every iteration
#   3. AIE code parallel the same code, THREADS threads inside every PSO
#                        iteration (particles split over the threads), the
#                        way the AIE graph spreads them over worker tiles
# then scripts/evaluate.py writes the per-image times into
# results/<Variant>/table/<Variant>_results.csv and results/summary.csv.
#
# Run it alone (no simulator or other heavy job): it measures wall time.
#
# Usage, from anywhere:
#   PSO_AI_Engine/scripts/compare_timing.sh [THREADS] [vi ...]   (default 6, all six)
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${1:-6}"
shift || true
VIS=("$@")
[ ${#VIS[@]} -gt 0 ] || VIS=(0 1 2 3 4 5)
NAMES=(StandardPSO_DBSCAN StandardPSO_noDBSCAN APSO_DBSCAN APSO_noDBSCAN SPPSO_DBSCAN SPPSO_noDBSCAN)
REF="../Particle-Swarm-Optimisation-for-Multiple-Ellipse-Detection--variantPSO"

SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
# the Vitis environment puts an older libstdc++ first; link it statically
CXX=(g++ -std=c++17 -O2 -static-libstdc++ -static-libgcc)

cd "$PROJECT"
mkdir -p build/bench results/timing
[ -f data/pso_input.txt ] || "${PYTHON[@]}" scripts/make_input.py

"${CXX[@]}" -I"$REF/cppCode/scriptCode" bench/reference_cpp.cpp -o build/bench/reference_cpp 2> /dev/null
"${CXX[@]}" -DPSO_NATIVE -pthread bench/pso_bench.cpp -o build/bench/pso_bench

for VI in "${VIS[@]}"; do
    V="${NAMES[$VI]}"
    echo "== $V (vi $VI) =="
    build/bench/reference_cpp --variant "$VI" --data "$REF/data" --out results/timing
    build/bench/pso_bench --variant "$VI" --threads 1 --out results/timing
    mv results/timing/pso_results.txt "results/timing/${V}_serial.txt"
    build/bench/pso_bench --variant "$VI" --threads "$THREADS" --out results/timing
    mv results/timing/pso_results.txt "results/timing/${V}_parallel.txt"
    echo "$THREADS" > "results/timing/${V}_threads.txt"
    # detections must not depend on the thread count
    "${PYTHON[@]}" - "$V" <<'EOF'
import sys
from pathlib import Path
v = sys.argv[1]
def recs(p):
    w = [t for t in Path(p).read_text().split() if t != "TLAST"]
    n = 1 + 13 + 200                                         # header + scalars + history
    return [w[i:i + 13] + w[i + 14:i + n] for i in range(0, len(w), n)]   # time word dropped
a, b = recs(f"results/timing/{v}_serial.txt"), recs(f"results/timing/{v}_parallel.txt")
print(f"  serial vs parallel detections identical: {a == b}")
if a != b:
    sys.exit(1)
EOF
    "${PYTHON[@]}" scripts/evaluate.py --variant "$V" --source bench
done
echo
column -s, -t < results/summary.csv
