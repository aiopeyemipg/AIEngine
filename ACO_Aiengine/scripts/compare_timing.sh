#!/usr/bin/env bash
# ============================================================
# compare_timing.sh  —  time per image: original C++ vs AIE code
#
# For each algorithm (FPED, StdACO), over the 1000 frames:
#   1. reference C++   ACO/cppCode detectors, unmodified, one thread
#   2. AIE code serial the AIE detection code (aco_core.h) natively,
#                      one thread does all the work of every iteration
#   3. AIE code parallel the same code, THREADS threads inside every
#                      colony iteration (trails, then ants), the way the
#                      AIE graph spreads them over worker tiles
# then scripts/evaluate.py writes the per-image times into
# results/<Variant>/table/<Variant>_results.csv and results/summary.csv.
#
# Run it alone (no simulator or other heavy job): it measures wall time.
#
# Usage, from anywhere:
#   ACO_Aiengine/scripts/compare_timing.sh [THREADS]      (default 6)
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${1:-6}"
SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
# the Vitis environment puts an older libstdc++ first; link it statically
CXX=(g++ -std=c++17 -O2 -static-libstdc++ -static-libgcc)

cd "$PROJECT"
mkdir -p build/bench results/timing
[ -f data/aco_input.txt ] || "${PYTHON[@]}" scripts/make_input.py

"${CXX[@]}" -I../ACO/cppCode/scriptCode bench/reference_cpp.cpp -o build/bench/reference_cpp 2> /dev/null
"${CXX[@]}" -DACO_NATIVE -pthread bench/aco_bench.cpp -o build/bench/aco_bench

for V in FPED StdACO; do
    echo "== $V =="
    build/bench/reference_cpp --variant "$V" --data ../ACO/data --out results/timing
    build/bench/aco_bench --variant "$V" --threads 1 --out results/timing
    mv results/timing/aco_results.txt "results/timing/${V}_serial.txt"
    build/bench/aco_bench --variant "$V" --threads "$THREADS" --out results/timing
    mv results/timing/aco_results.txt "results/timing/${V}_parallel.txt"
    echo "$THREADS" > "results/timing/${V}_threads.txt"
    # detections must not depend on the thread count
    "${PYTHON[@]}" - "$V" <<'EOF'
import sys
from pathlib import Path
v = sys.argv[1]
def recs(p):
    w = [t for t in Path(p).read_text().split() if t != "TLAST"]
    n = 1 + 13 + 100                                             # header + record (include.h)
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
