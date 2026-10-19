#!/usr/bin/env bash
# ============================================================
# compare_timing.sh  —  time per image: original C++ vs AIE code
#
# For each GA variant (ConventionalGA, MicroGA, IslandGA), 1000 frames:
#   1. reference C++   GaCppPython/cppCode detectors, unmodified, one thread
#   2. AIE code serial the AIE detection code (ga_algorithms.h) natively,
#                      one thread does all the work of every generation
#   3. AIE code parallel the same code, THREADS threads inside every
#                      generation (children or islands), the way the AIE
#                      graph spreads them over worker tiles
# then scripts/evaluate.py writes the per-image times into
# results/<Variant>/table/<Variant>_results.csv and results/summary.csv.
#
# Run it alone (no simulator or other heavy job): it measures wall time.
#
# Usage, from anywhere:
#   GA_AIE/scripts/compare_timing.sh [THREADS] [variants...]   (default 6, all three)
# ============================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${1:-6}"
shift || true
VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(ConventionalGA MicroGA IslandGA)
REF_SRC="$HOME/Desktop/GaCppPython/cppCode/scriptCode"
SYS_PYTHON="/usr/bin/python3"
[ -x "$SYS_PYTHON" ] || SYS_PYTHON="$(command -v python3)"
PYTHON=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SYS_PYTHON" -I)
# the Vitis environment puts an older libstdc++ first; link it statically
CXX=(g++ -std=c++17 -O2 -static-libstdc++ -static-libgcc)

cd "$PROJECT"
mkdir -p build/bench results/timing
[ -f data/ga_input.txt ] || "${PYTHON[@]}" scripts/make_input.py

"${CXX[@]}" -I"$REF_SRC" bench/reference_cpp.cpp -o build/bench/reference_cpp 2> /dev/null
"${CXX[@]}" -DGA_NATIVE -pthread bench/ga_bench.cpp -o build/bench/ga_bench

for V in "${VARIANTS[@]}"; do
    echo "== $V =="
    build/bench/reference_cpp --variant "$V" --data "$HOME/Desktop/GaCppPython/data" --out results/timing
    build/bench/ga_bench --variant "$V" --threads 1 --out results/timing
    mv results/timing/ga_results.txt "results/timing/${V}_serial.txt"
    build/bench/ga_bench --variant "$V" --threads "$THREADS" --out results/timing
    mv results/timing/ga_results.txt "results/timing/${V}_parallel.txt"
    echo "$THREADS" > "results/timing/${V}_threads.txt"
    # detections must not depend on the thread count
    "${PYTHON[@]}" - "$V" <<'EOF'
import sys
from pathlib import Path
v = sys.argv[1]
hist = {"ConventionalGA": 74, "MicroGA": 390, "IslandGA": 240}[v]   # include.h *_HIST_LEN
def recs(p):
    w = [t for t in Path(p).read_text().split() if t != "TLAST"]
    n = 1 + 13 + hist                                            # header + record
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
