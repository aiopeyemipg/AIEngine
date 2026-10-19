"""
compare_x86sim_bench.py  —  check the AIE graph (x86sim) against the native
build of the same detection code, record by record.

Compares every word of the first N result records (the time word excepted)
of build/x86sim_<V>/x86simulator_output/data/abc_results.txt with
results/timing/<V>_serial.txt (bench, packet format). Used after a partial
x86sim run (IMAGES=N scripts/run_x86sim.sh).

Usage: python3 scripts/compare_x86sim_bench.py --variant HA --images 200
"""
import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
WORDS = 13 + 150          # RESULT_WORDS
TIME = 12                 # RES_TIME_US


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["HA", "SR"], required=True)
    ap.add_argument("--images", type=int, required=True)
    a = ap.parse_args()
    sim = [int(t) for t in (PROJECT / "build" / f"x86sim_{a.variant}" / "x86simulator_output" / "data"
                            / "abc_results.txt").read_text().split() if t.lstrip("-").isdigit()]
    b = [int(t) for t in (PROJECT / "results" / "timing" / f"{a.variant}_serial.txt").read_text().split()
         if t != "TLAST"]
    same = 0
    diff = []
    for i in range(a.images):
        s = sim[i * WORDS:(i + 1) * WORDS]
        q = b[i * (WORDS + 1) + 1:(i + 1) * (WORDS + 1)]
        bad = [j for j in range(WORDS) if j != TIME and (s[j] & 0xFFFFFFFF) != (q[j] & 0xFFFFFFFF)]
        if bad:
            diff.append((s[1], len(bad)))
        else:
            same += 1
    print(f"{a.variant}: {same}/{a.images} x86sim records identical to the native bench (time word excluded)")
    for img, n in diff[:10]:
        print(f"  image {img}: {n} word(s) differ")


if __name__ == "__main__":
    main()
