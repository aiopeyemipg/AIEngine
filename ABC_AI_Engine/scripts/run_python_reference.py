"""
run_python_reference.py  —  run the ORIGINAL Python ABC scripts (../ABC)
on this machine, unchanged, to validate the C++ port and to time Python.

Loads hollowness-aware-ABC.py or standard-residual-ABC.py as a module,
points its paths at the local dataset, seeds `random` with RUN_SEED (1,
as in the scripts) and calls its own process_all_images(). Writes
  results/timing/<HA|SR>_python.csv
    imageID, detected, cx, cy, a, b, theta (pixels), fitness,
    time_ms (the script's own per-image timer), python_iou, python_pass
Needs numpy, OpenCV, Pillow, matplotlib (~/.venvs/aie_plots). The
module `long_side_form` the scripts import is not in ../ABC; a stand-in
is in scripts/python_shim/.

Usage: .../python scripts/run_python_reference.py --variant HA [--images N]
"""
import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = {"HA": "hollowness-aware-ABC.py", "SR": "standard-residual-ABC.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["HA", "SR"], required=True)
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--abc-dir", type=Path, default=PROJECT.parent / "ABC")
    ap.add_argument("--data", type=Path, default=PROJECT.parent / "ACO" / "data")
    args = ap.parse_args()

    sys.path.insert(0, str(PROJECT / "scripts" / "python_shim"))
    spec = importlib.util.spec_from_file_location("abc_script", args.abc_dir / SCRIPTS[args.variant])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MAKE_PLOTS = False

    gt = mod.load_ground_truth(str(args.data / "annotations.json"))
    gt = {k: gt[k] for k in sorted(gt)[:args.images]}
    random.seed(mod.RUN_SEED)
    results = mod.process_all_images(gt, str(args.data / "Ellipses"))

    out = PROJECT / "results" / "timing"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.variant}_python.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["imageID", "detected", "cx", "cy", "a", "b", "theta", "fitness", "time_ms",
                    "python_iou", "python_pass"])
        for r in results:
            d = r["det_px"]
            w.writerow([r["image_id"], 1, *[f"{v:.9g}" for v in d], f"{r['conv_hist'][-1]:.9g}",
                        f"{r['time_ms']:.6f}", f"{r['iou']:.6f}", int(r["passed"])])
    n_pass = sum(r["passed"] for r in results)
    print(f"{args.variant}-ABC (original Python): {n_pass}/{len(results)} pass, "
          f"{sum(r['time_ms'] for r in results) / len(results):.2f} ms/image")


if __name__ == "__main__":
    main()
