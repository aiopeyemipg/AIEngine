"""
plot_timing.py  —  time per image: original C++ vs the AI Engine code.

Reads results/summary.csv and results/<Variant>/table/<Variant>_results.csv
(written by scripts/evaluate.py) of one project and draws
  report_figures/timing_mean.png        mean ms per image: reference C++,
                                         AIE code serial, AIE code parallel
                                         (with the speed-up over the C++)
  report_figures/timing_per_image.png   the per-image times of every
                                         algorithm (box plots, log scale)
  report_figures/accuracy.png           accuracy of the AIE code next to the
                                         reference C++ (same images)
The same script serves ACO_Aiengine, GA_AIE and PSO_AI_Engine (--project).

Needs matplotlib, e.g. the environment in ~/.venvs/aie_plots:
    env -u PYTHONPATH -u PYTHONHOME \\
        LD_LIBRARY_PATH=/tools/Xilinx/Vitis/2024.2/tps/lnx64/python-3.8.3/lib \\
        ~/.venvs/aie_plots/bin/python scripts/plot_timing.py --project .
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [("time_ms_reference_cpp", "original C++ (1 thread)", "#8c8c8c"),
          ("time_ms_aie_serial", "AIE code, serial (1 thread)", "#6baed6"),
          ("time_ms_aie_parallel", "AIE code, parallel iterations", "#08519c")]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--title", default=None, help="algorithm family shown in the titles")
    args = ap.parse_args()
    proj = args.project.resolve()
    title = args.title or proj.name
    summary = read_csv(proj / "results" / "summary.csv")
    variants = [r["variant"] for r in summary]
    out = proj / "report_figures"
    out.mkdir(parents=True, exist_ok=True)

    # ---- mean time per image ----
    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(variants) + 2), 4.6))
    width = 0.26
    keys = ["mean_ms_reference_cpp", "mean_ms_aie_serial", "mean_ms_aie_parallel"]
    for k, (key, (_, label, colour)) in enumerate(zip(keys, SERIES)):
        vals = [num(r.get(key)) or 0.0 for r in summary]
        xs = [i + (k - 1) * width for i in range(len(variants))]
        bars = ax.bar(xs, vals, width, label=label, color=colour)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 2), textcoords="offset points")
    for i, r in enumerate(summary):
        sp, th = num(r.get("speedup_parallel_vs_reference")), r.get("threads", "")
        if sp:
            top = max(num(r.get(k)) or 0 for k in keys)
            ax.annotate(f"{sp:.2f}x faster than C++\n({th} threads)", (i, top), ha="center", va="bottom",
                        fontsize=9, xytext=(0, 14), textcoords="offset points", color="#08519c")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("mean detection time per image (ms)")
    ax.set_title(f"{title}: time per image, original C++ vs AI Engine code")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "timing_mean.png", dpi=150)
    plt.close(fig)

    # ---- per-image distribution ----
    fig, ax = plt.subplots(figsize=(max(6.5, 2.4 * len(variants) + 2), 4.6))
    data, pos, cols = [], [], []
    for i, v in enumerate(variants):
        rows = read_csv(proj / "results" / v / "table" / f"{v}_results.csv")
        for k, (col, _, colour) in enumerate(SERIES):
            vals = [num(r.get(col)) for r in rows]
            vals = [x for x in vals if x]
            if vals:
                data.append(vals)
                pos.append(i + (k - 1) * 0.27)
                cols.append(colour)
    bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    for med in bp["medians"]:
        med.set_color("white")
    ax.set_yscale("log")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("detection time per image (ms, log scale)")
    ax.set_title(f"{title}: per-image time ({len(data[0]) if data else 0} images per box)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in SERIES], fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "timing_per_image.png", dpi=150)
    plt.close(fig)

    # ---- accuracy ----
    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(variants) + 2), 4.2))
    aie = [num(r.get("accuracy_pct")) or 0 for r in summary]
    ref = [num(r.get("ref_cpp_accuracy_pct")) or 0 for r in summary]
    xs = range(len(variants))
    b1 = ax.bar([x - 0.2 for x in xs], ref, 0.4, label="original C++", color="#8c8c8c")
    b2 = ax.bar([x + 0.2 for x in xs], aie, 0.4, label="AI Engine code", color="#08519c")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(variants)
    ax.set_ylabel("accuracy (% of images, IoU pass)")
    ax.set_ylim(0, 110)
    ax.set_title(f"{title}: accuracy, original C++ vs AI Engine code")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=150)
    plt.close(fig)
    print(f"figures -> {out}/timing_mean.png, timing_per_image.png, accuracy.png")


if __name__ == "__main__":
    main()
