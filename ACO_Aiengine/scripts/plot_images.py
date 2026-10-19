"""
plot_images.py  —  per-image detection figure and convergence curve.

For every image of every evaluated algorithm of one project, as in the
reference ACO/pythonCode/scriptCode/helpers/perImageFigures.py:
  results/<Variant>/figures/id_<n>_detection.png    frame with the detected
        ellipse (cyan) over the ground truth (red, dashed); IoU, status and
        the AIE-code detection time in the title
  results/<Variant>/figures/id_<n>_convergence.png  best-so-far fitness after
        every iteration of that detection
Sources: results/<Variant>/table/<Variant>_results.csv and
results/<Variant>/convergence.csv (scripts/evaluate.py), data/annotations.json
and the PNG frames. The table may store the ellipses in any scale; they are
mapped back to image pixels with the annotation of the same image.
The same script serves ACO_Aiengine, GA_AIE and PSO_AI_Engine (--project).

Needs matplotlib (~/.venvs/aie_plots, see plot_timing.py). Resumable: existing
figures are skipped unless --force.

Usage:
    .../python scripts/plot_images.py --project . --images ../ACO/data/Ellipses
"""
import argparse
import csv
import json
import math
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

INVALID = -1.0e29
REFINE_MARK = {"FPED": 50}     # iteration of FPED's progressive refinement


def ellipse_xy(cx, cy, a, b, th, n=300):
    xs, ys = [], []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        xs.append(cx + a * math.cos(t) * math.cos(th) - b * math.sin(t) * math.sin(th))
        ys.append(cy + a * math.cos(t) * math.sin(th) + b * math.sin(t) * math.cos(th))
    return xs, ys


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def job(task):
    variant, r, hist, gt_raw, img_path, out_dir, force, family = task
    img_id = int(r["imageID"])
    det_png = out_dir / f"id_{img_id}_detection.png"
    conv_png = out_dir / f"id_{img_id}_convergence.png"
    if det_png.exists() and conv_png.exists() and not force:
        return
    img = mpimg.imread(str(img_path))
    h, w = img.shape[0], img.shape[1]

    # table units -> image pixels, from the annotation of this image
    # (annotations.json is in the x2 evaluation frame of the dataset)
    gx_t, ga_t = num(r.get("gt_cx")), num(r.get("gt_a"))
    pix = 0.5 * gt_raw[0] / gx_t if gx_t else (0.5 * gt_raw[2] / ga_t if ga_t else 0.5)
    gt = [gt_raw[0] * 0.5, gt_raw[1] * 0.5, gt_raw[2] * 0.5, gt_raw[3] * 0.5, gt_raw[4]]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, origin="upper", interpolation="nearest")
    xs, ys = ellipse_xy(*gt)
    ax.plot(xs, ys, "r--", lw=1.6, label="Ground truth")
    ax.plot(gt[0], gt[1], "r+", ms=9, mew=1.8)
    det = [num(r.get(k)) for k in ("det_cx", "det_cy", "det_a", "det_b", "det_theta")]
    if all(v is not None for v in det):
        d = [det[0] * pix, det[1] * pix, det[2] * pix, det[3] * pix, det[4]]
        xs, ys = ellipse_xy(*d)
        ax.plot(xs, ys, "c-", lw=1.6, label="Detected")
        ax.plot(d[0], d[1], "c+", ms=9, mew=1.8)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    t = num(r.get("time_ms_aie_parallel"))
    ttxt = f"  {t:.3f} ms" if t else ""
    ax.set_title(f"{variant} (AIE code)  frame {img_id}\nIoU = {num(r['IoU']) or 0:.4f}  [{r['Status']}]{ttxt}",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.7)
    fig.savefig(det_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.1))
    its = [i + 1 for i, v in enumerate(hist) if v is not None]
    vals = [v for v in hist if v is not None]
    ax.plot(its, vals, color="#2E6FB7", lw=1.5)
    if variant in REFINE_MARK:
        ax.axvline(REFINE_MARK[variant], color="#7E57A5", ls=":", lw=1.1, label="trail refinement")
        ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(1, max(len(hist), 2))
    ax.set_xlabel(f"{family} iteration")
    ax.set_ylabel("Best-so-far fitness")
    ax.set_title(f"{variant} (AIE code) convergence, frame {img_id}", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(conv_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--images", type=Path, required=True, help="folder with id_<n>.png")
    ap.add_argument("--variants", nargs="*", help="default: every variant in results/summary.csv")
    ap.add_argument("--iteration-name", default="Colony", help="x-axis label prefix (Colony / Generation / PSO)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    args = ap.parse_args()
    proj = args.project.resolve()

    gts = {r["image_id"]: (r["cx"], r["cy"], r["a"], r["b"], r["theta"])
           for r in json.loads((proj / "data" / "annotations.json").read_text())["annotations"]}
    variants = args.variants
    if not variants:
        with (proj / "results" / "summary.csv").open() as f:
            variants = [r["variant"] for r in csv.DictReader(f)]

    tasks = []
    for v in variants:
        with (proj / "results" / v / "table" / f"{v}_results.csv").open() as f:
            rows = list(csv.DictReader(f))
        if args.limit:
            rows = rows[:args.limit]
        hists = {}
        conv = proj / "results" / v / "convergence.csv"
        if conv.exists():
            with conv.open() as f:
                for r in csv.DictReader(f):
                    hv = [num(r[k]) for k in r if k != "imageID"]
                    hists[int(r["imageID"])] = [x if (x is not None and x > INVALID) else None for x in hv]
        out_dir = proj / "results" / v / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            i = int(r["imageID"])
            tasks.append((v, r, hists.get(i, []), gts[i], args.images / f"id_{i}.png", out_dir,
                          args.force, args.iteration_name))
    with Pool(args.jobs) as p:
        for n, _ in enumerate(p.imap_unordered(job, tasks, chunksize=8), 1):
            if n % 500 == 0:
                print(f"  {n}/{len(tasks)} images", flush=True)
    print(f"{len(tasks)} image(s): detection + convergence figures in results/<Variant>/figures/")


if __name__ == "__main__":
    main()
