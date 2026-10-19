"""
evaluate.py  —  score the ACO AIE detections and tabulate the time per image.

Detections: the result records of one algorithm (13 words + the
convergence history per image, layout in src/kernels/include.h), read from
  * the AIE graph output   build/x86sim_<Variant>/x86simulator_output/data/aco_results.txt
    (default when it exists), or
  * the native bench       results/timing/<Variant>_parallel.txt  (--source bench)
Both run the same detection code and give bit-identical detections.

Evaluation exactly as ACO/cppCode/scriptCode/Evaluation.h: detected ellipse
x2 vs ground truth x1, a >= b, theta in [0, pi); per-parameter errors; IoU of
the filled ellipses; PASS if IoU > 0.7; an undetected image is a failure with
IoU 0 and stays in the denominator.

Time per image (ms; hit extraction + seed + colony run, no I/O), merged from
results/timing/ when present:
  time_ms_reference_cpp     original C++ (ACO/cppCode, double, one thread)
  time_ms_aie_serial        AIE detection code, natively, one thread
  time_ms_aie_parallel      AIE detection code, natively, T threads inside
                            every colony iteration (trails, then ants)
  speedup_vs_reference      time_ms_reference_cpp / time_ms_aie_parallel
  time_ms_aie_x86sim        packet time word of the AIE graph run (x86sim =
                            host wall clock of a functional model; on hardware
                            = tile cycle counter / AIE_CLOCK_MHZ)

Writes
  results/<Variant>/table/<Variant>_results.csv     one row per image
  results/<Variant>/convergence.csv                 best-so-far fitness after every
                                                    colony iteration, one row per image
  results/summary.csv                               one row per algorithm
                                                    (re-written for every evaluated variant)

Usage (from the ACO_Aiengine folder):
    python3 scripts/evaluate.py --variant FPED
    python3 scripts/evaluate.py --variant StdACO --source bench
"""
import argparse
import csv
import json
import math
import struct
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
IOU_THRESHOLD = 0.7
HIST_LEN = 100                 # ACO_HIST_LEN
RESULT_WORDS = 13 + HIST_LEN
INVALID = -1.0e29              # below: no valid candidate yet
VARIANTS = ["FPED", "StdACO"]   # variant index vi


def as_float(word):
    return struct.unpack("<f", struct.pack("<i", word))[0]


def as_int32(v):
    """PLIO files print 32-bit words unsigned; reinterpret as int32."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def read_words(path):
    """Data words of a PLIO text file ('TLAST' markers and aiesimulator
    'T <time>' lines skipped)."""
    words = []
    for line in path.read_text().split("\n"):
        tok = line.strip()
        if not tok or tok == "TLAST" or tok.startswith("T "):
            continue
        words.extend(as_int32(int(v)) for v in tok.split())
    return words


def read_results(path, packet_header):
    """One record per image. The bench writes packets (header word +
    RESULT_WORDS words); the AIE graph writes plain RESULT_WORDS records."""
    words = read_words(path)
    step = RESULT_WORDS + (1 if packet_header else 0)
    out = {}
    for i in range(0, len(words) - step + 1, step):
        w = words[i + (1 if packet_header else 0): i + step]
        r = {"vi": w[0], "image": w[1], "detected": w[2] == 1,
             "ell": tuple(as_float(x) for x in w[3:8]), "fitness": as_float(w[8]),
             "evals": w[9], "iters": w[10], "flags": w[11], "time_ms": as_float(w[12]) / 1000.0,
             "hist": [as_float(x) for x in w[13:13 + HIST_LEN]]}
        if r["image"] >= 0:
            out[r["image"]] = r
    return out


def read_reference(path):
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["imageID"])] = {
                "detected": r["detected"] == "1",
                "ell": tuple(float(r[k]) for k in ("cx", "cy", "a", "b", "theta")),
                "time_ms": float(r["time_ms"])}
    return out


def cround(x):
    """std::round: half away from zero."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def normalize(e, scale):
    cx, cy, a, b, th = e[0] * scale, e[1] * scale, e[2] * scale, e[3] * scale, e[4]
    if b > a:
        a, b = b, a
        th += math.pi / 2.0
    th = math.fmod(th, math.pi)
    if th < 0:
        th += math.pi
    return (cx, cy, a, b, th)


def ang_err(t1, t2):
    d = math.fmod(abs(t1 - t2), math.pi)
    return min(d, math.pi - d)


def inside_mask(e, canvas):
    xc, yc, a, b = cround(e[0]), cround(e[1]), cround(e[2]), cround(e[3])
    if a < 1 or b < 1:
        return set()
    ct, st = math.cos(e[4]), math.sin(e[4])
    pts = set()
    for y in range(canvas):
        dy = y - yc
        for x in range(canvas):
            dx = x - xc
            u = (dx * ct + dy * st) / a
            v = (dx * st - dy * ct) / b
            if u * u + v * v <= 1.0:
                pts.add((x, y))
    return pts


def iou(det, gt):
    """IoU of the filled ellipses in the evaluation space (Evaluation.h computeIoU)."""
    canvas = int(max(max(det[0] + det[2], gt[0] + gt[2]), max(det[1] + det[2], gt[1] + gt[2])) + 10)
    md, mg = inside_mask(det, canvas), inside_mask(gt, canvas)
    union = len(md | mg)
    return len(md & mg) / union if union else 0.0


def score_all(results, gts, n_images):
    """Per-image evaluation rows and (pass, mean IoU over all images)."""
    rows, n_pass, iou_sum = {}, 0, 0.0
    for image_id in range(n_images):
        r = results.get(image_id)
        gt = normalize(gts[image_id], 1.0)
        if r is None or not r["detected"]:
            rows[image_id] = None
            continue
        det = normalize(r["ell"], 2.0)
        s = iou(det, gt)
        errs = [abs(det[0] - gt[0]), abs(det[1] - gt[1]), abs(det[2] - gt[2]),
                abs(det[3] - gt[3]), ang_err(det[4], gt[4])]
        passed = s > IOU_THRESHOLD
        n_pass += passed
        iou_sum += s
        rows[image_id] = (det, gt, s, errs, passed)
    return rows, n_pass, iou_sum / n_images


def fmt(v):
    return "" if v is None else f"{v:.6g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--source", choices=["auto", "x86sim", "bench"], default="auto")
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--annotations", type=Path, default=PROJECT / "data" / "annotations.json")
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    args = ap.parse_args()
    name, n_images = args.variant, args.images
    timing = args.results / "timing"

    gts = {r["image_id"]: (r["cx"], r["cy"], r["a"], r["b"], r["theta"])
           for r in json.loads(args.annotations.read_text())["annotations"]}

    sim_file = PROJECT / "build" / f"x86sim_{name}" / "x86simulator_output" / "data" / "aco_results.txt"
    bench_file = timing / f"{name}_parallel.txt"
    if args.source == "x86sim" or (args.source == "auto" and sim_file.exists()):
        source, results = "AIE graph (x86sim)", read_results(sim_file, packet_header=False)
    else:
        source, results = "native bench (AIE code; threads)", read_results(bench_file, packet_header=True)
    missing = [i for i in range(n_images) if i not in results]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} image(s) without a result in {source}, e.g. {missing[:5]}")

    def times(path, header):
        return {k: v["time_ms"] for k, v in read_results(path, header).items()} if path.exists() else {}

    t_ser = times(timing / f"{name}_serial.txt", True)
    t_par = times(timing / f"{name}_parallel.txt", True)
    ref_path = timing / f"{name}_reference.csv"
    ref = read_reference(ref_path) if ref_path.exists() else {}
    t_ref = {k: v["time_ms"] for k, v in ref.items()}
    t_sim = {k: v["time_ms"] for k, v in results.items()} if source.startswith("AIE") else {}

    rows, n_pass, miou = score_all(results, gts, n_images)
    table_dir = args.results / name / "table"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / f"{name}_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["imageID", "IoU", "Status",
                    "det_cx", "det_cy", "det_a", "det_b", "det_theta",
                    "gt_cx", "gt_cy", "gt_a", "gt_b", "gt_theta",
                    "err_cx", "err_cy", "err_a", "err_b", "err_theta",
                    "fitness", "evaluations", "iterations",
                    "time_ms_reference_cpp", "time_ms_aie_serial", "time_ms_aie_parallel",
                    "speedup_vs_reference", "time_ms_aie_x86sim"])
        for i in range(n_images):
            r, row = results[i], rows[i]
            tr, ts, tp = t_ref.get(i), t_ser.get(i), t_par.get(i)
            sp = tr / tp if (tr is not None and tp) else None
            tail = [fmt(r["fitness"]), r["evals"], r["iters"], fmt(tr), fmt(ts), fmt(tp), fmt(sp), fmt(t_sim.get(i))]
            if row is None:
                gt = normalize(gts[i], 1.0)
                w.writerow([i, 0, "UNDETECTED", *[""] * 5, *[fmt(v) for v in gt], *[""] * 5, *tail])
            else:
                det, gt, s, errs, passed = row
                w.writerow([i, f"{s:.6g}", "PASS" if passed else "FAIL", *[fmt(v) for v in det],
                            *[fmt(v) for v in gt], *[fmt(v) for v in errs], *tail])

    with (args.results / name / "convergence.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["imageID"] + [f"it{t + 1}" for t in range(HIST_LEN)])
        for i in range(n_images):
            w.writerow([i] + [f"{v:.6g}" if v > INVALID else "" for v in results[i]["hist"]])

    def mean(d):
        vals = [d[i] for i in range(n_images) if i in d]
        return sum(vals) / len(vals) if len(vals) == n_images else None

    ref_pass = ref_miou = None
    if ref:
        _, ref_pass, ref_miou = score_all(ref, gts, n_images)
    m_ref, m_ser, m_par = mean(t_ref), mean(t_ser), mean(t_par)
    threads = ""
    tf = timing / f"{name}_threads.txt"
    if tf.exists():
        threads = tf.read_text().strip()

    summary_path = args.results / "summary.csv"
    header = ["variant", "images", "source", "pass", "accuracy_pct", "mean_iou_all",
              "ref_cpp_pass", "ref_cpp_accuracy_pct", "ref_cpp_mean_iou_all",
              "mean_ms_reference_cpp", "mean_ms_aie_serial", "mean_ms_aie_parallel", "threads",
              "speedup_parallel_vs_reference", "speedup_parallel_vs_serial",
              "total_s_reference_cpp", "total_s_aie_parallel"]
    old = {}
    if summary_path.exists():
        with summary_path.open() as f:
            old = {r["variant"]: r for r in csv.DictReader(f)}
    old[name] = dict(zip(header, [
        name, n_images, source, n_pass, f"{100.0 * n_pass / n_images:.2f}", f"{miou:.6f}",
        fmt(ref_pass), fmt(None if ref_pass is None else 100.0 * ref_pass / n_images), fmt(ref_miou),
        fmt(m_ref), fmt(m_ser), fmt(m_par), threads,
        fmt(m_ref / m_par if (m_ref and m_par) else None),
        fmt(m_ser / m_par if (m_ser and m_par) else None),
        fmt(m_ref * n_images / 1000 if m_ref else None), fmt(m_par * n_images / 1000 if m_par else None)]))
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for v in VARIANTS:
            if v in old:
                w.writerow(old[v])

    print(f"{name}: detections from {source}")
    print(f"  accuracy {100.0 * n_pass / n_images:.2f}% ({n_pass}/{n_images})  mean IoU (all) {miou:.4f}")
    if ref_pass is not None:
        print(f"  reference C++ on this machine: {100.0 * ref_pass / n_images:.2f}%  mean IoU {ref_miou:.4f}")
    if m_ref and m_ser and m_par:
        print(f"  ms/image: reference C++ {m_ref:.4f} | AIE code serial {m_ser:.4f} | "
              f"AIE code parallel ({threads} threads) {m_par:.4f}  -> {m_ref / m_par:.2f}x vs reference")
    print(f"  table: {table_dir / f'{name}_results.csv'}")


if __name__ == "__main__":
    main()
