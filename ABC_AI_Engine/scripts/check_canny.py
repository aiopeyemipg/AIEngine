"""
check_canny.py  —  compare the integer Canny of src/kernels/abc_canny.h with
OpenCV, exactly as standard-residual-ABC.py calls it (resize x4 nearest,
GaussianBlur 3x3, Canny 30/120, < 5 edges -> lit pixels).

Input: the C++ edge points of every frame, one line per frame:
    <image> <fallback> <n> x,y x,y ...      (build/check/canny_cpp.txt)
Needs numpy, OpenCV and Pillow (~/.venvs/aie_plots).

Usage: python scripts/check_canny.py build/check/canny_cpp.txt ../ACO/data/Ellipses
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def python_points(path):
    img = np.array(Image.open(path).convert("L"))
    big = cv2.resize(img, (80, 80), interpolation=cv2.INTER_NEAREST)
    edges = cv2.Canny(cv2.GaussianBlur(big, (3, 3), 0), 30, 120)
    y, x = np.where(edges > 0)
    pts = list(zip(x / 4, y / 4))
    if len(pts) < 5:
        y, x = np.where(img > 0)
        pts = list(zip(x.astype(float), y.astype(float)))
    return pts


def main():
    cpp_file, img_dir = Path(sys.argv[1]), Path(sys.argv[2])
    same = diff = 0
    worst = []
    for line in cpp_file.read_text().splitlines():
        tok = line.split()
        i = int(tok[0])
        cpp = [tuple(float(v) for v in t.split(",")) for t in tok[3:]]
        py = [(float(a), float(b)) for a, b in python_points(img_dir / f"id_{i}.png")]
        if cpp == py:
            same += 1
        else:
            diff += 1
            sc, sp = set(cpp), set(py)
            worst.append((len(sc ^ sp), i, len(cpp), len(py)))
    print(f"frames with identical edge points: {same}, different: {diff}")
    for d, i, nc, npy in sorted(worst, reverse=True)[:10]:
        print(f"  frame {i}: C++ {nc} points, OpenCV {npy} points, {d} differ")


if __name__ == "__main__":
    main()
