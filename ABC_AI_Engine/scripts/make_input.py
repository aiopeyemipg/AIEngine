"""
make_input.py  —  build the PLIO input file for the ABC AIE graph.

The graph detects one image per graph iteration; inside that iteration
every ABC iteration works on its 30 bees in parallel on the worker tiles. This script writes the NUM_IMAGES image records (NUM_IMAGES
from src/kernels/include.h) into

    data/abc_input.txt

one 416-byte record per image, 4 bytes per line (one 32-bit PLIO beat):

    16-byte header : image id (2 bytes, little endian), valid flag, 13 zeros
    400 bytes      : 20x20 grayscale pixels, row-major

Real gray levels are kept: HA-ABC weights every lit pixel by its
intensity, SR-ABC runs Canny on the frame. The native bench (bench/abc_bench.cpp)
reads the same file, so both see exactly the same bytes.

Only the Python standard library is needed (PNG decoded with zlib).

Usage (from the ABC_AI_Engine folder):
    python3 scripts/make_input.py
    python3 scripts/make_input.py --images /path/to/Ellipses
"""
import argparse
import re
import struct
import zlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = PROJECT.parent / "ACO" / "data" / "Ellipses"   # the same 1000 frames as ../ABC
INPUT_FILE = "abc_input.txt"
IMG_W = IMG_H = 20


def read_defines(path):
    """Integer #defines of include.h, with simple expressions resolved."""
    raw = {}
    for m in re.finditer(r"^[ \t]*#define[ \t]+(\w+)[ \t]+(.+?)[ \t]*(?://.*)?$", path.read_text(), re.M):
        raw.setdefault(m.group(1), m.group(2))
    vals = {}

    def ev(name):
        if name not in vals:
            expr = re.sub(r"\b([A-Za-z_]\w*)\b", lambda t: str(ev(t.group(1))), raw[name])
            vals[name] = int(eval(expr.replace("/", "//"), {}))
        return vals[name]

    for k in ("NUM_IMAGES", "IMG_HEADER_BYTES", "IMG_PIXELS"):
        ev(k)
    return vals


def read_png_gray(path):
    """Decode an 8-bit, non-interlaced PNG to a list of rows of gray values.
    Colour images are converted with the BGR2GRAY luma used by the C++ code."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    pos, idat = 8, b""
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            w, h, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            idat += chunk
    if depth != 8 or interlace != 0:
        raise ValueError(f"{path}: only 8-bit non-interlaced PNGs are supported")
    bpp = {0: 1, 2: 3, 4: 2, 6: 4}[color]

    raw = zlib.decompress(idat)
    stride = w * bpp
    prev = bytearray(stride)
    rows, p = [], 0
    for _ in range(h):
        filt = raw[p]
        line = bytearray(raw[p + 1:p + 1 + stride])
        p += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if filt == 1:
                line[x] = (line[x] + a) & 0xFF
            elif filt == 2:
                line[x] = (line[x] + b) & 0xFF
            elif filt == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 0xFF
            elif filt == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        rows.append(line)
        prev = line

    gray = []
    for line in rows:
        if color in (0, 4):
            gray.append([line[x * bpp] for x in range(w)])
        else:
            gray.append([int(round(0.299 * line[x * bpp] + 0.587 * line[x * bpp + 1]
                                   + 0.114 * line[x * bpp + 2])) for x in range(w)])
    return w, h, gray


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="folder with id_<n>.png")
    ap.add_argument("--include", type=Path, default=PROJECT / "src" / "kernels" / "include.h")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "data")
    args = ap.parse_args()

    d = read_defines(args.include)
    n_images, header = d["NUM_IMAGES"], d["IMG_HEADER_BYTES"]

    lines = []
    for image_id in range(n_images):
        path = args.images / f"id_{image_id}.png"
        w, h, gray = read_png_gray(path)
        if (w, h) != (IMG_W, IMG_H):
            raise SystemExit(f"{path}: size {w}x{h}, expected {IMG_W}x{IMG_H}")
        rec = [image_id & 0xFF, (image_id >> 8) & 0xFF, 1] + [0] * (header - 3)
        rec += [v for row in gray for v in row]
        for i in range(0, len(rec), 4):
            lines.append(" ".join(str(v) for v in rec[i:i + 4]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / INPUT_FILE).write_text("\n".join(lines) + "\n")
    print(f"{n_images} image(s) -> data/{INPUT_FILE} (one image per graph iteration)")


if __name__ == "__main__":
    main()
