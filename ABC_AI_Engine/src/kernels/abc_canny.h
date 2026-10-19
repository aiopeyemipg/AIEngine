#ifndef ABC_CANNY_H
#define ABC_CANNY_H

// ============================================================
// abc_canny.h  —  edge points of SR-ABC, integer only
//
// Re-implements the pre-processing of standard-residual-ABC.py
// (extract_points) without OpenCV:
//   cv2.resize(img, 80x80, INTER_NEAREST)   4x pixel replication
//   cv2.GaussianBlur(3x3, sigma 0)          [1 2 1]^2 / 16, rounded,
//                                           BORDER_REFLECT_101
//   cv2.Canny(30, 120)                      Sobel 3x3 (BORDER_REPLICATE),
//                                           L1 magnitude, OpenCV's
//                                           non-maximum suppression with
//                                           the tan(22.5) fixed-point test,
//                                           8-connected hysteresis
//                                           (sweeps instead of a stack:
//                                           same edge set, less memory)
//   points = (x / 4, y / 4) of every edge pixel, row-major
//   fewer than 5 edge points -> every non-zero pixel of the 20x20 image
// scripts/check_canny.py compares the result with OpenCV for all frames.
//
// Plain C++ (no AIE API): used by the AIE master tile, the native
// bench and the reference C++ port. The caller supplies the work
// buffers (blur, mag, dir; CannyWork natively), so the AIE graph can
// place them in neighbouring tile memories (graph-side parameters).
// ============================================================

#include <stdint.h>
#include "include.h"

#define CANNY_UP     4                        // upscale factor
#define CANNY_W      (IMG_W * CANNY_UP)       // 80
#define CANNY_H      (IMG_H * CANNY_UP)
#define CANNY_PIX    (CANNY_W * CANNY_H)      // 6400
#define CANNY_LOW    30
#define CANNY_HIGH   120
#define CANNY_SHIFT  15
#define CANNY_TG22   13573                    // round(tan(22.5 deg) * 2^15)

struct CannyWork {
    uint8_t blur[CANNY_PIX];                  // blurred upscaled image
    int16_t mag[CANNY_PIX];                   // |dx| + |dy|
    int8_t  dir[CANNY_PIX];                   // NMS result: 0 none, 1 weak, 3 edge
};

static inline int canny_reflect101(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}
static inline int canny_clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// upscaled source pixel (nearest neighbour)
static inline int canny_src(const uint8_t* img20, int x, int y) {
    return img20[(y / CANNY_UP) * IMG_W + (x / CANNY_UP)];
}

// ------------------------------------------------------------
// Edge points of one 20x20 frame. Writes px/py (image pixels) of at most
// maxPts points, returns the number found (may exceed maxPts: flagged
// by the caller) and sets *fallback when the < 5 rule applied.
// ------------------------------------------------------------
static inline int canny_points(const uint8_t* img20, uint8_t* blur, int16_t* mag, int8_t* dir,
                               float* px, float* py, int maxPts, int* fallback) {
    // 1. nearest-neighbour upscale + 3x3 Gaussian [1 2 1]^2 / 16 (reflect-101)
    for (int y = 0; y < CANNY_H; ++y)
        for (int x = 0; x < CANNY_W; ++x) {
            int s = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = canny_reflect101(y + dy, CANNY_H);
                const int ky = (dy == 0) ? 2 : 1;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = canny_reflect101(x + dx, CANNY_W);
                    const int kx = (dx == 0) ? 2 : 1;
                    s += ky * kx * canny_src(img20, xx, yy);
                }
            }
            blur[y * CANNY_W + x] = (uint8_t)((s + 8) >> 4);
        }

    // 2. Sobel 3x3 (replicate) and L1 magnitude
    #define CB(x, y) ((int)blur[canny_clampi(y, CANNY_H) * CANNY_W + canny_clampi(x, CANNY_W)])
    for (int y = 0; y < CANNY_H; ++y)
        for (int x = 0; x < CANNY_W; ++x) {
            const int gx = (CB(x + 1, y - 1) + 2 * CB(x + 1, y) + CB(x + 1, y + 1))
                         - (CB(x - 1, y - 1) + 2 * CB(x - 1, y) + CB(x - 1, y + 1));
            const int gy = (CB(x - 1, y + 1) + 2 * CB(x, y + 1) + CB(x + 1, y + 1))
                         - (CB(x - 1, y - 1) + 2 * CB(x, y - 1) + CB(x + 1, y - 1));
            mag[y * CANNY_W + x] = (int16_t)((gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy));
            dir[y * CANNY_W + x] = 0;
        }

    // 3. non-maximum suppression (OpenCV canny.cpp, L1 path); the
    //    magnitude outside the image counts as 0
    #define CM(x, y) (((x) < 0 || (x) >= CANNY_W || (y) < 0 || (y) >= CANNY_H) ? 0 : (int)mag[(y) * CANNY_W + (x)])
    for (int y = 0; y < CANNY_H; ++y)
        for (int x = 0; x < CANNY_W; ++x) {
            const int m = CM(x, y);
            if (m <= CANNY_LOW) continue;
            const int dxv = (CB(x + 1, y - 1) + 2 * CB(x + 1, y) + CB(x + 1, y + 1))
                          - (CB(x - 1, y - 1) + 2 * CB(x - 1, y) + CB(x - 1, y + 1));
            const int dyv = (CB(x - 1, y + 1) + 2 * CB(x, y + 1) + CB(x + 1, y + 1))
                          - (CB(x - 1, y - 1) + 2 * CB(x, y - 1) + CB(x + 1, y - 1));
            const int ax = dxv < 0 ? -dxv : dxv;
            const int ay = (dyv < 0 ? -dyv : dyv) << CANNY_SHIFT;
            const int tg22x = ax * CANNY_TG22;
            bool keep;
            if (ay < tg22x) {
                keep = m > CM(x - 1, y) && m >= CM(x + 1, y);
            } else {
                const int tg67x = tg22x + (ax << (CANNY_SHIFT + 1));
                if (ay > tg67x) {
                    keep = m > CM(x, y - 1) && m >= CM(x, y + 1);
                } else {
                    const int s = ((dxv ^ dyv) < 0) ? -1 : 1;
                    keep = m > CM(x - s, y - 1) && m > CM(x + s, y + 1);
                }
            }
            if (!keep) continue;
            dir[y * CANNY_W + x] = (m > CANNY_HIGH) ? 3 : 1;   // edge / weak
        }
    #undef CM
    #undef CB

    // 4. hysteresis: a weak pixel 8-connected to an edge becomes an edge;
    //    forward and backward sweeps until nothing changes
    for (bool changed = true; changed; ) {
        changed = false;
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < CANNY_PIX; ++i) {
                const int p = pass == 0 ? i : CANNY_PIX - 1 - i;
                if (dir[p] != 1) continue;
                const int x = p % CANNY_W, y = p / CANNY_W;
                bool linked = false;
                for (int dy = -1; dy <= 1 && !linked; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = x + dx, yy = y + dy;
                        if (xx < 0 || xx >= CANNY_W || yy < 0 || yy >= CANNY_H) continue;
                        if (dir[yy * CANNY_W + xx] == 3) { linked = true; break; }
                    }
                if (linked) { dir[p] = 3; changed = true; }
            }
    }

    // 5. edge pixels -> points in the 20x20 frame (row-major, as np.where)
    int n = 0;
    for (int p = 0; p < CANNY_PIX; ++p)
        if (dir[p] == 3) {
            if (n < maxPts) {
                px[n] = (float)(p % CANNY_W) / (float)CANNY_UP;
                py[n] = (float)(p / CANNY_W) / (float)CANNY_UP;
            }
            ++n;
        }
    *fallback = 0;
    if (n < 5) {                       // too few edges: every lit pixel instead
        *fallback = 1;
        n = 0;
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x)
                if (img20[y * IMG_W + x] > 0) {
                    if (n < maxPts) { px[n] = (float)x; py[n] = (float)y; }
                    ++n;
                }
    }
    return n;
}

#endif
