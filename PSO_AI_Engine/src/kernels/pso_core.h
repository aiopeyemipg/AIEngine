
#ifndef PSO_CORE_H
#define PSO_CORE_H

// ============================================================
// pso_core.h  —  the three PSO detectors, one image per call, with the
// particles of every PSO iteration moved and scored in parallel
//
//   pso_load_image   image record -> EdgeSet, optional DBSCAN
//                    (GrayImage.h edges, Dbscan.h)
//   pso_fitness      hollowness-model fitness      (Fitness.h)
//   pso_fit_five     5-point algebraic ellipse fit (EllipseFitter.h)
//   pso_seed_swarm   5-point seeding + random fill (Seeding.h)
//   pso_shard_*      worker side: move + score the particles of a shard
//   pso_detect       master side: one image record -> PsoResult
//                    (PsoDetector.h, ApsoDetector.h, SpPsoDetector.h)
//
// Concurrency inside one image. The PSO iterations are sequential, but
// inside an iteration every particle is moved and scored on its own:
// particle i lives on worker i % W (its position, velocity and personal
// best stay there, GPU style). Per iteration the master sends the swarm
// leader (gbest; for SP-PSO the ring table of personal bests), the
// workers move + score their particles at the same time and return
// their best, and the master reduces them to the new gbest.
//   * AIE graph (pso_detectors.cc): W = PSO_WORKERS worker tiles.
//   * native bench (bench/pso_bench.cpp, -DPSO_NATIVE): W threads, or
//     W = 1 on one thread (serial baseline).
// Particle i draws its random numbers in iteration t from its own
// stream (seed, t, i), and every reduction keeps the first index on a
// tie, so the result does not depend on W: serial, threaded and AIE
// runs give the same detections.
//
// Difference from the reference: Standard PSO and APSO there update
// gbest inside the particle loop (asynchronous: particle i+1 already
// follows a gbest found by particle i of the same iteration). Here the
// gbest of iteration t is fixed while its particles move (synchronous
// PSO, the usual parallel form) and updated once after the iteration.
// SP-PSO is unchanged: its ring bests are already fixed per iteration.
//
// No ADF I/O here. An Engine provides
//   void begin(const EdgeSet& es, uint32_t seed);              new image
//   void init(const float (*pos)[5], int P, float* score);     score the initial swarm
//   void step(int it, int P, const float* gbest, PsoBest& b);  Std / APSO iteration
//   void step_sp(int it, const float (*tabP)[5], const float* tabS,
//                const uint8_t* reset, float (*newP)[5], float* newS);   SP-PSO iteration
//
// A particle / ellipse is a float[5] = {cx, cy, a, b, theta}.
// ============================================================

#ifndef PSO_NATIVE
#include <aie_api/aie.hpp>
#endif
#include "include.h"
#include "pso_types.h"
#include "pso_math.h"

// ------------------------------------------------------------
// Image record (header + pixels) -> EdgeSet. Returns the image id.
//   * every pixel with gray > 0 is an edge point, row-major order
//     (GrayImage::edgePoints)
//   * useDbscan: DBSCAN noise removal (Dbscan.h): keep core points
//     (>= MinPts neighbours within eps, the point itself included) and
//     points within eps of a core point; eps = 2, MinPts = 5
// Padding points sit far outside every candidate ellipse and carry
// zero weight, so they never contribute to the fitness.
// ------------------------------------------------------------
static inline int pso_load_image(const uint8_t* rec, bool useDbscan,
                                 EdgeSet& es, EdgeScratch& sc, int& flags) {
    const uint8_t* pix = rec + IMG_HEADER_BYTES;
    const int image = (int)rec[0] | ((int)rec[1] << 8);

    // collect edge pixels, packed x | y << 8 | gray << 16
    int n = 0;
    flags = 0;
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            const int g = pix[y * IMG_W + x];
            if (g == 0) continue;
            if (n == MAX_EDGES) { flags |= 1; continue; }
            sc.raw[n++] = (int32_t)(x | (y << 8) | (g << 16));
        }

    if (useDbscan) {
        for (int i = 0; i < n; ++i) {
            const int xi = sc.raw[i] & 0xFF, yi = (sc.raw[i] >> 8) & 0xFF;
            int cnt = 0;
            for (int j = 0; j < n; ++j) {
                const int dx = xi - (sc.raw[j] & 0xFF), dy = yi - ((sc.raw[j] >> 8) & 0xFF);
                if (dx * dx + dy * dy <= DBSCAN_EPS2) ++cnt;
            }
            sc.core[i] = (cnt >= DBSCAN_MIN_SAMPLES) ? 1 : 0;
        }
    }

    int kept = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t w = sc.raw[i];
        if (useDbscan) {
            bool keep = sc.core[i];
            const int xi = w & 0xFF, yi = (w >> 8) & 0xFF;
            for (int j = 0; j < n && !keep; ++j) {
                if (!sc.core[j]) continue;
                const int dx = xi - (sc.raw[j] & 0xFF), dy = yi - ((sc.raw[j] >> 8) & 0xFF);
                if (dx * dx + dy * dy <= DBSCAN_EPS2) keep = true;
            }
            if (!keep) continue;
        }
        const float inten = (float)((w >> 16) & 0xFF) * (1.0f / 255.0f);
        es.px[kept] = (float)(w & 0xFF);
        es.py[kept] = (float)((w >> 8) & 0xFF);
        es.boosted[kept] = inten * inten;                               // intensity^2
        es.bright[kept] = (inten >= HIGH_INTENSITY_THRESHOLD) ? 1.0f : 0.0f;
        ++kept;
    }

    es.n = kept;
    es.nPad = (kept + 7) & ~7;
    for (int i = kept; i < es.nPad; ++i) {
        es.px[i] = 1.0e4f;
        es.py[i] = 1.0e4f;
        es.boosted[i] = 0.0f;
        es.bright[i] = 0.0f;
    }
    return image;
}

// ------------------------------------------------------------
// Fitness (hollowness model). For every edge point:
//   |eq - 1| < tol  -> rim:      +intensity^2, bright rim points counted
//   eq < 1 - tol    -> interior: -penalty * intensity^2
// Invalid candidates (too small, leaving the image, or with a rim
// count outside [MIN, MAX]) score PSO_NEG_INF.
// The point loop runs on 8-lane float vectors.
// ------------------------------------------------------------
static inline float pso_fitness(const EdgeSet& es, const float* p) {
    const float h = p[0], k = p[1], a = p[2], b = p[3], theta = p[4];
    if (!(a >= 2.0f) || !(b >= 2.0f)) return PSO_NEG_INF;          // also rejects NaN
    if (!(pso_abs(theta) < 1.0e6f) || !(pso_abs(h) < 1.0e6f) || !(pso_abs(k) < 1.0e6f))
        return PSO_NEG_INF;

    float st, ct;
    pso_sincos(theta, st, ct);
    const float bboxHW = pso_sqrt((a * ct) * (a * ct) + (b * st) * (b * st));
    const float bboxHH = pso_sqrt((a * st) * (a * st) + (b * ct) * (b * ct));
    if (h - bboxHW < 0.0f || h + bboxHW > (float)IMG_W ||
        k - bboxHH < 0.0f || k + bboxHH > (float)IMG_H)
        return PSO_NEG_INF;

    const float inva2 = 1.0f / (a * a);
    const float invb2 = 1.0f / (b * b);

#if defined(PSO_NATIVE) || (defined(__X86SIM__) && !defined(PSO_X86SIM_VECTOR_FITNESS))
    // x86 simulation and native bench: plain scalar loop with the same
    // arithmetic. The AIE vector intrinsics are emulated bit-accurately by
    // x86sim, which is far too slow for a 1000-image run. Add
    // -DPSO_X86SIM_VECTOR_FITNESS to the x86sim build to simulate the vector
    // loop below instead.
    float perim = 0.0f, intern = 0.0f, brightCount = 0.0f;
    int numPerimeter = 0;
    for (int i = 0; i < es.n; ++i) {
        const float dx = es.px[i] - h, dy = es.py[i] - k;
        const float t1 = dx * ct + dy * st;
        const float t2 = dx * st - dy * ct;
        const float eq = t1 * t1 * inva2 + t2 * t2 * invb2;
        if (pso_abs(eq - 1.0f) < INLIER_TOLERANCE) {
            perim += es.boosted[i];
            brightCount += es.bright[i];
            ++numPerimeter;
        } else if (eq < 1.0f - INLIER_TOLERANCE) {
            intern += es.boosted[i];
        }
    }
    if (numPerimeter < MIN_INLIERS_FOR_DETECTION || numPerimeter > MAX_INLIERS_FOR_DETECTION)
        return PSO_NEG_INF;
    return perim - HOLLOWNESS_PENALTY * intern + BRIGHT_POINT_BONUS * brightCount;
#else
    const aie::vector<float, 8> zero = aie::zeros<float, 8>();
    const aie::vector<float, 8> one  = aie::broadcast<float, 8>(1.0f);
    const aie::vector<float, 8> tol  = aie::broadcast<float, 8>(INLIER_TOLERANCE);
    const aie::vector<float, 8> inLim = aie::broadcast<float, 8>(1.0f - INLIER_TOLERANCE);

    aie::vector<float, 8> perimV = zero, internV = zero, brightV = zero, countV = zero;

    for (int i = 0; i < es.nPad; i += 8)
        chess_prepare_for_pipelining
    {
        aie::vector<float, 8> X = aie::load_v<8>(es.px + i);
        aie::vector<float, 8> Y = aie::load_v<8>(es.py + i);
        aie::vector<float, 8> dx = aie::sub(X, h);
        aie::vector<float, 8> dy = aie::sub(Y, k);

        aie::vector<float, 8> t1 = aie::mac(aie::mul(dx, ct), dy, st).template to_vector<float>();
        aie::vector<float, 8> t2 = aie::msc(aie::mul(dx, st), dy, ct).template to_vector<float>();
        aie::vector<float, 8> t1s = aie::mul(t1, t1).template to_vector<float>();
        aie::vector<float, 8> t2s = aie::mul(t2, t2).template to_vector<float>();
        aie::vector<float, 8> eq = aie::mac(aie::mul(t1s, inva2), t2s, invb2).template to_vector<float>();

        aie::vector<float, 8> dev = aie::abs(aie::sub(eq, 1.0f));
        aie::mask<8> rim    = aie::lt(dev, tol);
        aie::mask<8> inside = aie::lt(eq, inLim);   // disjoint from rim

        aie::vector<float, 8> B = aie::load_v<8>(es.boosted + i);
        aie::vector<float, 8> R = aie::load_v<8>(es.bright + i);

        perimV  = aie::add(perimV,  aie::select(zero, B,   rim));
        internV = aie::add(internV, aie::select(zero, B,   inside));
        brightV = aie::add(brightV, aie::select(zero, R,   rim));
        countV  = aie::add(countV,  aie::select(zero, one, rim));
    }

    const int numPerimeter = (int)(aie::reduce_add(countV) + 0.5f);
    if (numPerimeter < MIN_INLIERS_FOR_DETECTION || numPerimeter > MAX_INLIERS_FOR_DETECTION)
        return PSO_NEG_INF;

    return aie::reduce_add(perimV)
         - HOLLOWNESS_PENALTY * aie::reduce_add(internV)
         + BRIGHT_POINT_BONUS * aie::reduce_add(brightV);
#endif
}

// ------------------------------------------------------------
// 5-point algebraic fit of  A x^2 + B xy + C y^2 + D x + E y + F = 0.
//
// The reference code takes the null vector of the 5x6 design matrix
// from a double-precision eigen-decomposition of D^T D. In single
// precision D^T D loses too many digits, so here the points are first
// centred and scaled to [-1, 1] and the null vector is obtained with
// Gauss-Jordan elimination (complete pivoting) directly on D. The
// conic, and therefore the ellipse, is the same; the checks that
// follow are the reference checks, done in the scaled frame
// (they are all invariant to that change of frame).
// ------------------------------------------------------------
static inline bool pso_fit_five(const float* xs, const float* ys, float* out) {
    const float CX = 0.5f * (float)IMG_W, CY = 0.5f * (float)IMG_H;
    const float S  = 0.5f * (float)(IMG_W > IMG_H ? IMG_W : IMG_H);
    const float invS = 1.0f / S;

    float M[5][6];
    int   col[6] = { 0, 1, 2, 3, 4, 5 };
    for (int i = 0; i < 5; ++i) {
        float x = (xs[i] - CX) * invS, y = (ys[i] - CY) * invS;
        M[i][0] = x * x; M[i][1] = x * y; M[i][2] = y * y;
        M[i][3] = x;     M[i][4] = y;     M[i][5] = 1.0f;
    }

    for (int r = 0; r < 5; ++r) {
        // complete pivoting over the remaining sub-matrix
        int pr = r, pc = r;
        float best = 0.0f;
        for (int i = r; i < 5; ++i)
            for (int j = r; j < 6; ++j)
                if (pso_abs(M[i][j]) > best) { best = pso_abs(M[i][j]); pr = i; pc = j; }
        if (best < 1.0e-6f) return false;                 // rank < 5: degenerate point set

        if (pr != r)
            for (int j = 0; j < 6; ++j) { float t = M[r][j]; M[r][j] = M[pr][j]; M[pr][j] = t; }
        if (pc != r) {
            for (int i = 0; i < 5; ++i) { float t = M[i][r]; M[i][r] = M[i][pc]; M[i][pc] = t; }
            int t = col[r]; col[r] = col[pc]; col[pc] = t;
        }

        float inv = 1.0f / M[r][r];
        for (int j = r; j < 6; ++j) M[r][j] *= inv;
        for (int i = 0; i < 5; ++i) {
            if (i == r) continue;
            float f = M[i][r];
            if (f == 0.0f) continue;
            for (int j = r; j < 6; ++j) M[i][j] -= f * M[r][j];
        }
    }

    // null vector: last (free) column = 1, pivots = -M[r][5]
    float v[6];
    v[col[5]] = 1.0f;
    for (int r = 0; r < 5; ++r) v[col[r]] = -M[r][5];

    // The null vector is defined only up to sign, and 0.5*atan2(b, a - c)
    // gives the major-axis direction only when a + c < 0 (for a + c > 0 it
    // gives the minor axis, i.e. theta is off by 90 degrees). Fix the sign so
    // every seed is oriented correctly. The reference code leaves the sign to
    // the eigen-solver, so its seeds are oriented at random.
    if (v[0] + v[2] > 0.0f)
        for (int j = 0; j < 6; ++j) v[j] = -v[j];

    const float a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5];

    // must be an ellipse: b^2 - 4ac < 0
    if (b * b - 4.0f * a * c >= 0.0f) return false;

    // real (non-imaginary) ellipse: det(M3) / (a + c) < 0
    const float det3 =
        a * (c * f - (e * 0.5f) * (e * 0.5f)) -
        (b * 0.5f) * ((b * 0.5f) * f - (e * 0.5f) * (d * 0.5f)) +
        (d * 0.5f) * ((b * 0.5f) * (e * 0.5f) - c * (d * 0.5f));
    if (det3 / (a + c) >= 0.0f) return false;

    const float denom = 4.0f * a * c - b * b;
    const float cxn = (b * e - 2.0f * c * d) / denom;
    const float cyn = (b * d - 2.0f * a * e) / denom;
    const float val = a * cxn * cxn + b * cxn * cyn + c * cyn * cyn + d * cxn + e * cyn + f;

    // eigenvalues of [[a, b/2], [b/2, c]]
    const float tr = a + c;
    const float diff = pso_sqrt(((a - c) * 0.5f) * ((a - c) * 0.5f) + (b * 0.5f) * (b * 0.5f));
    const float l1 = tr * 0.5f - diff;
    const float l2 = tr * 0.5f + diff;
    if (val / l1 >= 0.0f || val / l2 >= 0.0f) return false;

    const float s1 = pso_sqrt(-val / l1) * S;
    const float s2 = pso_sqrt(-val / l2) * S;
    const float semiA = s1 > s2 ? s1 : s2;
    const float semiB = s1 > s2 ? s2 : s1;
    const float theta = pso_wrap_pi(0.5f * pso_atan2(b, a - c));
    const float cx = CX + S * cxn;
    const float cy = CY + S * cyn;

    if (semiA < 2.0f || semiB < 2.0f) return false;
    if (!(cx >= 0.0f && cx <= (float)IMG_W && cy >= 0.0f && cy <= (float)IMG_H)) return false;
    if (semiA > (float)IMG_W || semiB > (float)IMG_H) return false;

    out[0] = cx; out[1] = cy; out[2] = semiA; out[3] = semiB; out[4] = theta;
    return true;
}

// ------------------------------------------------------------
// Seeding: `count` particles from 5-point fits through random edge
// points (up to count * 30 attempts), then random ellipses for any
// shortfall. Returns the number of ellipse-seeded particles.
// ------------------------------------------------------------
static inline int pso_seed_swarm(const EdgeSet& es, PsoRng& rng, float (*out)[5], int count) {
    const int n = es.n;
    int got = 0;
    for (int attempt = 0; attempt < count * SEED_ATTEMPTS_PER_PARTICLE && got < count; ++attempt) {
        if (n < 5) break;
        // five distinct edge points, uniformly at random
        int idx[5];
        float xs[5], ys[5];
        for (int s = 0; s < 5; ++s) {
            bool dup;
            do {
                int r = (int)(rng.u01() * (float)n);
                if (r >= n) r = n - 1;
                idx[s] = r;
                dup = false;
                for (int t = 0; t < s; ++t) if (idx[t] == r) dup = true;
            } while (dup);
            xs[s] = es.px[idx[s]];
            ys[s] = es.py[idx[s]];
        }
        if (pso_fit_five(xs, ys, out[got])) ++got;
    }
    const int seeded = got;
    for (; got < count; ++got) {
        out[got][0] = rng.u01() * (float)IMG_W;
        out[got][1] = rng.u01() * (float)IMG_H;
        out[got][2] = rng.u01() * (float)IMG_W / 3.0f;
        out[got][3] = rng.u01() * (float)IMG_H / 3.0f;
        out[got][4] = rng.u01() * PSO_PI;
    }
    return seeded;
}

static inline void pso_copy5(float* dst, const float* src) {
    for (int d = 0; d < 5; ++d) dst[d] = src[d];
}

// Random stream of particle i in iteration t (counter based, so it does
// not depend on which worker owns the particle).
static inline uint32_t pso_fmix32(uint32_t h) {          // MurmurHash3 finaliser
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}
static inline uint32_t pso_particle_seed(uint32_t seed, int it, int i) {
    uint32_t h = pso_fmix32(seed ^ 0x5BD1E995u);
    h = pso_fmix32(h ^ ((uint32_t)it * 0x9E3779B1u));
    return pso_fmix32(h ^ ((uint32_t)i * 0x85EBCA77u + 0x632BE5ABu));
}

// Keep b if it beats cur (higher score, first index on a tie).
static inline void pso_best_merge(PsoBest& cur, const PsoBest& b) {
    if (b.idx < 0) return;
    if (cur.idx < 0 || b.score > cur.score || (b.score == cur.score && b.idx < cur.idx)) cur = b;
}

// ============================================================
// Worker side: the particles i = w, w + W, ... of one shard
// ============================================================

// Start particle i at s.pos[i]: velocity 0, personal best = position.
// Returns its fitness.
static inline float pso_particle_init(PsoShard& s, const EdgeSet& es, int i) {
    for (int d = 0; d < 5; ++d) s.vel[i][d] = 0.0f;
    pso_copy5(s.pbest[i], s.pos[i]);
    return s.pbestScore[i] = pso_fitness(es, s.pos[i]);
}

// Initial swarm: store the own particles, score them (score[i]).
static inline void pso_shard_init(PsoShard& s, const EdgeSet& es, int w, int W,
                                  const float (*pos)[5], int P, float* score) {
    for (int i = w; i < P; i += W) {
        pso_copy5(s.pos[i], pos[i]);
        score[i] = pso_particle_init(s, es, i);
    }
}

// One Standard PSO or APSO iteration of the own particles; b = the best
// particle of the shard after the move (first index on a tie).
//   Std : v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x);  x = x + v
//   APSO: x = (1 - beta) x + beta gbest + alpha(t) eps,  eps ~ N(0, 1),
//         alpha(t) = alpha0 (alphaMin / alpha0)^(t / maxIter)
static inline void pso_shard_step(PsoShard& s, const EdgeSet& es, int algo, uint32_t seed,
                                  int it, int w, int W, int P, const float* gbest, PsoBest& b) {
    b.idx = -1;
    b.score = PSO_NEG_INF;
    const float alpha = (algo == ALGO_APSO)
        ? APSO_ALPHA0 * pso_exp(pso_log(APSO_ALPHA_MIN / APSO_ALPHA0) * (float)it / (float)APSO_ITERS)
        : 0.0f;
    for (int i = w; i < P; i += W) {
        PsoRng rng(pso_particle_seed(seed, it, i));
        if (algo == ALGO_STD) {
            for (int d = 0; d < 5; ++d) {
                const float r1 = rng.u01(), r2 = rng.u01();
                s.vel[i][d] = STD_W * s.vel[i][d]
                            + STD_C1 * r1 * (s.pbest[i][d] - s.pos[i][d])
                            + STD_C2 * r2 * (gbest[d]      - s.pos[i][d]);
                s.pos[i][d] += s.vel[i][d];
            }
        } else {
            for (int d = 0; d < 5; ++d) {
                const float eps = rng.gauss();
                s.pos[i][d] = (1.0f - APSO_BETA) * s.pos[i][d] + APSO_BETA * gbest[d] + alpha * eps;
            }
        }
        const float sc = pso_fitness(es, s.pos[i]);
        if (sc > s.pbestScore[i]) { s.pbestScore[i] = sc; pso_copy5(s.pbest[i], s.pos[i]); }
        if (b.idx < 0 || sc > b.score) { b.idx = i; b.score = sc; pso_copy5(b.pos, s.pos[i]); }
    }
}

// One SP-PSO iteration of the own particles. tabP / tabS = the personal
// bests of the whole swarm (ring table), reset[i] = 1 if particle i was
// re-seeded (its position becomes tabP[i], velocity 0). Writes the new
// personal best of every own particle into newP / newS.
static inline void pso_shard_step_sp(PsoShard& s, const EdgeSet& es, uint32_t seed, int it,
                                     int w, int W, const float (*tabP)[5], const float* tabS,
                                     const uint8_t* reset, float (*newP)[5], float* newS) {
    const int P = SP_PARTICLES, nb = SP_NEIGHBORHOOD;
    for (int i = w; i < P; i += W) {
        if (reset[i]) {
            pso_copy5(s.pos[i], tabP[i]);
            for (int d = 0; d < 5; ++d) s.vel[i][d] = 0.0f;
        }
        // local best of the ring neighbourhood; start at the first
        // neighbour so ties keep a non-zero social pull
        int bestN = ((i - nb) % P + P) % P;
        for (int d = -nb; d <= nb; ++d) {
            const int j = ((i + d) % P + P) % P;
            if (tabS[j] > tabS[bestN]) bestN = j;
        }
        PsoRng rng(pso_particle_seed(seed, it, i));
        for (int d = 0; d < 5; ++d) {
            const float r1 = rng.u01(), r2 = rng.u01();
            s.vel[i][d] = SP_W * s.vel[i][d]
                        + SP_C1 * r1 * (tabP[i][d]     - s.pos[i][d])
                        + SP_C2 * r2 * (tabP[bestN][d] - s.pos[i][d]);
            s.pos[i][d] += s.vel[i][d];
        }
        const float sc = pso_fitness(es, s.pos[i]);
        if (sc > tabS[i]) { newS[i] = sc; pso_copy5(newP[i], s.pos[i]); }
        else              { newS[i] = tabS[i]; pso_copy5(newP[i], tabP[i]); }
    }
}

// ============================================================
// Master side
// ============================================================

// Standard PSO / APSO: seed the swarm, score it in parallel, then per
// iteration one parallel step and a reduction to the new gbest.
template <typename Engine>
static inline void pso_run_global(int algo, uint32_t seed, PsoMasterState& m, Engine& eng,
                                  PsoResult& r) {
    const EdgeSet& es = m.es;
    const int P     = (algo == ALGO_STD) ? STD_PARTICLES : APSO_PARTICLES;
    const int iters = (algo == ALGO_STD) ? STD_ITERS : APSO_ITERS;
    PsoRng rng(seed);
    pso_seed_swarm(es, rng, m.pos, P);
    eng.init(m.pos, P, m.score);

    int g = 0;
    for (int i = 1; i < P; ++i) if (m.score[i] > m.score[g]) g = i;
    float gbest[5];
    pso_copy5(gbest, m.pos[g]);
    float gbestScore = m.score[g];

    for (int it = 0; it < iters; ++it) {
        PsoBest b;
        eng.step(it, P, gbest, b);
        if (b.idx >= 0 && b.score > gbestScore) { gbestScore = b.score; pso_copy5(gbest, b.pos); }
        r.hist[it] = gbestScore;
    }
    for (int it = iters; it < PSO_HIST_LEN; ++it) r.hist[it] = gbestScore;

    pso_copy5(r.best, gbest);
    r.score = gbestScore;
    r.evals = P + iters * P;
    r.iters = iters;
}

// SP-PSO: ring topology, the reinitCount worst particles re-seeded every
// reinitInterval iterations, gbest tracked throughout.
template <typename Engine>
static inline void pso_run_sp(uint32_t seed, PsoMasterState& m, Engine& eng, PsoResult& r) {
    const EdgeSet& es = m.es;
    const int P = SP_PARTICLES;
    PsoRng rng(seed);
    pso_seed_swarm(es, rng, m.pos, P);
    eng.init(m.pos, P, m.score);
    for (int i = 0; i < P; ++i) {
        pso_copy5(m.tabP[i], m.pos[i]);
        m.tabS[i] = m.score[i];
        m.reset[i] = 0;
    }

    int g = 0;
    for (int i = 1; i < P; ++i) if (m.tabS[i] > m.tabS[g]) g = i;
    float gbest[5];
    pso_copy5(gbest, m.tabP[g]);
    float gbestScore = m.tabS[g];

    for (int it = 0; it < SP_ITERS; ++it) {
        eng.step_sp(it, m.tabP, m.tabS, m.reset, m.newP, m.newS);
        for (int i = 0; i < P; ++i) {
            m.reset[i] = 0;
            if (m.newS[i] > gbestScore) { gbestScore = m.newS[i]; pso_copy5(gbest, m.newP[i]); }
            m.tabS[i] = m.newS[i];
            pso_copy5(m.tabP[i], m.newP[i]);
        }

        // periodic reinitialisation of the worst particles
        if ((it + 1) % SP_REINIT_INTERVAL == 0) {
            // order = particle indices sorted by personal-best score, ascending
            for (int i = 0; i < P; ++i) m.order[i] = i;
            for (int i = 1; i < P; ++i) {
                const int key = m.order[i];
                int j = i - 1;
                while (j >= 0 && m.tabS[m.order[j]] > m.tabS[key]) { m.order[j + 1] = m.order[j]; --j; }
                m.order[j + 1] = key;
            }
            pso_seed_swarm(es, rng, m.fresh, SP_REINIT_COUNT);
            for (int j = 0; j < SP_REINIT_COUNT; ++j) {
                const int wi = m.order[j];
                const float sc = pso_fitness(es, m.fresh[j]);
                pso_copy5(m.tabP[wi], m.fresh[j]);
                m.tabS[wi] = sc;
                m.reset[wi] = 1;
                if (sc > gbestScore) { gbestScore = sc; pso_copy5(gbest, m.fresh[j]); }
            }
        }
        r.hist[it] = gbestScore;
    }
    for (int it = SP_ITERS; it < PSO_HIST_LEN; ++it) r.hist[it] = gbestScore;

    pso_copy5(r.best, gbest);
    r.score = gbestScore;
    r.evals = P + SP_ITERS * P + (SP_ITERS / SP_REINIT_INTERVAL) * SP_REINIT_COUNT;
    r.iters = SP_ITERS;
}

// ------------------------------------------------------------
// One image record -> PsoResult for variant vi (even vi = DBSCAN).
// seedOffset = rep * 15485863 gives independent seed sets (0 = the
// main study).
// ------------------------------------------------------------
template <typename Engine>
static inline void pso_detect(int vi, const uint8_t* rec, PsoMasterState& m, Engine& eng,
                              PsoResult& r, uint32_t seedOffset = 0) {
    int flags;
    const int image = pso_load_image(rec, (vi % 2) == 0, m.es, m.scratch, flags);
    const uint32_t seed = pso_seed(image, vi) + seedOffset;
    r.variant = vi; r.image = image; r.flags = flags;
    eng.begin(m.es, seed);
    if (vi / 2 == ALGO_SPPSO) pso_run_sp(seed, m, eng, r);
    else                      pso_run_global(vi / 2, seed, m, eng, r);
}

#endif
