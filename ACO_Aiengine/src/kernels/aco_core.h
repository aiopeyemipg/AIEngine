#ifndef ACO_CORE_H
#define ACO_CORE_H

// ============================================================
// aco_core.h  —  the two ACO ellipse detectors, one image per call,
// with the work of every colony iteration spread over parallel workers
//
//   aco_load_image   image record -> AcoEdgeSet         (GrayImage.h)
//   aco_fitness      hollowness-model fitness           (Fitness.h)
//   aco_moments      image-moment seed                  (Moments.h)
//   aco_build_space  seeded grids + heuristics          (Detectors.h buildSearchSpace)
//   fped_detect      Factorized-Pheromone ACO           (Detectors.h FpedDetector)
//   stdaco_detect    joint-pheromone Standard ACO       (Detectors.h StdAcoDetector)
//
// Concurrency inside one image. The colony iterations are sequential
// (iteration t+1 samples from the trails that iteration t deposited),
// but the work inside one iteration is not:
//   1. trail phase  — FPED's five pheromone trails (h, k, a, b, theta)
//                     are independent: each one is updated (evaporation,
//                     deposit, smoothing, bounds), turned into a sampling
//                     table and sampled for all ants, at the same time.
//   2. score phase  — the ants are scored at the same time.
//   3. bookkeeping  — best-so-far and ranking (30 values), serial.
// The detectors are templates on an Engine that runs phases 1 and 2:
//   * AIE graph (aco_detectors.cc): the master tile broadcasts each phase
//     to ACO_WORKERS worker tiles; workers 0..4 own one trail each, and
//     all workers score their share of the ants.
//   * native bench (bench/aco_bench.cpp, -DACO_NATIVE): the same phases
//     on a pool of std::threads, or on one thread (serial baseline).
// Every trail has its own random stream and the fitness consumes none,
// so serial, threaded and AIE runs give bit-identical detections.
//
// This file has no ADF dependency. An Engine provides
//   void begin(const AcoEdgeSet& es, uint32_t seed, bool fped);  // new image
//                                                  // (FPED: set up the trails)
//   void trails(const FpedUpdate* up, int16_t (*bins)[FPED_ANTS],
//               float (*vals)[FPED_ANTS]);               // phase 1 (up = 0 first)
//   void score(const float (*cand)[5], int n, float* fit);  // phase 2
// with fit[i] = fitness, or ACO_INVALID_FIT for an invalid candidate.
//
// An ellipse is a float[5] = {cx, cy, a, b, theta} in image pixels.
// ============================================================

#include <stdint.h>
#include "include.h"
#include "aco_types.h"
#include "aco_math.h"

// Scalar fitness loop natively and in x86sim (vector emulation is very
// slow); 8-lane float vectors on the AI Engine. Add
// -DACO_X86SIM_VECTOR_FITNESS to simulate the vector loop in x86sim.
#if defined(ACO_NATIVE) || (defined(__X86SIM__) && !defined(ACO_X86SIM_VECTOR_FITNESS))
#define ACO_SCALAR_FITNESS 1
#endif

// ------------------------------------------------------------
// Image record -> AcoEdgeSet. Every pixel with
// gray > 0 is a hit, in row-major order, with its normalised intensity
// (GrayImage::edgePoints + withIntensity). No noise filtering: the
// squared-intensity weighting already de-emphasises dark counts.
// Returns the image id, or -1 for a padding record.
// Padding points sit far outside every candidate and carry zero weight.
// ------------------------------------------------------------
static inline int aco_load_image(const uint8_t* rec, AcoEdgeSet& es, int& flags) {
    const uint8_t* pix = rec + IMG_HEADER_BYTES;
    const int  image = (int)rec[0] | ((int)rec[1] << 8);
    const bool real  = rec[2] != 0;

    int n = 0;
    flags = 0;
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            const int g = pix[y * IMG_W + x];
            if (g == 0) continue;
            if (n == MAX_EDGES) { flags |= 1; continue; }
            const float inten = (float)g * (1.0f / 255.0f);
            es.px[n] = (float)x;
            es.py[n] = (float)y;
            es.boosted[n] = inten * inten;
            es.bright[n] = (inten >= HIGH_INTENSITY_THRESHOLD) ? 1.0f : 0.0f;
            ++n;
        }
    es.n = real ? n : 0;
    es.nPad = (es.n + 7) & ~7;
    for (int i = es.n; i < es.nPad; ++i) {
        es.px[i] = 1.0e4f;
        es.py[i] = 1.0e4f;
        es.boosted[i] = 0.0f;
        es.bright[i] = 0.0f;
    }
    return real ? image : -1;
}

static inline float aco_min(float a, float b) { return a < b ? a : b; }
static inline float aco_max(float a, float b) { return a > b ? a : b; }

// a >= b and theta in [0, pi)   (canonical() in Detectors.h)
static inline void aco_canonical(float* e, float h, float k, float a, float b, float th) {
    if (b > a) { const float t = a; a = b; b = t; th += ACO_HALF_PI; }
    e[0] = h; e[1] = k; e[2] = a; e[3] = b; e[4] = aco_wrap_pi(th);
}

// ------------------------------------------------------------
// Hollowness fitness of a canonical candidate. For every hit:
//   |E - 1| < tol  -> rim:      +intensity^2, +gamma per bright point
//   E < 1 - tol    -> interior: -lambda * intensity^2
// Returns false (the reference's -inf) for semi-axes < 2 px, a box
// leaving the frame, or a rim count outside [MIN, MAX].
// ------------------------------------------------------------
static inline bool aco_fitness(const AcoEdgeSet& es, const float* e, float& fit) {
    const float h = e[0], k = e[1], a = e[2], b = e[3];
    fit = ACO_INVALID_FIT;
    if (es.n == 0 || a < 2.0f || b < 2.0f) return false;

    float st, ct;
    aco_sincos(e[4], st, ct);
    const float bboxHW = aco_sqrt((a * ct) * (a * ct) + (b * st) * (b * st));
    const float bboxHH = aco_sqrt((a * st) * (a * st) + (b * ct) * (b * ct));
    if (h - bboxHW < 0.0f || h + bboxHW > (float)IMG_W ||
        k - bboxHH < 0.0f || k + bboxHH > (float)IMG_H)
        return false;

    const float inva2 = 1.0f / (a * a);
    const float invb2 = 1.0f / (b * b);

#ifdef ACO_SCALAR_FITNESS
    float perim = 0.0f, intern = 0.0f, brightCount = 0.0f;
    int numPerimeter = 0;
    for (int i = 0; i < es.n; ++i) {
        const float dx = es.px[i] - h, dy = es.py[i] - k;
        const float t1 = dx * ct + dy * st;
        const float t2 = dx * st - dy * ct;
        const float r = t1 * t1 * inva2 + t2 * t2 * invb2;
        if (aco_abs(r - 1.0f) < INLIER_TOLERANCE) {
            perim += es.boosted[i];
            brightCount += es.bright[i];
            ++numPerimeter;
        } else if (r < 1.0f - INLIER_TOLERANCE) {
            intern += es.boosted[i];
        }
    }
    if (numPerimeter < MIN_INLIERS_FOR_DETECTION || numPerimeter > MAX_INLIERS_FOR_DETECTION)
        return false;
    fit = perim - HOLLOWNESS_PENALTY * intern + BRIGHT_POINT_BONUS * brightCount;
    return true;
#else
    const aie::vector<float, 8> zero  = aie::zeros<float, 8>();
    const aie::vector<float, 8> one   = aie::broadcast<float, 8>(1.0f);
    const aie::vector<float, 8> tol   = aie::broadcast<float, 8>(INLIER_TOLERANCE);
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
        aie::vector<float, 8> r = aie::mac(aie::mul(t1s, inva2), t2s, invb2).template to_vector<float>();

        aie::vector<float, 8> dev = aie::abs(aie::sub(r, 1.0f));
        aie::mask<8> rim    = aie::lt(dev, tol);
        aie::mask<8> inside = aie::lt(r, inLim);    // disjoint from rim

        aie::vector<float, 8> B = aie::load_v<8>(es.boosted + i);
        aie::vector<float, 8> R = aie::load_v<8>(es.bright + i);

        perimV  = aie::add(perimV,  aie::select(zero, B,   rim));
        internV = aie::add(internV, aie::select(zero, B,   inside));
        brightV = aie::add(brightV, aie::select(zero, R,   rim));
        countV  = aie::add(countV,  aie::select(zero, one, rim));
    }

    const int numPerimeter = (int)(aie::reduce_add(countV) + 0.5f);
    if (numPerimeter < MIN_INLIERS_FOR_DETECTION || numPerimeter > MAX_INLIERS_FOR_DETECTION)
        return false;
    fit = aie::reduce_add(perimV)
        - HOLLOWNESS_PENALTY * aie::reduce_add(internV)
        + BRIGHT_POINT_BONUS * aie::reduce_add(brightV);
    return true;
#endif
}

// ------------------------------------------------------------
// Image-moment seed: intensity^2-weighted centroid and covariance;
// a0 = sqrt(2 lambda1), b0 = sqrt(2 lambda2), theta0 from the leading
// eigenvector (momentEstimate).
// ------------------------------------------------------------
static inline void aco_moments(const AcoEdgeSet& es, AcoSeed& s) {
    float ws = 0.0f, sx = 0.0f, sy = 0.0f;
    for (int i = 0; i < es.n; ++i) {
        const float w = es.boosted[i];
        ws += w; sx += w * es.px[i]; sy += w * es.py[i];
    }
    const bool unweighted = !(ws > 0.0f);
    if (unweighted) {
        ws = (float)es.n; sx = sy = 0.0f;
        for (int i = 0; i < es.n; ++i) { sx += es.px[i]; sy += es.py[i]; }
    }
    s.h0 = sx / ws;
    s.k0 = sy / ws;

    // the reference keeps the squared-intensity weights here
    float cxx = 0.0f, cyy = 0.0f, cxy = 0.0f;
    for (int i = 0; i < es.n; ++i) {
        const float w = es.boosted[i];
        const float dx = es.px[i] - s.h0, dy = es.py[i] - s.k0;
        cxx += w * dx * dx; cyy += w * dy * dy; cxy += w * dx * dy;
    }
    cxx /= ws; cyy /= ws; cxy /= ws;

    const float tr = cxx + cyy;
    const float det = cxx * cyy - cxy * cxy;
    const float disc = aco_sqrt(aco_max(0.0f, tr * tr / 4.0f - det));
    const float lam1 = aco_max(1.0e-6f, tr / 2.0f + disc);
    const float lam2 = aco_max(1.0e-6f, tr / 2.0f - disc);

    s.a0 = aco_sqrt(2.0f * lam1);
    s.b0 = aco_sqrt(2.0f * lam2);

    float vx, vy;
    if (aco_abs(cxy) > 1.0e-12f) { vx = lam1 - cyy; vy = cxy; }
    else if (cxx >= cyy)         { vx = 1.0f; vy = 0.0f; }
    else                         { vx = 0.0f; vy = 1.0f; }
    s.theta0 = aco_wrap_pi(aco_atan2(vy, vx));

    const float half = 0.5f * (float)(IMG_W < IMG_H ? IMG_W : IMG_H);
    s.a0 = aco_min(aco_max(s.a0, 2.0f), half);
    s.b0 = aco_min(aco_max(s.b0, 2.0f), half);
    if (s.b0 > s.a0) {
        const float t = s.a0; s.a0 = s.b0; s.b0 = t;
        s.theta0 = aco_wrap_pi(s.theta0 + ACO_HALF_PI);
    }
}

static inline void aco_linspace(float* v, float lo, float hi, int n, bool endpoint) {
    const float step = (n > 1) ? (hi - lo) / (float)(endpoint ? n - 1 : n) : 0.0f;
    for (int i = 0; i < n; ++i) v[i] = lo + step * (float)i;
}

// ------------------------------------------------------------
// Grid of dimension d centred on the moment seed (centre +- 7 px,
// axes +- 4 px, theta over [0, pi)) and its Gaussian desirability
// profile with a floor, normalised to sum 1 (buildSearchSpace,
// useSeed = true).
// ------------------------------------------------------------
static inline void aco_build_dim(const AcoSeed& s, int d, float* grid, float* eta) {
    const int n = ACO_BINS;
    const float half = 0.5f * (float)(IMG_W < IMG_H ? IMG_W : IMG_H);
    if (d == 0) {
        aco_linspace(grid, aco_max(0.5f, s.h0 - CENTER_HALFSPAN),
                     aco_min((float)IMG_W - 0.5f, s.h0 + CENTER_HALFSPAN), n, true);
    } else if (d == 1) {
        aco_linspace(grid, aco_max(0.5f, s.k0 - CENTER_HALFSPAN),
                     aco_min((float)IMG_H - 0.5f, s.k0 + CENTER_HALFSPAN), n, true);
    } else if (d == 2 || d == 3) {
        const float c = (d == 2) ? s.a0 : s.b0;
        const float lo = aco_max(2.0f, c - AXIS_HALFSPAN);
        aco_linspace(grid, lo, aco_max(lo + 0.5f, aco_min(half + 0.5f, c + AXIS_HALFSPAN)), n, true);
    } else {
        aco_linspace(grid, 0.0f, ACO_PI, n, false);
    }

    const bool nearCircular = (s.a0 / aco_max(s.b0, 1.0e-6f)) < NEAR_CIRCULAR;
    const float centre[5] = { s.h0, s.k0, s.a0, s.b0, s.theta0 };
    const float sigma[5]  = { ETA_SIGMA_C, ETA_SIGMA_C, ETA_SIGMA_AX, ETA_SIGMA_AX, ETA_SIGMA_TH };
    const bool flat = (d == 4 && nearCircular);
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        float v = 1.0f;
        if (!flat) {
            float diff = grid[j] - centre[d];
            if (d == 4) {
                diff = aco_wrap_pi(aco_abs(diff));
                diff = aco_min(diff, ACO_PI - diff);
            }
            const float z = diff / sigma[d];
            v = aco_exp(-0.5f * z * z) + ETA_FLOOR;
        }
        eta[j] = v;
        sum += v;
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j < n; ++j) eta[j] *= inv;
}

static inline void aco_build_space(const AcoSeed& s, AcoSpace& sp) {
    for (int d = 0; d < 5; ++d) aco_build_dim(s, d, sp.grid[d], sp.eta[d]);
}

// Cumulative sums of the weights (std::discrete_distribution table).
static inline void aco_cumsum(float* cdf, const float* w, int n) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) { acc += w[j]; cdf[j] = acc; }
}

// Index j with probability w[j] / sum(w): first j with cdf[j] > u * total
// (binary search; zero-weight bins are never chosen).
static inline int aco_sample(const float* cdf, int n, AcoRng& rng) {
    const float r = rng.u01() * cdf[n - 1];
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (cdf[mid] > r) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// tau^alpha (alpha = 1.5 -> tau * sqrt(tau); otherwise exp(alpha ln tau))
static inline float aco_pow_alpha(float x, float alpha) {
    if (alpha == 1.5f) return x * aco_sqrt(x);
    if (alpha == 1.0f) return x;
    return aco_exp(alpha * aco_log(x));
}

// Sort (fit, idx) pairs, best fitness first (insertion sort, n <= 30).
static inline void aco_sort_desc(float* fit, int16_t* idx, int n) {
    for (int i = 1; i < n; ++i) {
        const float f = fit[i];
        const int16_t id = idx[i];
        int j = i - 1;
        while (j >= 0 && fit[j] < f) { fit[j + 1] = fit[j]; idx[j + 1] = idx[j]; --j; }
        fit[j + 1] = f;
        idx[j + 1] = id;
    }
}

// ------------------------------------------------------------
// 1-D Gaussian trail smoothing (sigma 1.5 bins, radius 6): wrap-around
// for the periodic theta trail, 'nearest' edges otherwise.
// ------------------------------------------------------------
static inline void aco_smooth(float* v, float* tmp, const float* kern, bool wrap) {
    const int n = ACO_BINS, R = FPED_SMOOTH_RAD;
    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int j = -R; j <= R; ++j) {
            int idx = i + j;
            if (wrap) { if (idx < 0) idx += n; else if (idx >= n) idx -= n; }
            else      { if (idx < 0) idx = 0;  else if (idx >= n) idx = n - 1; }
            acc += kern[j + R] * v[idx];
        }
        tmp[i] = acc;
    }
    for (int i = 0; i < n; ++i) v[i] = tmp[i];
}

// Score candidates begin, begin+step, ... < end (one worker's share).
static inline void aco_score_range(const AcoEdgeSet& es, const float (*cand)[5],
                                   int begin, int end, int step, float* fit) {
    for (int i = begin; i < end; i += step) aco_fitness(es, cand[i], fit[i]);
}

// ============================================================
// FPED: five independent 1-D pheromone trails. Each ant picks one bin
// per dimension with p ~ tau^alpha * eta^beta. Update per iteration:
// evaporation -> rank-based deposit of the 10 best valid ants ->
// elitist deposit on the global-best bins -> trail smoothing -> clamp.
// At iteration 50 the grids are rebuilt as tight windows around the
// incumbent best (progressive trail refinement).
// ============================================================

// Random stream of trail d of one detection.
static inline uint32_t fped_trail_seed(uint32_t seed, int d) {
    return seed + 0x6A09E667u * (uint32_t)(d + 1);
}

// Set up trail d from the moment seed: grid, eta^beta, tau = 1.
static inline void fped_trail_init(FpedTrail& t, int d, const AcoSeed& s, uint32_t seed) {
    aco_build_dim(s, d, t.grid, t.tmp);
    for (int j = 0; j < ACO_BINS; ++j) {
        const float e = t.tmp[j];
        t.etaPow[j] = (FPED_BETA == 3.0f) ? e * e * e : aco_exp(FPED_BETA * aco_log(e));
        t.tau[j] = 1.0f;
    }
    float ksum = 0.0f;
    for (int i = -FPED_SMOOTH_RAD; i <= FPED_SMOOTH_RAD; ++i) {
        const float z = (float)i / FPED_SMOOTH_SIGMA;
        t.kern[i + FPED_SMOOTH_RAD] = aco_exp(-0.5f * z * z);
        ksum += t.kern[i + FPED_SMOOTH_RAD];
    }
    for (int i = 0; i < 2 * FPED_SMOOTH_RAD + 1; ++i) t.kern[i] /= ksum;
    t.wrap = (d == 4) ? 1 : 0;
    t.rng = AcoRng(fped_trail_seed(seed, d));
}

// ------------------------------------------------------------
// Trail phase of one dimension (runs on its own worker):
//   finish the previous iteration with the master's update (if any),
//   then build the sampling table and draw the bin of every ant.
// ------------------------------------------------------------
static inline void fped_trail_step(FpedTrail& t, int d, const FpedUpdate* up,
                                   int16_t* bins, float* vals) {
    const int n = ACO_BINS;
    if (up) {
        for (int j = 0; j < n; ++j) t.tau[j] *= (1.0f - FPED_RHO);           // evaporation
        for (int r = 0; r < up->ne; ++r)                                      // rank-based deposit
            t.tau[up->bins[r]] += FPED_Q * (float)(FPED_ELITE - r) / (float)FPED_ELITE;
        if (up->bestBin >= 0) t.tau[up->bestBin] += FPED_QBEST;              // elitist deposit
        aco_smooth(t.tau, t.tmp, t.kern, t.wrap != 0);                       // smoothing
        for (int j = 0; j < n; ++j)                                           // MAX-MIN bounds
            t.tau[j] = aco_min(aco_max(t.tau[j], FPED_TAU_MIN), FPED_TAU_MAX);

        if (up->refine) {          // tight window around the incumbent best
            const float c = up->centre;
            const float half = 0.5f * (float)(IMG_W < IMG_H ? IMG_W : IMG_H);
            if (d == 0)
                aco_linspace(t.grid, aco_max(0.5f, c - FPED_REFINE_C),
                             aco_min((float)IMG_W - 0.5f, c + FPED_REFINE_C), n, true);
            else if (d == 1)
                aco_linspace(t.grid, aco_max(0.5f, c - FPED_REFINE_C),
                             aco_min((float)IMG_H - 0.5f, c + FPED_REFINE_C), n, true);
            else if (d == 2 || d == 3) {
                const float lo = aco_max(2.0f, c - FPED_REFINE_AX);
                aco_linspace(t.grid, lo, aco_max(lo + 0.25f, aco_min(half + 0.5f, c + FPED_REFINE_AX)), n, true);
            } else
                aco_linspace(t.grid, c - FPED_REFINE_TH, c + FPED_REFINE_TH, n, true);
            t.wrap = 0;                                 // window grid, not periodic
            const float u = 1.0f / (float)n;
            for (int j = 0; j < n; ++j) { t.tau[j] = 1.0f; t.etaPow[j] = u; }
        }
    }

    for (int j = 0; j < n; ++j) t.tmp[j] = aco_pow_alpha(t.tau[j], FPED_ALPHA) * t.etaPow[j];
    aco_cumsum(t.cdf, t.tmp, n);
    for (int a = 0; a < FPED_ANTS; ++a) {
        const int b = aco_sample(t.cdf, n, t.rng);
        bins[a] = (int16_t)b;
        vals[a] = t.grid[b];
    }
}

// Five trails held locally (serial and threaded native engines).
struct FpedLocalTrails {
    FpedTrail tr[5];
    void init(const AcoEdgeSet& es, uint32_t seed) {
        AcoSeed s;
        aco_moments(es, s);
        for (int d = 0; d < 5; ++d) fped_trail_init(tr[d], d, s, seed);
    }
};

// ------------------------------------------------------------
// FPED master: per iteration, trail phase -> assemble the ants ->
// score phase -> bookkeeping, which produces the trail updates of the
// next trail phase.
// ------------------------------------------------------------
template <typename Engine>
static inline void fped_detect(const AcoEdgeSet& es, uint32_t seed, FpedState& st,
                               Engine& eng, AcoResult& res) {
    eng.begin(es, seed, true);

    bool  bestValid = false;
    float bestFit = 0.0f;
    float bestE[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int   bestIdx[5] = { -1, -1, -1, -1, -1 };
    int   evals = 0;

    for (int it = 0; it < FPED_ITERS; ++it) {
        // 1. trail phase: all five trails at once
        eng.trails(it == 0 ? (const FpedUpdate*)0 : st.up, st.bins, st.vals);

        // 2. score phase: all ants at once
        for (int a = 0; a < FPED_ANTS; ++a)
            aco_canonical(st.cand[a], st.vals[0][a], st.vals[1][a], st.vals[2][a],
                          st.vals[3][a], st.vals[4][a]);
        eng.score(st.cand, FPED_ANTS, st.fit);
        evals += FPED_ANTS;

        // 3. bookkeeping, in ant order as in the reference
        int ns = 0;
        for (int a = 0; a < FPED_ANTS; ++a) {
            const float f = st.fit[a];
            if (!aco_valid_fit(f)) continue;
            st.scoredFit[ns] = f;
            st.scoredAnt[ns] = (int16_t)a;
            ++ns;
            if (!bestValid || f > bestFit) {
                bestValid = true;
                bestFit = f;
                for (int d = 0; d < 5; ++d) { bestE[d] = st.cand[a][d]; bestIdx[d] = st.bins[d][a]; }
            }
        }
        aco_sort_desc(st.scoredFit, st.scoredAnt, ns);
        const int ne = ns < FPED_ELITE ? ns : FPED_ELITE;
        const int refine = (FPED_REFINE_AT > 0 && it + 1 == FPED_REFINE_AT && bestValid) ? 1 : 0;
        for (int d = 0; d < 5; ++d) {
            FpedUpdate& u = st.up[d];
            u.ne = ne;
            for (int r = 0; r < ne; ++r) u.bins[r] = st.bins[d][st.scoredAnt[r]];
            u.bestBin = (bestIdx[0] >= 0) ? bestIdx[d] : -1;
            u.refine = refine;
            u.centre = bestE[d];
        }
        if (refine) bestIdx[0] = -1;                  // old bins no longer valid
        if (it < ACO_HIST_LEN) res.hist[it] = bestValid ? bestFit : ACO_INVALID_FIT;
    }

    res.detected = bestValid ? 1 : 0;
    for (int d = 0; d < 5; ++d) res.ell[d] = bestE[d];
    res.fitness = bestFit;
    res.evals = evals;
    res.iters = FPED_ITERS;
}

// ============================================================
// Standard ACO baseline: one pheromone value per complete tuple of a
// fixed archive of STD_TUPLES candidates drawn from the same heuristic
// distributions. Same deposit / evaporation / bounds as FPED. A tuple
// is scored once (cached), so evals = unique tuples scored. Each
// iteration the not-yet-scored tuples picked by the ants are scored
// concurrently.
// ============================================================
template <typename Engine>
static inline void stdaco_detect(const AcoEdgeSet& es, uint32_t seed, StdAcoState& st,
                                 Engine& eng, AcoResult& res) {
    const int n = ACO_BINS;
    eng.begin(es, seed, false);
    AcoRng rng(seed);
    AcoSeed s;
    aco_moments(es, s);
    aco_build_space(s, st.sp);

    // fixed joint archive: one heuristic draw per dimension, in order h, k, a, b, theta
    for (int j = 0; j < STD_TUPLES; ++j) {
        int idx[5];
        for (int d = 0; d < 5; ++d) {
            aco_cumsum(st.cdf, st.sp.eta[d], n);
            idx[d] = aco_sample(st.cdf, n, rng);
        }
        aco_canonical(st.cand[j], st.sp.grid[0][idx[0]], st.sp.grid[1][idx[1]],
                      st.sp.grid[2][idx[2]], st.sp.grid[3][idx[3]], st.sp.grid[4][idx[4]]);
        st.cacheState[j] = 0;
        st.tau[j] = 1.0f;
    }

    float bestFit = 0.0f;
    int   bestJ = -1;
    int   evals = 0;

    for (int it = 0; it < STD_ITERS; ++it) {
        for (int j = 0; j < STD_TUPLES; ++j) st.cdf[j] = aco_pow_alpha(st.tau[j], STD_ALPHA);
        float acc = 0.0f;
        for (int j = 0; j < STD_TUPLES; ++j) { acc += st.cdf[j]; st.cdf[j] = acc; }

        // the ants pick their tuples (the distribution is fixed within an iteration)
        int nb = 0;
        for (int a = 0; a < STD_ANTS; ++a) {
            const int j = aco_sample(st.cdf, STD_TUPLES, rng);
            st.pick[a] = (int16_t)j;
            if (st.cacheState[j] == 0) {                 // first time: queue for scoring
                st.cacheState[j] = 1;
                st.batchIdx[nb] = (int16_t)j;
                for (int d = 0; d < 5; ++d) st.batchCand[nb][d] = st.cand[j][d];
                ++nb;
            }
        }
        // score the new tuples concurrently
        if (nb > 0) {
            eng.score(st.batchCand, nb, st.batchFit);
            for (int b = 0; b < nb; ++b) st.cacheFit[st.batchIdx[b]] = st.batchFit[b];
            evals += nb;
        }

        int ns = 0;
        for (int a = 0; a < STD_ANTS; ++a) {
            const int j = st.pick[a];
            const float f = st.cacheFit[j];
            if (!aco_valid_fit(f)) continue;
            st.scoredFit[ns] = f;
            st.scoredIdx[ns] = (int16_t)j;
            ++ns;
            if (bestJ < 0 || f > bestFit) { bestFit = f; bestJ = j; }
        }

        for (int j = 0; j < STD_TUPLES; ++j) st.tau[j] *= (1.0f - STD_RHO);
        aco_sort_desc(st.scoredFit, st.scoredIdx, ns);
        const int ne = ns < STD_ELITE ? ns : STD_ELITE;
        for (int r = 0; r < ne; ++r) {
            const float wr = (float)(STD_ELITE - r) / (float)STD_ELITE;
            st.tau[st.scoredIdx[r]] += STD_Q * wr;
        }
        if (bestJ >= 0) st.tau[bestJ] += STD_QBEST;
        for (int j = 0; j < STD_TUPLES; ++j)
            st.tau[j] = aco_min(aco_max(st.tau[j], STD_TAU_MIN), STD_TAU_MAX);
        if (it < ACO_HIST_LEN) res.hist[it] = bestJ >= 0 ? bestFit : ACO_INVALID_FIT;
    }

    res.detected = bestJ >= 0 ? 1 : 0;
    for (int d = 0; d < 5; ++d) res.ell[d] = bestJ >= 0 ? st.cand[bestJ][d] : 0.0f;
    res.fitness = bestFit;
    res.evals = evals;
    res.iters = STD_ITERS;
}

// ------------------------------------------------------------
// Detect the ellipse of one image record with the algorithm `vi`.
// seedOffset = rep * 15485863 gives the independent seed sets of the
// reference repeat study (0 = the main study).
// res.image = -1 for a padding record; images with fewer hits than the
// minimum rim count are returned undetected, as in the reference.
// ------------------------------------------------------------
template <typename Engine>
static inline void aco_detect_image(const uint8_t* rec, int vi, AcoEdgeSet& es,
                                    FpedState* fped, StdAcoState* stdaco,
                                    Engine& eng, AcoResult& res, uint32_t seedOffset = 0) {
    res.image = aco_load_image(rec, es, res.flags);
    res.detected = 0; res.fitness = 0.0f; res.evals = 0; res.iters = 0;
    for (int d = 0; d < 5; ++d) res.ell[d] = 0.0f;
    for (int i = 0; i < ACO_HIST_LEN; ++i) res.hist[i] = ACO_INVALID_FIT;
    if (res.image < 0 || es.n < MIN_INLIERS_FOR_DETECTION) return;
    const uint32_t seed = aco_seed(res.image, vi) + seedOffset;
    if (vi == VI_FPED) fped_detect(es, seed, *fped, eng, res);
    else               stdaco_detect(es, seed, *stdaco, eng, res);
}

#endif
