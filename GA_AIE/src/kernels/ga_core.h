
#ifndef GA_CORE_H
#define GA_CORE_H

// ============================================================
// ga_core.h  —  building blocks shared by the three GA detectors
//
//   ga_load_image      image record -> EdgeSet           (GrayImage.h)
//   ga_normalize       a >= b, theta in [0, pi)          (Ellipse.h normalize)
//   ga_fitness         hollowness-model fitness          (Fitness.h)
//   ga_random_individual / ga_tournament / ga_blx /
//   ga_mutate          GA operators                      (GaOperators.h)
//   ga_fitness_std     convergence test of the Micro GA  (GaOperators.h)
//   ga_sort_desc       sort a small population by fitness
//   ga_order_desc      sorted index of a large population (heapsort)
//
// No adf dependency (the packet I/O is in ga_detectors.cc), so the
// same code also builds natively with -DGA_NATIVE.
// An individual is a float[IND_WORDS] = {cx, cy, a, b, theta, fitness}.
// ============================================================

#include "include.h"
#include "ga_types.h"
#include "ga_math.h"

// The 8-lane vector fitness runs on the AI Engine. x86sim and the
// native build use a scalar loop with the same arithmetic (see ga_fitness).
#if !defined(GA_NATIVE) && (!defined(__X86SIM__) || defined(GA_X86SIM_VECTOR_FITNESS))
#define GA_VECTOR_FITNESS 1
#else
#define GA_VECTOR_FITNESS 0
#endif

// ------------------------------------------------------------
// Image record (IMG_RECORD_BYTES) -> EdgeSet. Every pixel with gray > 0
// is an edge point, in row-major order, weighted by its normalised
// intensity (GrayImage::edgePoints + withIntensity).
// Returns the image id, or -1 for a padding record.
// Padding points sit far outside every candidate ellipse and carry
// zero weight, so they never contribute to the fitness.
// ------------------------------------------------------------
static inline int ga_load_image(const uint8_t* rec, EdgeSet& es, int& flags) {
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
            es.boosted[n] = inten * inten;                            // intensity^2
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

static inline void ga_copy5(float* dst, const float* src) {
    for (int d = 0; d < 5; ++d) dst[d] = src[d];
}

static inline void ga_copy_ind(float* dst, const float* src) {
    for (int d = 0; d < IND_WORDS; ++d) dst[d] = src[d];
}

// Force a >= b and wrap theta into [0, pi)   (normalize() in Ellipse.h)
static inline void ga_normalize(float* g) {
    if (g[3] > g[2]) {
        const float t = g[2]; g[2] = g[3]; g[3] = t;
        g[4] += GA_HALF_PI;
    }
    g[4] = ga_wrap_pi(g[4]);
}

// ------------------------------------------------------------
// Fitness (hollowness model). The candidate is normalised first,
// like the reference. For every edge point:
//   |r - 1| < tol  -> rim:      +intensity^2, bright rim points counted
//   r < 1 - tol    -> interior: -penalty * intensity^2
// Invalid candidates (semi-axis < 2, leaving the image, or a rim count
// outside [MIN, MAX]) score INVALID_FITNESS (-1e6, finite on purpose:
// the Micro-GA convergence test relies on it).
// The point loop runs on 8-lane float vectors on the AI Engine.
// ------------------------------------------------------------
static inline float ga_fitness(const EdgeSet& es, const float* raw) {
    if (es.n == 0) return INVALID_FITNESS;
    for (int d = 0; d < 5; ++d)
        if (!(ga_abs(raw[d]) < 1.0e6f)) return INVALID_FITNESS;    // also rejects NaN

    float p[5];
    ga_copy5(p, raw);
    ga_normalize(p);
    const float h = p[0], k = p[1], a = p[2], b = p[3], theta = p[4];
    if (!(a >= 2.0f) || !(b >= 2.0f)) return INVALID_FITNESS;

    float st, ct;
    ga_sincos(theta, st, ct);
    const float bboxHW = ga_sqrt((a * ct) * (a * ct) + (b * st) * (b * st));
    const float bboxHH = ga_sqrt((a * st) * (a * st) + (b * ct) * (b * ct));
    if (h - bboxHW < 0.0f || h + bboxHW > (float)IMG_W ||
        k - bboxHH < 0.0f || k + bboxHH > (float)IMG_H)
        return INVALID_FITNESS;

    const float inva2 = 1.0f / (a * a);
    const float invb2 = 1.0f / (b * b);

#if !GA_VECTOR_FITNESS
    // x86 simulation / native build: plain scalar loop with the same arithmetic. The AIE
    // vector intrinsics are emulated bit-accurately by x86sim, which is far
    // too slow for a 1000-image run. Add -DGA_X86SIM_VECTOR_FITNESS to the
    // x86sim build to simulate the vector loop below instead.
    float perim = 0.0f, intern = 0.0f, brightCount = 0.0f;
    int numPerimeter = 0;
    for (int i = 0; i < es.n; ++i) {
        const float dx = es.px[i] - h, dy = es.py[i] - k;
        const float t1 = dx * ct + dy * st;
        const float t2 = dx * st - dy * ct;
        const float r = t1 * t1 * inva2 + t2 * t2 * invb2;
        if (ga_abs(r - 1.0f) < INLIER_TOLERANCE) {
            perim += es.boosted[i];
            brightCount += es.bright[i];
            ++numPerimeter;
        } else if (r < 1.0f - INLIER_TOLERANCE) {
            intern += es.boosted[i];
        }
    }
    if (numPerimeter < MIN_INLIERS_FOR_DETECTION || numPerimeter > MAX_INLIERS_FOR_DETECTION)
        return INVALID_FITNESS;
    return perim - HOLLOWNESS_PENALTY * intern + BRIGHT_POINT_BONUS * brightCount;
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
        return INVALID_FITNESS;

    return aie::reduce_add(perimV)
         - HOLLOWNESS_PENALTY * aie::reduce_add(internV)
         + BRIGHT_POINT_BONUS * aie::reduce_add(brightV);
#endif
}

// A candidate is a valid detection when it clears every hard constraint.
static inline bool ga_is_valid(const EdgeSet& es, const float* g) {
    return ga_fitness(es, g) > INVALID_FITNESS * 0.5f;
}

// ------------------------------------------------------------
// Uniformly sampled ellipse constrained within the image bounds
// (randomIndividual).
// ------------------------------------------------------------
static inline void ga_random_individual(GaRng& rng, float* g) {
    const float aMax = 0.5f * (float)(IMG_W < IMG_H ? IMG_W : IMG_H);
    const float a = rng.uniform(1.5f, aMax);
    const float b = rng.uniform(1.5f, a);          // b <= a before normalisation
    g[0] = rng.uniform(1.0f, (float)IMG_W - 1.0f);
    g[1] = rng.uniform(1.0f, (float)IMG_H - 1.0f);
    g[2] = a;
    g[3] = b;
    g[4] = rng.uniform(0.0f, GA_PI);
    ga_normalize(g);
}

// ------------------------------------------------------------
// k-way tournament: index of the best among k distinct individuals
// (tournamentSelect). The k distinct indices are drawn by rejection,
// which gives the same distribution as random.sample.
// ------------------------------------------------------------
static inline int ga_tournament(const float* pop, int n, int k, GaRng& rng) {
    if (k > n) k = n;
    int picks[16];
    int best = -1;
    for (int s = 0; s < k; ++s) {
        int r;
        bool dup;
        do {
            r = (int)(rng.u01() * (float)n);
            if (r >= n) r = n - 1;
            dup = false;
            for (int t = 0; t < s; ++t) if (picks[t] == r) dup = true;
        } while (dup);
        picks[s] = r;
        if (best < 0 || pop[r * IND_WORDS + IND_FIT] > pop[best * IND_WORDS + IND_FIT]) best = r;
    }
    return best;
}

// ------------------------------------------------------------
// BLX-alpha crossover (blxAlphaCrossover): with probability `rate`
// each child gene is drawn from the parents' range widened by alpha.
// ------------------------------------------------------------
static inline void ga_blx(const float* p1, const float* p2, float rate, float alpha,
                          GaRng& rng, float* c1, float* c2) {
    if (rng.u01() > rate) { ga_copy5(c1, p1); ga_copy5(c2, p2); return; }
    for (int i = 0; i < 5; ++i) {
        float lo = p1[i] < p2[i] ? p1[i] : p2[i];
        float hi = p1[i] < p2[i] ? p2[i] : p1[i];
        const float d = hi - lo + 1.0e-10f;
        lo -= alpha * d;
        hi += alpha * d;
        c1[i] = rng.uniform(lo, hi);
        c2[i] = rng.uniform(lo, hi);
    }
    ga_normalize(c1);
    ga_normalize(c2);
}

// ------------------------------------------------------------
// Per-gene Gaussian mutation with cosine-annealed rate and linearly
// decaying step sizes (adaptiveMutate). progress = fraction of the
// search budget consumed (0 -> 1).
// ------------------------------------------------------------
static inline void ga_mutate(float* g, float progress, float mut0, GaRng& rng) {
    float s, c;
    ga_sincos(GA_PI * progress, s, c);
    const float rate = (mut0 - MUT_FINAL_RATE) * 0.5f * (1.0f + c) + MUT_FINAL_RATE;
    float scale = 1.0f - 0.90f * progress;
    if (scale < 0.05f) scale = 0.05f;
    const float sigma[5] = {
        (float)IMG_W * MUT_CX_FRAC * scale,
        (float)IMG_H * MUT_CY_FRAC * scale,
        MUT_AXIS_SCALE * scale,
        MUT_AXIS_SCALE * scale,
        MUT_THETA_SCALE * scale,
    };
    for (int i = 0; i < 5; ++i)
        if (rng.u01() < rate) g[i] += rng.gauss(0.0f, sigma[i]);
    ga_normalize(g);
}

// Std-dev of the fitness values above INVALID_FITNESS * 100 (fitnessStd).
// The -1e6 sentinel is included, as in the reference.
static inline float ga_fitness_std(const float* pop, int n) {
    float sum = 0.0f;
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        const float f = pop[i * IND_WORDS + IND_FIT];
        if (f > INVALID_FITNESS * 100.0f) { sum += f; ++cnt; }
    }
    if (cnt < 2) return 1.0e30f;            // "not converged"
    const float mean = sum / (float)cnt;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float f = pop[i * IND_WORDS + IND_FIT];
        if (f > INVALID_FITNESS * 100.0f) { const float d = f - mean; acc += d * d; }
    }
    return ga_sqrt(acc / (float)cnt);
}

// Sort a small population in place, best fitness first (insertion sort).
static inline void ga_sort_desc(float* pop, int n) {
    float tmp[IND_WORDS];
    for (int i = 1; i < n; ++i) {
        ga_copy_ind(tmp, pop + i * IND_WORDS);
        int j = i - 1;
        while (j >= 0 && pop[j * IND_WORDS + IND_FIT] < tmp[IND_FIT]) {
            ga_copy_ind(pop + (j + 1) * IND_WORDS, pop + j * IND_WORDS);
            --j;
        }
        ga_copy_ind(pop + (j + 1) * IND_WORDS, tmp);
    }
}

// ------------------------------------------------------------
// ord[0..n-1] = population indices, best fitness first (heapsort with
// a min-heap; O(n log n), no recursion, no extra memory).
// ------------------------------------------------------------
static inline void ga_sift_min(const float* pop, int16_t* ord, int root, int end) {
    while (2 * root + 1 < end) {
        int child = 2 * root + 1;
        if (child + 1 < end &&
            pop[ord[child + 1] * IND_WORDS + IND_FIT] < pop[ord[child] * IND_WORDS + IND_FIT])
            ++child;
        if (pop[ord[child] * IND_WORDS + IND_FIT] < pop[ord[root] * IND_WORDS + IND_FIT]) {
            const int16_t t = ord[root]; ord[root] = ord[child]; ord[child] = t;
            root = child;
        } else {
            return;
        }
    }
}

static inline void ga_order_desc(const float* pop, int16_t* ord, int n) {
    for (int i = 0; i < n; ++i) ord[i] = (int16_t)i;
    for (int s = n / 2 - 1; s >= 0; --s) ga_sift_min(pop, ord, s, n);
    for (int end = n - 1; end > 0; --end) {
        const int16_t t = ord[0]; ord[0] = ord[end]; ord[end] = t;
        ga_sift_min(pop, ord, 0, end);
    }
}

#endif
