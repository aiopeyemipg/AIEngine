#ifndef ABC_CORE_H
#define ABC_CORE_H

// ============================================================
// abc_core.h  —  the two ABC ellipse detectors, one image per call,
// with the bees of every ABC iteration processed in parallel
//
//   abc_load_points  image record -> AbcPoints (HA: lit pixels,
//                    SR: Canny edge points, abc_canny.h)
//   abc_fitness_ha   hollowness fitness       (hollowness-aware-ABC.py)
//   abc_fitness_sr   residual fitness         (standard-residual-ABC.py)
//   abc_task         one bee of one phase: build the candidate, score it
//   abc_detect       the colony: init -> per iteration employed,
//                    onlooker, scout phases -> global best
//
// Concurrency inside one image. The ABC iterations are sequential,
// but every phase of an iteration is a set of independent bee tasks:
//   employed : bee i perturbs one parameter against a random partner
//   onlooker : onlooker o perturbs the bee it chose on the roulette
//   scout    : exhausted bees are replaced by random bees
// Every task runs on its own worker (AIE tile / thread) against the
// colony as it was at the start of the phase; the master then applies
// the greedy acceptance in bee / onlooker order. This is the usual
// synchronous form of a parallel ABC. The Python scripts update the
// colony bee by bee instead (a later bee can see an earlier bee's new
// position); bench/reference_cpp.cpp keeps that sequential form.
//
// Every task has its own random stream, keyed by (image seed, iteration,
// phase, task), so serial, threaded and AIE runs give bit-identical
// detections whatever the number of workers.
//
// An Engine provides
//   void begin(const AbcPoints& pts, uint32_t seed);          // new image
//   void phase(const AbcPhase& m, int nTasks, float (*cand)[5],
//              float* fit, int* evals);                        // tasks 0..n-1
// ============================================================

#include <stdint.h>
#include "include.h"
#include "abc_types.h"
#include "abc_math.h"
#include "abc_canny.h"

// Scalar fitness loops natively and in x86sim; 8-lane float vectors on
// the AI Engine (-DABC_X86SIM_VECTOR_FITNESS simulates them in x86sim).
#if defined(ABC_NATIVE) || (defined(__X86SIM__) && !defined(ABC_X86SIM_VECTOR_FITNESS))
#define ABC_SCALAR_FITNESS 1
#endif


static inline float abc_min(float a, float b) { return a < b ? a : b; }
static inline float abc_max(float a, float b) { return a > b ? a : b; }

// ------------------------------------------------------------
// Input points of one frame.
//   HA: every pixel > 0 with intensity / (max pixel of the frame)
//       (load_points_with_intensity)
//   SR: Canny edge points of the 4x upscaled frame, or the lit pixels
//       when Canny finds fewer than 5 (extract_points)
// Returns the flags (RES_FLAGS bits 0 and 1).
// ------------------------------------------------------------
static inline int abc_load_points(const uint8_t* pix, int vi, uint8_t* blur, int16_t* mag, int8_t* dir,
                                  AbcPoints& p) {
    int flags = 0, n = 0;
    if (vi == VI_HA) {
        int maxv = 0;
        for (int i = 0; i < IMG_PIXELS; ++i) if (pix[i] > maxv) maxv = pix[i];
        const float inv = 1.0f / (float)(maxv > 0 ? maxv : 1);
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x) {
                const int g = pix[y * IMG_W + x];
                if (g == 0) continue;
                if (n == MAX_POINTS) { flags |= 1; continue; }
                p.px[n] = (float)x; p.py[n] = (float)y; p.w[n] = (float)g * inv;
                ++n;
            }
    } else {
        int fallback = 0;
        n = canny_points(pix, blur, mag, dir, p.px, p.py, MAX_POINTS, &fallback);
        if (n > MAX_POINTS) { flags |= 1; n = MAX_POINTS; }
        if (fallback) flags |= 2;
        for (int i = 0; i < n; ++i) p.w[i] = 1.0f;
    }
    p.n = n;
    p.nPad = (n + 7) & ~7;
    for (int i = n; i < p.nPad; ++i) { p.px[i] = 1.0e4f; p.py[i] = 1.0e4f; p.w[i] = 0.0f; }
    return flags;
}

// ------------------------------------------------------------
// HA-ABC fitness (hollowness_fitness): for every lit pixel
//   |E - 1| < 0.35 -> perimeter += I,  E < 0.65 -> interior += I
//   fitness = perimeter - 0.5 * interior
// Invalid (-inf): an axis < 0.5 px, bounding box outside the frame,
// perimeter count outside [3, 500].
// ------------------------------------------------------------
static inline float abc_fitness_ha(const AbcPoints& p, const float* e) {
    const float h = e[0], k = e[1], a = e[2], b = e[3];
    if (a < HA_MIN_AXIS || b < HA_MIN_AXIS) return ABC_INVALID_FIT;
    float st, ct;
    abc_sincos(e[4], st, ct);
    const float bw = abc_sqrt((a * ct) * (a * ct) + (b * st) * (b * st));
    const float bh = abc_sqrt((a * st) * (a * st) + (b * ct) * (b * ct));
    if (h - bw < 0.0f || h + bw > (float)IMG_W || k - bh < 0.0f || k + bh > (float)IMG_H)
        return ABC_INVALID_FIT;
    const float ia2 = 1.0f / (a * a), ib2 = 1.0f / (b * b);
#ifdef ABC_SCALAR_FITNESS
    // 8 partial sums, one per lane of the AIE vector loop below
    // (padding points are far outside: never on the rim or inside)
    float perL[8] = { 0 }, inL[8] = { 0 }, cntL[8] = { 0 };
    for (int i0 = 0; i0 < p.nPad; i0 += 8)
        for (int l = 0; l < 8; ++l) {
            const int i = i0 + l;
            const float dx = p.px[i] - h, dy = p.py[i] - k;
            const float t1 = dx * ct + dy * st, t2 = dx * st - dy * ct;
            const float eq = t1 * t1 * ia2 + t2 * t2 * ib2;
            const float rim = abc_abs(eq - 1.0f) < HA_INLIER_TOLERANCE ? 1.0f : 0.0f;
            const float in  = eq < 1.0f - HA_INLIER_TOLERANCE ? 1.0f : 0.0f;  // disjoint from rim
            perL[l] += rim * p.w[i];
            inL[l]  += in * p.w[i];
            cntL[l] += rim;
        }
    float per = 0.0f, inside = 0.0f, cnt = 0.0f;
    for (int l = 0; l < 8; ++l) { per += perL[l]; inside += inL[l]; cnt += cntL[l]; }
    const int nPer = (int)(cnt + 0.5f);
#else
    const aie::vector<float, 8> zero = aie::zeros<float, 8>();
    const aie::vector<float, 8> one  = aie::broadcast<float, 8>(1.0f);
    const aie::vector<float, 8> tol  = aie::broadcast<float, 8>(HA_INLIER_TOLERANCE);
    const aie::vector<float, 8> lim  = aie::broadcast<float, 8>(1.0f - HA_INLIER_TOLERANCE);
    aie::vector<float, 8> perV = zero, inV = zero, cntV = zero;
    for (int i = 0; i < p.nPad; i += 8)
        chess_prepare_for_pipelining
    {
        aie::vector<float, 8> dx = aie::sub(aie::load_v<8>(p.px + i), h);
        aie::vector<float, 8> dy = aie::sub(aie::load_v<8>(p.py + i), k);
        aie::vector<float, 8> t1 = aie::mac(aie::mul(dx, ct), dy, st).template to_vector<float>();
        aie::vector<float, 8> t2 = aie::msc(aie::mul(dx, st), dy, ct).template to_vector<float>();
        aie::vector<float, 8> eq = aie::mac(aie::mul(aie::mul(t1, t1).template to_vector<float>(), ia2),
                                            aie::mul(t2, t2).template to_vector<float>(), ib2).template to_vector<float>();
        aie::mask<8> rim = aie::lt(aie::abs(aie::sub(eq, 1.0f)), tol);
        aie::mask<8> in  = aie::lt(eq, lim);
        aie::vector<float, 8> W = aie::load_v<8>(p.w + i);
        perV = aie::add(perV, aie::select(zero, W, rim));
        inV  = aie::add(inV,  aie::select(zero, W, in));
        // padding points (w = 0, far away) are never on the rim
        cntV = aie::add(cntV, aie::select(zero, one, rim));
    }
    const float per = aie::reduce_add(perV), inside = aie::reduce_add(inV);
    const int nPer = (int)(aie::reduce_add(cntV) + 0.5f);
#endif
    if (nPer < HA_MIN_INLIERS || nPer > HA_MAX_INLIERS) return ABC_INVALID_FIT;
    return per - HA_HOLLOWNESS * inside;
}

// ------------------------------------------------------------
// SR-ABC fitness (compute_fitness): 1 / (mean |E - 1| + 1e-6) over the
// edge points; 0 for a degenerate ellipse.
// ------------------------------------------------------------
static inline float abc_fitness_sr(const AbcPoints& p, const float* e) {
    const float h = e[0], k = e[1], a = e[2], b = e[3];
    if (a <= 0.0f || b <= 0.0f || p.n == 0) return 0.0f;
    float st, ct;
    abc_sincos(e[4], st, ct);
    const float ia2 = 1.0f / (a * a), ib2 = 1.0f / (b * b);
#ifdef ABC_SCALAR_FITNESS
    // 8 partial sums, one per lane of the AIE vector loop below
    // (w = 0 masks the padding points)
    float errL[8] = { 0 };
    for (int i0 = 0; i0 < p.nPad; i0 += 8)
        for (int l = 0; l < 8; ++l) {
            const int i = i0 + l;
            const float dx = p.px[i] - h, dy = p.py[i] - k;
            const float xr = dx * ct + dy * st, yr = -dx * st + dy * ct;
            errL[l] += abc_abs(xr * xr * ia2 + yr * yr * ib2 - 1.0f) * p.w[i];
        }
    float err = 0.0f;
    for (int l = 0; l < 8; ++l) err += errL[l];
#else
    aie::vector<float, 8> errV = aie::zeros<float, 8>();
    for (int i = 0; i < p.nPad; i += 8)
        chess_prepare_for_pipelining
    {
        aie::vector<float, 8> dx = aie::sub(aie::load_v<8>(p.px + i), h);
        aie::vector<float, 8> dy = aie::sub(aie::load_v<8>(p.py + i), k);
        aie::vector<float, 8> xr = aie::mac(aie::mul(dx, ct), dy, st).template to_vector<float>();
        aie::vector<float, 8> yr = aie::msc(aie::mul(dy, ct), dx, st).template to_vector<float>();
        aie::vector<float, 8> E  = aie::mac(aie::mul(aie::mul(xr, xr).template to_vector<float>(), ia2),
                                            aie::mul(yr, yr).template to_vector<float>(), ib2).template to_vector<float>();
        // w = 0 masks the padding points
        errV = aie::mac(errV, aie::abs(aie::sub(E, 1.0f)), aie::load_v<8>(p.w + i)).template to_vector<float>();
    }
    const float err = aie::reduce_add(errV);
#endif
    return 1.0f / (err / (float)p.n + SR_EPS);
}

static inline float abc_fitness(const AbcPoints& p, int vi, const float* e) {
    return vi == VI_HA ? abc_fitness_ha(p, e) : abc_fitness_sr(p, e);
}

// random_bee: cx, cy ~ U(1, size - 1), a, b ~ U(axisMin, size / 2), theta ~ U(0, 2 pi)
static inline void abc_random_bee(AbcRng& r, int vi, float* e) {
    const float axMin = vi == VI_HA ? HA_BEE_AXIS_MIN : SR_BEE_AXIS_MIN;
    e[0] = r.uniform(1.0f, (float)IMG_W - 1.0f);
    e[1] = r.uniform(1.0f, (float)IMG_H - 1.0f);
    e[2] = r.uniform(axMin, (float)IMG_W / 2.0f);
    e[3] = r.uniform(axMin, (float)IMG_H / 2.0f);
    e[4] = r.uniform(0.0f, ABC_TWO_PI);
}

// clamp_bee: centre in [0.5, size - 0.5], axes in [0.5, size / 2], theta mod 2 pi
static inline void abc_clamp_bee(float* e) {
    e[0] = abc_max(0.5f, abc_min((float)IMG_W - 0.5f, e[0]));
    e[1] = abc_max(0.5f, abc_min((float)IMG_H - 0.5f, e[1]));
    e[2] = abc_max(0.5f, abc_min((float)IMG_W / 2.0f, e[2]));
    e[3] = abc_max(0.5f, abc_min((float)IMG_H / 2.0f, e[3]));
    float t = e[4] - ABC_TWO_PI * abc_floor(e[4] / ABC_TWO_PI);
    if (t >= ABC_TWO_PI) t -= ABC_TWO_PI;
    if (t < 0.0f) t = 0.0f;
    e[4] = t;
}

// Random stream of task t of a phase: (image seed, iteration, phase, task)
static inline uint32_t abc_stream(uint32_t seed, int iter, int kind, int t) {
    return seed * 0x9E3779B1u + (uint32_t)(iter + 2) * 0x85EBCA77u
         + (uint32_t)(kind + 1) * 0xC2B2AE3Du + (uint32_t)(t + 1) * 0x27D4EB2Fu;
}

// ------------------------------------------------------------
// One bee task of one phase (runs on a worker). Writes the candidate,
// its fitness and the number of fitness evaluations (0 = nothing done).
// ------------------------------------------------------------
static inline void abc_task(const AbcPoints& p, int vi, uint32_t seed, const AbcPhase& m, int t,
                            float* cand, float& fit, int& evals) {
    AbcRng r(abc_stream(seed, m.iter, m.kind, t));
    evals = 0;
    if (m.kind == ABC_TASK_INIT || m.kind == ABC_TASK_SCOUT) {
        if (m.kind == ABC_TASK_SCOUT && !m.sel[t]) return;
        abc_random_bee(r, vi, cand);
        fit = abc_fitness(p, vi, cand);
        evals = 1;
        // HA initialisation restart loop: re-draw an invalid bee up to 50 times
        if (vi == VI_HA && m.kind == ABC_TASK_INIT)
            for (int att = 0; !abc_valid_fit(fit) && att < HA_INIT_RESTARTS; ++att) {
                abc_random_bee(r, vi, cand);
                fit = abc_fitness(p, vi, cand);
                ++evals;
            }
        return;
    }
    // employed bee t, or onlooker t following bee sel[t]
    const int i = (m.kind == ABC_TASK_EMPLOYED) ? t : m.sel[t];
    int k = r.below(NUM_BEES);
    while (k == i) k = r.below(NUM_BEES);
    const int j = r.below(5);
    const float phi = r.uniform(-1.0f, 1.0f);
    for (int d = 0; d < 5; ++d) cand[d] = m.bees[i][d];
    cand[j] += phi * (m.bees[i][j] - m.bees[k][j]);
    abc_clamp_bee(cand);
    fit = abc_fitness(p, vi, cand);
    evals = 1;
}

// Score tasks begin, begin+step, ... < n (one worker's share).
static inline void abc_task_range(const AbcPoints& p, int vi, uint32_t seed, const AbcPhase& m,
                                  int begin, int n, int step, float (*cand)[5], float* fit, int* evals) {
    for (int t = begin; t < n; t += step) abc_task(p, vi, seed, m, t, cand[t], fit[t], evals[t]);
}

// ============================================================
// The colony (run_abc). Per ABC iteration: employed phase (30 tasks),
// onlooker roulette + onlooker phase (30 tasks), scout phase, global
// best. HA uses the zero-floored roulette and its -inf guards; SR the
// plain fitness-proportional roulette.
// ============================================================
template <typename Engine>
static inline void abc_detect(const AbcPoints& p, int vi, uint32_t seed, AbcState& st,
                              Engine& eng, AbcResult& res) {
    const int N = NUM_BEES;
    AbcPhase& m = st.msg;
    int evals = 0;
    eng.begin(p, seed);

    // ---- initialise the colony (one task per bee) ----
    m.kind = ABC_TASK_INIT;
    m.iter = -1;
    eng.phase(m, N, st.cand, st.candFit, st.candEvals);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 5; ++d) st.bees[i][d] = st.cand[i][d];
        st.fit[i] = st.candFit[i];
        st.trials[i] = 0;
        evals += st.candEvals[i];
    }
    // global best among the valid bees (bee 0 as a placeholder otherwise)
    int bi = -1;
    for (int i = 0; i < N; ++i)
        if (abc_valid_fit(st.fit[i]) && (bi < 0 || st.fit[i] > st.fit[bi])) bi = i;
    float best[5];
    float bestFit = bi >= 0 ? st.fit[bi] : ABC_INVALID_FIT;
    for (int d = 0; d < 5; ++d) best[d] = st.bees[bi >= 0 ? bi : 0][d];

    for (int it = 0; it < MAX_ITERATIONS; ++it) {
        m.iter = it;

        // ---- employed phase: every bee at once ----
        m.kind = ABC_TASK_EMPLOYED;
        for (int i = 0; i < N; ++i) for (int d = 0; d < 5; ++d) m.bees[i][d] = st.bees[i][d];
        eng.phase(m, N, st.cand, st.candFit, st.candEvals);
        for (int i = 0; i < N; ++i) {
            evals += st.candEvals[i];
            if (st.candFit[i] > st.fit[i]) {
                for (int d = 0; d < 5; ++d) st.bees[i][d] = st.cand[i][d];
                st.fit[i] = st.candFit[i];
                st.trials[i] = 0;
            } else {
                ++st.trials[i];
            }
        }

        // ---- onlooker roulette (master) ----
        float total = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float f = (vi == VI_HA) ? abc_max(st.fit[i], 0.0f) : st.fit[i];
            total += f;
            st.cumul[i] = total;
        }
        total += 1.0e-12f;
        AbcRng sel(abc_stream(seed, it, ABC_STREAM_SELECT, 0));
        for (int o = 0; o < N; ++o) {
            const float r = sel.u01() * total;          // cumul / total >= r  <=>  cumul >= r * total
            int si = 0;
            while (si < N - 1 && st.cumul[si] < r) ++si;  // np.searchsorted(side='left')
            m.sel[o] = si;
        }

        // ---- onlooker phase: every onlooker at once, accepted in order ----
        m.kind = ABC_TASK_ONLOOKER;
        for (int i = 0; i < N; ++i) for (int d = 0; d < 5; ++d) m.bees[i][d] = st.bees[i][d];
        eng.phase(m, N, st.cand, st.candFit, st.candEvals);
        for (int o = 0; o < N; ++o) {
            const int si = m.sel[o];
            evals += st.candEvals[o];
            if (st.candFit[o] > st.fit[si]) {
                for (int d = 0; d < 5; ++d) st.bees[si][d] = st.cand[o][d];
                st.fit[si] = st.candFit[o];
                st.trials[si] = 0;
            } else {
                ++st.trials[si];
            }
        }

        // ---- scout phase: every exhausted bee at once ----
        int nScout = 0;
        for (int i = 0; i < N; ++i) { m.sel[i] = st.trials[i] >= SCOUT_LIMIT ? 1 : 0; nScout += m.sel[i]; }
        if (nScout > 0) {
            m.kind = ABC_TASK_SCOUT;
            eng.phase(m, N, st.cand, st.candFit, st.candEvals);
            for (int i = 0; i < N; ++i)
                if (m.sel[i]) {
                    for (int d = 0; d < 5; ++d) st.bees[i][d] = st.cand[i][d];
                    st.fit[i] = st.candFit[i];
                    st.trials[i] = 0;
                    evals += st.candEvals[i];
                }
        }

        // ---- global best ----
        int cb = -1;
        for (int i = 0; i < N; ++i)
            if (abc_valid_fit(st.fit[i]) && (cb < 0 || st.fit[i] > st.fit[cb])) cb = i;
        if (cb >= 0 && st.fit[cb] > bestFit) {
            bestFit = st.fit[cb];
            for (int d = 0; d < 5; ++d) best[d] = st.bees[cb][d];
        }
        // the scripts record 0.0 while no valid bee exists
        if (it < ABC_HIST_LEN) res.hist[it] = abc_valid_fit(bestFit) ? bestFit : 0.0f;
    }

    res.detected = abc_valid_fit(bestFit) ? 1 : 0;
    for (int d = 0; d < 5; ++d) res.ell[d] = best[d];
    res.fitness = abc_valid_fit(bestFit) ? bestFit : 0.0f;
    res.evals = evals;
    res.iters = MAX_ITERATIONS;
}

// ------------------------------------------------------------
// Detect the ellipse of one image record with the algorithm `vi`.
// Too few points (HA: < 3, SR: < 5) -> the scripts' centred fallback
// ellipse, flat history (flags bit 2).
// ------------------------------------------------------------
template <typename Engine>
static inline void abc_detect_image(const uint8_t* rec, int vi, AbcPoints& p,
                                    uint8_t* blur, int16_t* mag, int8_t* dir,
                                    AbcState& st, Engine& eng, AbcResult& res,
                                    uint32_t seedOffset = 0) {
    res.image = (int)rec[0] | ((int)rec[1] << 8);
    res.detected = 0; res.fitness = 0.0f; res.evals = 0; res.iters = 0;
    for (int i = 0; i < ABC_HIST_LEN; ++i) res.hist[i] = 0.0f;
    res.flags = abc_load_points(rec + IMG_HEADER_BYTES, vi, blur, mag, dir, p);
    const int minPts = vi == VI_HA ? HA_MIN_INLIERS : SR_MIN_POINTS;
    if (p.n < minPts) {
        res.flags |= 4;
        res.ell[0] = IMG_W / 2.0f; res.ell[1] = IMG_H / 2.0f;
        res.ell[2] = IMG_W / 4.0f; res.ell[3] = IMG_H / 4.0f; res.ell[4] = 0.0f;
        res.iters = MAX_ITERATIONS;
        return;
    }
    abc_detect(p, vi, abc_seed(res.image, vi) + seedOffset, st, eng, res);
}

#endif
