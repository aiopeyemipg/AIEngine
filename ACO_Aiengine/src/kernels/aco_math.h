
#ifndef ACO_MATH_H
#define ACO_MATH_H

// ============================================================
// aco_math.h  —  scalar float maths for the
// AIE kernels (same code as GA_AIE/src/kernels/ga_math.h).
//
// The AIE1 core has no libm, so sin/cos/atan2/log/exp are written
// here as short polynomials (error < 1e-6 on the ranges used).
// sqrt uses the hardware square-root unit through aie::sqrt.
// The random number generator is in aco_types.h.
// The same code runs unchanged in the x86 simulator, and natively
// (-DACO_NATIVE, bench/aco_bench.cpp) with std::sqrt.
// ============================================================

#include <stdint.h>
#ifdef ACO_NATIVE
#include <cmath>
#else
#include <aie_api/aie.hpp>
#endif

static constexpr float ACO_PI       = 3.14159265358979f;
static constexpr float ACO_TWO_PI   = 6.28318530717959f;
static constexpr float ACO_HALF_PI  = 1.57079632679490f;
static constexpr float ACO_LN2      = 0.693147180559945f;
static constexpr float ACO_NEG_INF  = -1.0e30f;   // log(0)

// |x|; natively std::fabs (a bit mask instead of a branch, same value)
#ifdef ACO_NATIVE
static inline float aco_abs(float x)       { return std::fabs(x); }
#else
static inline float aco_abs(float x)       { return x < 0.0f ? -x : x; }
#endif

#ifdef ACO_NATIVE
static inline float aco_sqrt(float x) { return x > 0.0f ? std::sqrt(x) : 0.0f; }
#else
static inline float aco_sqrt(float x) { return x > 0.0f ? (float)aie::sqrt(x) : 0.0f; }
#endif

// floor for |x| < 2^31 (larger values are already integral in float)
static inline float aco_floor(float x) {
    if (x >= 2.0e9f || x <= -2.0e9f) return x;
    float t = (float)(int)x;
    return (t > x) ? t - 1.0f : t;
}

// round half away from zero (std::round)
static inline float aco_round(float x) {
    return x >= 0.0f ? aco_floor(x + 0.5f) : -aco_floor(-x + 0.5f);
}

// x mod pi, result in [0, pi)
static inline float aco_wrap_pi(float x) {
    float r = x - ACO_PI * aco_floor(x / ACO_PI);
    if (r >= ACO_PI) r -= ACO_PI;
    if (r < 0.0f)    r = 0.0f;
    return r;
}

// sine and cosine of x (radians)
static inline void aco_sincos(float x, float& s, float& c) {
    x -= ACO_TWO_PI * aco_round(x * (1.0f / ACO_TWO_PI));       // [-pi, pi]
    float sign = 1.0f;
    if (x > ACO_HALF_PI)       { x =  ACO_PI - x; sign = -1.0f; } // sin kept, cos flips
    else if (x < -ACO_HALF_PI) { x = -ACO_PI - x; sign = -1.0f; }
    float x2 = x * x;
    s = x * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f + x2 * (-1.0f / 5040.0f
          + x2 * (1.0f / 362880.0f + x2 * (-1.0f / 39916800.0f))))));
    c = 1.0f + x2 * (-0.5f + x2 * (1.0f / 24.0f + x2 * (-1.0f / 720.0f
          + x2 * (1.0f / 40320.0f + x2 * (-1.0f / 3628800.0f + x2 * (1.0f / 479001600.0f))))));
    c *= sign;
}

// atan(z) for 0 <= z <= 1 (Abramowitz & Stegun 4.4.49, |err| < 2e-8)
static inline float aco_atan01(float z) {
    float z2 = z * z;
    return z * (1.0f + z2 * (-0.3333314528f + z2 * (0.1999355085f + z2 * (-0.1420889944f
             + z2 * (0.1065626393f + z2 * (-0.0752896400f + z2 * (0.0429096138f
             + z2 * (-0.0161657367f + z2 * 0.0028662257f))))))));
}

static inline float aco_atan2(float y, float x) {
    float ax = aco_abs(x), ay = aco_abs(y);
    if (ax == 0.0f && ay == 0.0f) return 0.0f;
    float r = (ay > ax) ? ACO_HALF_PI - aco_atan01(ax / ay) : aco_atan01(ay / ax);
    if (x < 0.0f) r = ACO_PI - r;
    return (y < 0.0f) ? -r : r;
}

// natural log, x > 0
static inline float aco_log(float x) {
    if (!(x > 0.0f)) return ACO_NEG_INF;
    int e = 0;
    while (x >= 1.41421356f) { x *= 0.5f; ++e; }
    while (x <  0.70710678f) { x *= 2.0f; --e; }
    float s = (x - 1.0f) / (x + 1.0f), s2 = s * s;
    float l = 2.0f * s * (1.0f + s2 * (1.0f / 3.0f + s2 * (1.0f / 5.0f + s2 * (1.0f / 7.0f + s2 * (1.0f / 9.0f)))));
    return l + (float)e * ACO_LN2;
}

static inline float aco_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x >  88.0f) x = 88.0f;
    float k = aco_round(x * (1.0f / ACO_LN2));
    float r = x - k * ACO_LN2;                                  // |r| <= ln2 / 2
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6.0f + r * (1.0f / 24.0f
            + r * (1.0f / 120.0f + r * (1.0f / 720.0f + r * (1.0f / 5040.0f)))))));
    int ki = (int)k;
    if (ki > 0) for (int i = 0; i < ki; ++i)  p *= 2.0f;
    else        for (int i = 0; i < -ki; ++i) p *= 0.5f;
    return p;
}


// Same per-image / per-variant seed as ACO runAll.cpp:
//   seed = imageID * 7919 + vi * 104729 + 1
static inline uint32_t aco_seed(int imageID, int variantIndex) {
    return (uint32_t)imageID * 7919u + (uint32_t)variantIndex * 104729u + 1u;
}

#endif
