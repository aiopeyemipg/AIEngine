
#ifndef GA_MATH_H
#define GA_MATH_H

// ============================================================
// ga_math.h  —  scalar float maths and random numbers for the
// AIE kernels (same code as PSO_AI_Engine/src/kernels/pso_math.h).
//
// The AIE1 core has no libm, so sin/cos/atan2/log/exp are written
// here as short polynomials (error < 1e-6 on the ranges used).
// sqrt uses the hardware square-root unit through aie::sqrt.
// The same code runs unchanged in the x86 simulator, and natively
// (-DGA_NATIVE, bench/ga_bench.cpp) with std::sqrt.
// ============================================================

#include <stdint.h>
#ifdef GA_NATIVE
#include <cmath>
#else
#include <aie_api/aie.hpp>
#endif

static constexpr float GA_PI       = 3.14159265358979f;
static constexpr float GA_TWO_PI   = 6.28318530717959f;
static constexpr float GA_HALF_PI  = 1.57079632679490f;
static constexpr float GA_LN2      = 0.693147180559945f;
static constexpr float GA_NEG_INF  = -1.0e30f;   // log(0)

static inline float ga_abs(float x)       { return x < 0.0f ? -x : x; }

#ifdef GA_NATIVE
static inline float ga_sqrt(float x) { return x > 0.0f ? std::sqrt(x) : 0.0f; }
#else
static inline float ga_sqrt(float x) { return x > 0.0f ? (float)aie::sqrt(x) : 0.0f; }
#endif

// floor for |x| < 2^31 (larger values are already integral in float)
static inline float ga_floor(float x) {
    if (x >= 2.0e9f || x <= -2.0e9f) return x;
    float t = (float)(int)x;
    return (t > x) ? t - 1.0f : t;
}

// round half away from zero (std::round)
static inline float ga_round(float x) {
    return x >= 0.0f ? ga_floor(x + 0.5f) : -ga_floor(-x + 0.5f);
}

// x mod pi, result in [0, pi)
static inline float ga_wrap_pi(float x) {
    float r = x - GA_PI * ga_floor(x / GA_PI);
    if (r >= GA_PI) r -= GA_PI;
    if (r < 0.0f)    r = 0.0f;
    return r;
}

// sine and cosine of x (radians)
static inline void ga_sincos(float x, float& s, float& c) {
    x -= GA_TWO_PI * ga_round(x * (1.0f / GA_TWO_PI));       // [-pi, pi]
    float sign = 1.0f;
    if (x > GA_HALF_PI)       { x =  GA_PI - x; sign = -1.0f; } // sin kept, cos flips
    else if (x < -GA_HALF_PI) { x = -GA_PI - x; sign = -1.0f; }
    float x2 = x * x;
    s = x * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f + x2 * (-1.0f / 5040.0f
          + x2 * (1.0f / 362880.0f + x2 * (-1.0f / 39916800.0f))))));
    c = 1.0f + x2 * (-0.5f + x2 * (1.0f / 24.0f + x2 * (-1.0f / 720.0f
          + x2 * (1.0f / 40320.0f + x2 * (-1.0f / 3628800.0f + x2 * (1.0f / 479001600.0f))))));
    c *= sign;
}

// atan(z) for 0 <= z <= 1 (Abramowitz & Stegun 4.4.49, |err| < 2e-8)
static inline float ga_atan01(float z) {
    float z2 = z * z;
    return z * (1.0f + z2 * (-0.3333314528f + z2 * (0.1999355085f + z2 * (-0.1420889944f
             + z2 * (0.1065626393f + z2 * (-0.0752896400f + z2 * (0.0429096138f
             + z2 * (-0.0161657367f + z2 * 0.0028662257f))))))));
}

static inline float ga_atan2(float y, float x) {
    float ax = ga_abs(x), ay = ga_abs(y);
    if (ax == 0.0f && ay == 0.0f) return 0.0f;
    float r = (ay > ax) ? GA_HALF_PI - ga_atan01(ax / ay) : ga_atan01(ay / ax);
    if (x < 0.0f) r = GA_PI - r;
    return (y < 0.0f) ? -r : r;
}

// natural log, x > 0
static inline float ga_log(float x) {
    if (!(x > 0.0f)) return GA_NEG_INF;
    int e = 0;
    while (x >= 1.41421356f) { x *= 0.5f; ++e; }
    while (x <  0.70710678f) { x *= 2.0f; --e; }
    float s = (x - 1.0f) / (x + 1.0f), s2 = s * s;
    float l = 2.0f * s * (1.0f + s2 * (1.0f / 3.0f + s2 * (1.0f / 5.0f + s2 * (1.0f / 7.0f + s2 * (1.0f / 9.0f)))));
    return l + (float)e * GA_LN2;
}

static inline float ga_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x >  88.0f) x = 88.0f;
    float k = ga_round(x * (1.0f / GA_LN2));
    float r = x - k * GA_LN2;                                  // |r| <= ln2 / 2
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6.0f + r * (1.0f / 24.0f
            + r * (1.0f / 120.0f + r * (1.0f / 720.0f + r * (1.0f / 5040.0f)))))));
    int ki = (int)k;
    if (ki > 0) for (int i = 0; i < ki; ++i)  p *= 2.0f;
    else        for (int i = 0; i < -ki; ++i) p *= 0.5f;
    return p;
}

// ------------------------------------------------------------
// Random numbers: xoshiro128** seeded through splitmix32.
// ------------------------------------------------------------
class GaRng {
public:
    explicit GaRng(uint32_t seed) {
        for (int i = 0; i < 4; ++i) {
            seed += 0x9E3779B9u;
            uint32_t z = seed;
            z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
            z = (z ^ (z >> 13)) * 0xC2B2AE35u;
            s_[i] = z ^ (z >> 16);
        }
        if ((s_[0] | s_[1] | s_[2] | s_[3]) == 0u) s_[0] = 1u;
        hasSpare_ = false;
        spare_ = 0.0f;
    }

    uint32_t next() {
        uint32_t result = rotl(s_[1] * 5u, 7) * 9u;
        uint32_t t = s_[1] << 9;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 11);
        return result;
    }

    // uniform in [0, 1)
    float u01() { return (float)(next() >> 8) * (1.0f / 16777216.0f); }

    // uniform in [lo, hi)   (std::uniform_real_distribution / random.uniform)
    float uniform(float lo, float hi) { return lo + (hi - lo) * u01(); }

    // standard normal (Box-Muller)
    float gauss() {
        if (hasSpare_) { hasSpare_ = false; return spare_; }
        float u1 = (float)((next() >> 8) + 1u) * (1.0f / 16777216.0f);   // (0, 1]
        float u2 = u01();
        float r = ga_sqrt(-2.0f * ga_log(u1));
        float s, c;
        ga_sincos(GA_TWO_PI * u2, s, c);
        spare_ = r * s;
        hasSpare_ = true;
        return r * c;
    }

    // N(mu, sigma^2)   (std::normal_distribution / random.gauss)
    float gauss(float mu, float sigma) { return mu + sigma * gauss(); }

private:
    static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
    uint32_t s_[4];
    bool     hasSpare_;
    float    spare_;
};

// Independent random stream for one parallel task: a counter-based
// seed from (detection seed, a, b), e.g. (generation, child pair). Each
// task draws only from its own stream, so the result does not depend on
// which worker (tile / thread) runs it or in what order.
static inline uint32_t ga_mix32(uint32_t z) {
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}
static inline uint32_t ga_stream(uint32_t seed, uint32_t a, uint32_t b) {
    return ga_mix32(ga_mix32(seed ^ ga_mix32(a + 0x9E3779B9u)) + b * 0x632BE5ABu);
}

// Same per-image / per-variant seed as runAll.cpp:
//   seed = imageID * 7919 + vi * 104729 + 1
static inline uint32_t ga_seed(int imageID, int variantIndex) {
    return (uint32_t)imageID * 7919u + (uint32_t)variantIndex * 104729u + 1u;
}

#endif
