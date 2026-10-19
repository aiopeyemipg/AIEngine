#ifndef ABC_TYPES_H
#define ABC_TYPES_H

// ============================================================
// abc_types.h  —  data structures of the ABC detectors and the
// random number generator. Plain C++ (no AIE API), so the graph
// (kernels.h / project.h) can include it to size the kernel state.
// ============================================================

#include <stdint.h>
#include "include.h"

// ------------------------------------------------------------
// Random numbers: xoshiro128** seeded through splitmix32
// (same generator as the ACO, GA and PSO ports).
// ------------------------------------------------------------
class AbcRng {
public:
    AbcRng() : AbcRng(1u) {}
    explicit AbcRng(uint32_t seed) {
        for (int i = 0; i < 4; ++i) {
            seed += 0x9E3779B9u;
            uint32_t z = seed;
            z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
            z = (z ^ (z >> 13)) * 0xC2B2AE35u;
            s_[i] = z ^ (z >> 16);
        }
        if ((s_[0] | s_[1] | s_[2] | s_[3]) == 0u) s_[0] = 1u;
    }

    uint32_t next() {
        uint32_t result = rotl(s_[1] * 5u, 7) * 9u;
        uint32_t t = s_[1] << 9;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 11);
        return result;
    }

    // uniform in [0, 1)          (random.random)
    float u01() { return (float)(next() >> 8) * (1.0f / 16777216.0f); }
    // uniform in [lo, hi)        (random.uniform)
    float uniform(float lo, float hi) { return lo + (hi - lo) * u01(); }
    // integer in [0, n - 1]      (random.randint(0, n - 1))
    int below(int n) { int r = (int)(u01() * (float)n); return r < n ? r : n - 1; }

private:
    static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
    uint32_t s_[4];
};

// Fitness of an invalid bee (HA-ABC's -inf).
#define ABC_INVALID_FIT (-1.0e30f)
static inline bool abc_valid_fit(float f) { return f > -1.0e29f; }

// Input points of one image, structure of arrays, padded to a multiple of 8.
//   HA-ABC: the lit pixels, w = intensity / max intensity of the frame
//   SR-ABC: the Canny edge points, w = 1 (0 for the padding)
struct AbcPoints {
    int n;
    int nPad;
    alignas(32) float px[MAX_POINTS];
    alignas(32) float py[MAX_POINTS];
    alignas(32) float w[MAX_POINTS];
};

// The four kinds of parallel work of one ABC iteration (one task = one bee).
#define ABC_TASK_INIT      0     // random bee (+ HA restarts while invalid)
#define ABC_TASK_EMPLOYED  1     // bee t perturbed against a random partner
#define ABC_TASK_ONLOOKER  2     // onlooker t perturbs bee sel[t]
#define ABC_TASK_SCOUT     3     // bee t replaced by a random bee if sel[t] = 1
#define ABC_STREAM_SELECT  4     // random stream of the onlooker roulette (master)

// What every worker needs for one phase.
struct AbcPhase {
    int   kind;                  // ABC_TASK_*
    int   iter;                  // ABC iteration (-1 for the initialisation)
    float bees[NUM_BEES][5];     // colony at the start of the phase
    int   sel[NUM_BEES];         // ONLOOKER: followed bee; SCOUT: 1 = replace
};

// Master-side state of one detection.
struct AbcState {
    float bees[NUM_BEES][5];
    float fit[NUM_BEES];
    int   trials[NUM_BEES];
    float cand[NUM_BEES][5];     // results of one phase
    float candFit[NUM_BEES];
    int   candEvals[NUM_BEES];
    float cumul[NUM_BEES];
    AbcPhase msg;
};

struct AbcResult {
    int   image;
    int   detected;
    float ell[5];
    float fitness;
    int   evals;
    int   iters;
    int   flags;
    float hist[ABC_HIST_LEN];    // best-so-far fitness after each ABC iteration
};

#endif
