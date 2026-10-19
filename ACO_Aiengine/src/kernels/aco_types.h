#ifndef ACO_TYPES_H
#define ACO_TYPES_H

// ============================================================
// aco_types.h  —  data structures of the ACO detectors and the
// random number generator. Plain C++ (no AIE API), so the graph
// (kernels.h / project.h) can include it to size the kernel state.
// ============================================================

#include <stdint.h>
#include "include.h"

// ------------------------------------------------------------
// Random numbers: xoshiro128** seeded through splitmix32
// (same generator as GA_AIE / PSO_AI_Engine).
// ------------------------------------------------------------
class AcoRng {
public:
    AcoRng() : AcoRng(1u) {}
    explicit AcoRng(uint32_t seed) {
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

    // uniform in [0, 1)
    float u01() { return (float)(next() >> 8) * (1.0f / 16777216.0f); }

private:
    static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
    uint32_t s_[4];
};

// Hit pixels of one image in structure-of-arrays form, padded to a
// multiple of 8 for the vector fitness loop.
struct AcoEdgeSet {
    int n;      // real hit pixels
    int nPad;   // n rounded up to a multiple of 8
    alignas(32) float px[MAX_EDGES];
    alignas(32) float py[MAX_EDGES];
    alignas(32) float boosted[MAX_EDGES];  // intensity^2 (fitness and moment weight)
    alignas(32) float bright[MAX_EDGES];   // 1 if intensity >= HIGH_INTENSITY_THRESHOLD
};

// Fitness of an invalid candidate (the reference's -inf).
#define ACO_INVALID_FIT (-1.0e30f)
static inline bool aco_valid_fit(float f) { return f > -1.0e29f; }

struct AcoSeed { float h0, k0, a0, b0, theta0; };

// Per-dimension grids (h, k, a, b, theta) and normalised heuristics.
struct AcoSpace {
    float grid[5][ACO_BINS];
    float eta[5][ACO_BINS];
};

// One FPED pheromone trail = one parameter dimension (one worker).
struct FpedTrail {
    float grid[ACO_BINS];
    float etaPow[ACO_BINS];
    float tau[ACO_BINS];
    float cdf[ACO_BINS];
    float tmp[ACO_BINS];
    float kern[2 * FPED_SMOOTH_RAD + 1];   // Gaussian smoothing kernel
    int   wrap;                            // periodic trail (theta before refinement)
    AcoRng rng;                            // this trail's random stream
};

// What one trail needs from the master to finish an iteration:
// the bins of the ranked ants, the global-best bin and the refinement.
#define FPED_UPDATE_WORDS (4 + FPED_ELITE)
struct FpedUpdate {
    int   ne;                    // ranked ants that deposit (<= FPED_ELITE)
    int   bestBin;               // global-best bin, -1 = none
    int   refine;                // 1 = rebuild the grid around `centre`
    float centre;                // incumbent best value of this parameter
    int   bins[FPED_ELITE];      // bin of the ant of rank r
};

// Master-side working memory of one FPED detection.
struct FpedState {
    int16_t bins[5][FPED_ANTS];            // bin chosen by every ant, per trail
    float vals[5][FPED_ANTS];              // the corresponding grid values
    float cand[FPED_ANTS][5];              // the ants of one iteration
    float fit[FPED_ANTS];                  // their fitness
    float scoredFit[FPED_ANTS];
    int16_t scoredAnt[FPED_ANTS];
    FpedUpdate up[5];
};

// Working memory of one Standard ACO detector (master side).
struct StdAcoState {
    AcoSpace sp;
    float cdf[STD_TUPLES];                 // also the heuristic cdfs (first ACO_BINS)
    float cand[STD_TUPLES][5];             // fixed joint archive
    float cacheFit[STD_TUPLES];
    int8_t cacheState[STD_TUPLES];         // 0 not scored, 1 scored
    float tau[STD_TUPLES];
    int16_t pick[STD_ANTS];                // tuple chosen by each ant
    int16_t batchIdx[STD_ANTS];            // unscored tuples of one iteration
    float batchCand[STD_ANTS][5];
    float batchFit[STD_ANTS];
    float scoredFit[STD_ANTS];
    int16_t scoredIdx[STD_ANTS];
};

struct AcoResult {
    int   image;       // -1 = padding record
    int   detected;
    float ell[5];
    float fitness;
    int   evals;
    int   iters;
    int   flags;
    float hist[ACO_HIST_LEN];   // best-so-far fitness after each colony iteration
};

// Master state of the algorithm built in.
#if ACO_VARIANT == VI_FPED
typedef FpedState AcoMasterState;
#else
typedef StdAcoState AcoMasterState;
#endif

#endif
