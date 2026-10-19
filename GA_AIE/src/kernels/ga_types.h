#ifndef GA_TYPES_H
#define GA_TYPES_H

// ============================================================
// ga_types.h  —  plain C++ data types of the GA detectors
//
// No adf / aie_api dependency: the graph (kernels.h / project.h)
// includes it to size the kernel state, and the native bench
// (bench/ga_bench.cpp) uses the same structs.
//
// An individual is a float[IND_WORDS] = {cx, cy, a, b, theta, fitness}.
// ============================================================

#include <stdint.h>
#include "include.h"

#define IND_WORDS 6     // 5 genes + fitness
#define IND_FIT   5     // index of the fitness word

// Edge points of one image in structure-of-arrays form, padded to a
// multiple of 8 so the fitness loop can run on 8-lane float vectors.
struct EdgeSet {
    int n;      // real edge points
    int nPad;   // n rounded up to a multiple of 8
    alignas(32) float px[MAX_EDGES];
    alignas(32) float py[MAX_EDGES];
    alignas(32) float boosted[MAX_EDGES];  // intensity^2
    alignas(32) float bright[MAX_EDGES];   // 1 if intensity >= HIGH_INTENSITY_THRESHOLD
};

// Outcome of one detection = the fields of one result record.
struct GaResult {
    int   variant;
    int   image;
    int   detected;
    float ell[5];      // cx, cy, a, b, theta (image space, 20x20)
    float fitness;
    int   evals;
    int   gens;
    int   flags;
    float hist[GA_HIST_MAX];   // best-so-far fitness per generation (ga_hist_len(vi) used)
};

// History length of variant vi (include.h).
static inline int ga_hist_len(int vi) {
    return vi == VI_CONV ? CONV_HIST_LEN : vi == VI_MICRO ? MICRO_HIST_LEN : ISL_HIST_LEN;
}

// ---- master-side state of one detection ----

// Conventional GA. The two population arrays (19.2 KB each) are not
// part of the state: on the AIE they are graph-side kernel parameters.
struct ConvGaState {
    int16_t order[CONV_POP];      // population sorted by fitness (indices)
};

struct MicroGaState {
    float pop[MICRO_POP][IND_WORDS];
    float next[MICRO_POP][IND_WORDS];
};

// ---- Island GA: one island per worker ----

// One island, held by its worker between epochs.
struct GaIsland {
    float pop[ISL_POP][IND_WORDS];
    float next[ISL_POP][IND_WORDS];
    float hist[ISL_MIG_INTERVAL];
};

// What the master sends to every island for one epoch.
struct IslEpochCmd {
    int   restart;
    int   epoch;
    int   init;                // 1 = build the island (first epoch of a cycle)
    int   haveSeed;            // warm seed for init (best of earlier cycles)
    float seed[5];
    int   migrate;             // 1 = take migrants[island] before evolving
    float migrants[ISL_ISLANDS][ISL_MIG_SIZE][IND_WORDS];
};
#define ISL_CMD_WORDS (10 + ISL_ISLANDS * ISL_MIG_SIZE * IND_WORDS)

// What every island answers after the epoch: its ISL_MIG_SIZE best
// (the emigrants of the next migration; top[0] = island best).
struct IslEpochReply {
    int   evals;
    int   gens;
    float top[ISL_MIG_SIZE][IND_WORDS];
    float hist[ISL_MIG_INTERVAL];   // island best after each generation of the epoch
};

struct IslandGaState {
    IslEpochCmd   cmd;
    IslEpochReply rep[ISL_ISLANDS];
};

// Master state of the variant built in.
#if GA_VARIANT == VI_CONV
typedef ConvGaState GaMasterState;
#elif GA_VARIANT == VI_MICRO
typedef MicroGaState GaMasterState;
#else
typedef IslandGaState GaMasterState;
#endif

#endif
