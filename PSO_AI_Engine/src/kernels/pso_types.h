#ifndef PSO_TYPES_H
#define PSO_TYPES_H

// ============================================================
// pso_types.h  —  plain data types of one PSO detection
//
// No ADF types and no AIE API here, so the graph (kernels.h) can
// include it to size the kernel state. The same structs are used by
// the AIE kernels (master tile, worker tiles) and by the native bench
// (bench/pso_bench.cpp, one shard per thread).
// ============================================================

#include <stdint.h>
#include "include.h"

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

// Scratch of the edge extraction + DBSCAN (master only).
struct EdgeScratch {
    int32_t raw[MAX_EDGES];                // x | y << 8 | gray << 16
    uint8_t core[MAX_EDGES];               // DBSCAN core-point flags
};

// Particles owned by one worker (particle i on worker i % W). Arrays
// are indexed by the global particle index; a worker touches only its own.
struct PsoShard {
    float pos[PSO_MAX_PARTICLES][5];
    float vel[PSO_MAX_PARTICLES][5];
    float pbest[PSO_MAX_PARTICLES][5];
    float pbestScore[PSO_MAX_PARTICLES];
};

// Best particle of one worker after a step (first index on ties).
struct PsoBest {
    int   idx;        // -1 = none
    float score;
    float pos[5];
};

// Master-side working memory of one detection.
struct PsoMasterState {
    EdgeSet     es;
    EdgeScratch scratch;
    float   pos[PSO_MAX_PARTICLES][5];         // initial swarm
    float   score[PSO_MAX_PARTICLES];          // its fitness
    // SP-PSO ring table (all personal bests) and its update
    float   tabP[PSO_MAX_PARTICLES][5];
    float   tabS[PSO_MAX_PARTICLES];
    uint8_t reset[PSO_MAX_PARTICLES];          // 1 = particle re-seeded
    float   newP[PSO_MAX_PARTICLES][5];
    float   newS[PSO_MAX_PARTICLES];
    int     order[PSO_MAX_PARTICLES];
    float   fresh[SP_REINIT_COUNT][5];
};

// Outcome of one detection = the fields of one result record.
struct PsoResult {
    int   variant;
    int   image;
    float best[5];    // cx, cy, a, b, theta (image pixels)
    float score;      // best fitness, PSO_NEG_INF if nothing valid
    int   evals;
    int   iters;
    int   flags;
    float hist[PSO_HIST_LEN];   // gbest fitness after every iteration (padded)
};

#endif
