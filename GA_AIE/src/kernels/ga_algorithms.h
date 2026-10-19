#ifndef GA_ALGORITHMS_H
#define GA_ALGORITHMS_H

// ============================================================
// ga_algorithms.h  —  the three GA ellipse detectors, one image per
// call, with the work of every generation spread over parallel workers
//
//   ga_detect_conv   : Conventional generational GA   (ConventionalGaDetector.h)
//   ga_detect_micro  : Micro GA with restarts         (MicroGaDetector.h)
//   ga_detect_island : Island Model GA                (IslandGaDetector.h)
//
// Concurrency inside one image. The generations are sequential
// (generation t+1 breeds from generation t), but the work inside one
// generation is not:
//   Conventional : the 780 children of a generation are built
//                  (tournament, BLX-alpha, mutation) AND scored in
//                  parallel, one child pair per task; the master keeps
//                  the elite, the ranking and the best-so-far.
//   Micro        : the 13 children of a step, likewise (small: 7 tasks).
//   Island       : each of the 8 islands evolves a whole epoch on its own
//                  worker; the master does the ring migration.
// Every task draws from its own counter-based random stream
// (ga_stream(seed, generation, task)), so serial, threaded and AIE runs
// give the same detections whatever the number of workers.
//
// The detectors are templates on an Engine that runs the parallel phases:
//   void begin(const EdgeSet& es, uint32_t seed);              // new image
//   void conv_init(float* pop);                                // CONV_POP individuals
//   void conv_children(const float* cur, int gen, float* out); // CONV_CHILDREN children
//   void micro_random(float* pop, int first, int count, int cycle);
//   void micro_children(const float* pop, int step, float progress, float* out);
//   void island_epoch(const IslEpochCmd& cmd, IslEpochReply* rep);
// implemented by the AIE master (ga_detectors.cc: broadcast to the worker
// tiles) and by the native bench (bench/ga_bench.cpp: threads or serial).
// The task functions below are what a worker runs. No adf types here.
// ============================================================

#include "include.h"
#include "ga_types.h"
#include "ga_core.h"
#include "ga_math.h"

#define CONV_ELITE     (CONV_ELITISM < CONV_POP ? CONV_ELITISM : CONV_POP)
#define CONV_CHILDREN  (CONV_POP - CONV_ELITE)
#define CONV_PAIRS     ((CONV_CHILDREN + 1) / 2)
#define MICRO_CHILDREN (MICRO_POP - 1)
#define MICRO_PAIRS    ((MICRO_CHILDREN + 1) / 2)

// stream ids (first key of ga_stream) of the non-generation tasks
#define STREAM_CONV_INIT   0x7FFF0000u
#define STREAM_MICRO_INIT  0x7FFE0000u
#define STREAM_ISLAND      0x7FFD0000u

// ============================================================
// Worker tasks
// ============================================================

// Conventional GA: individual i of the initial population.
static inline void conv_init_task(const EdgeSet& es, uint32_t seed, int i, float* g) {
    GaRng rng(ga_stream(seed, STREAM_CONV_INIT, (uint32_t)i));
    ga_random_individual(rng, g);
    g[IND_FIT] = ga_fitness(es, g);
}

// Child pair k of a generation: two tournaments on the current
// population, BLX-alpha, mutation of both children, fitness.
// Writes `count` (1 or 2) individuals to out.
static inline void ga_pair_task(const EdgeSet& es, GaRng& rng, const float* cur, int n, int tour,
                                float xrate, float blx, float mut0, float progress,
                                float* out, int count) {
    const int p1 = ga_tournament(cur, n, tour, rng);
    const int p2 = ga_tournament(cur, n, tour, rng);
    float c1[5], c2[5];
    ga_blx(cur + p1 * IND_WORDS, cur + p2 * IND_WORDS, xrate, blx, rng, c1, c2);
    ga_mutate(c1, progress, mut0, rng);
    ga_copy5(out, c1);
    out[IND_FIT] = ga_fitness(es, out);
    if (count > 1) {
        ga_mutate(c2, progress, mut0, rng);
        ga_copy5(out + IND_WORDS, c2);
        out[IND_WORDS + IND_FIT] = ga_fitness(es, out + IND_WORDS);
    }
}

static inline float conv_progress(int gen) {
    float progress = (float)gen / (float)(CONV_GENS - 1 > 1 ? CONV_GENS - 1 : 1);
    return progress > 1.0f ? 1.0f : progress;
}

static inline void conv_pair_task(const EdgeSet& es, uint32_t seed, int gen, int k,
                                  const float* cur, float* out) {
    GaRng rng(ga_stream(seed, (uint32_t)gen, (uint32_t)k));
    const int count = (2 * k + 1 < CONV_CHILDREN) ? 2 : 1;
    ga_pair_task(es, rng, cur, CONV_POP, CONV_TOURNAMENT, CONV_XRATE, CONV_BLX, CONV_MUT0,
                 conv_progress(gen), out, count);
}

// Micro GA: random individual i of restart cycle `cycle`.
static inline void micro_random_task(const EdgeSet& es, uint32_t seed, int cycle, int i, float* g) {
    GaRng rng(ga_stream(seed, STREAM_MICRO_INIT + (uint32_t)cycle, (uint32_t)i));
    ga_random_individual(rng, g);
    g[IND_FIT] = ga_fitness(es, g);
}

// Micro GA: child pair k of step `step`.
static inline void micro_pair_task(const EdgeSet& es, uint32_t seed, int step, int k,
                                   const float* pop, float progress, float* out) {
    GaRng rng(ga_stream(seed, (uint32_t)step, (uint32_t)k));
    const int count = (2 * k + 1 < MICRO_CHILDREN) ? 2 : 1;
    ga_pair_task(es, rng, pop, MICRO_POP, MICRO_TOURNAMENT, MICRO_XRATE, MICRO_BLX, MICRO_MUT0,
                 progress, out, count);
}

// Build one island: up to 25 % warm seeds, the rest random (initIsland).
static inline void isl_init(float* pop, const EdgeSet& es, GaRng& rng,
                     bool haveSeed, const float* seed, int& evals) {
    int n = 0;
    const int nSeed = haveSeed ? (1 < ISL_POP / 4 ? 1 : ISL_POP / 4) : 0;
    for (; n < nSeed; ++n) {
        float* g = pop + n * IND_WORDS;
        ga_copy5(g, seed);
        g[IND_FIT] = ga_fitness(es, g);
        ++evals;
    }
    for (; n < ISL_POP; ++n) {
        float* g = pop + n * IND_WORDS;
        ga_random_individual(rng, g);
        g[IND_FIT] = ga_fitness(es, g);
        ++evals;
    }
    ga_sort_desc(pop, ISL_POP);
}

// Std-dev of the last `win` entries of hist (windowStd).
static inline float isl_window_std(const float* hist, int len, int win) {
    const int start = len > win ? len - win : 0;
    const int n = len - start;
    if (n < 1) return 1.0e30f;
    float sum = 0.0f;
    for (int i = start; i < len; ++i) sum += hist[i];
    const float mean = sum / (float)n;
    float acc = 0.0f;
    for (int i = start; i < len; ++i) { const float d = hist[i] - mean; acc += d * d; }
    return ga_sqrt(acc / (float)n);
}

// Evolve one island for one epoch (evolveIsland).
static inline void isl_evolve(float* pop, float* nxt, float* hist, int genOffset,
                       const EdgeSet& es, GaRng& rng, int& evals, int& gens) {
    float bestFit = pop[IND_FIT];
    int stagCount = 0;
    int histLen = 0;

    for (int localGen = 0; localGen < ISL_MIG_INTERVAL; ++localGen) {
        const int globalGen = genOffset + localGen;
        const float progress = (float)globalGen / (float)(ISL_GENERATIONS > 1 ? ISL_GENERATIONS : 1);

        int n = 0;
        for (; n < ISL_ELITISM && n < ISL_POP; ++n)
            ga_copy_ind(nxt + n * IND_WORDS, pop + n * IND_WORDS);
        while (n < ISL_POP) {
            const int p1 = ga_tournament(pop, ISL_POP, ISL_TOURNAMENT, rng);
            const int p2 = ga_tournament(pop, ISL_POP, ISL_TOURNAMENT, rng);
            float c1[5], c2[5];
            ga_blx(pop + p1 * IND_WORDS, pop + p2 * IND_WORDS, ISL_XRATE, ISL_BLX, rng, c1, c2);
            ga_mutate(c1, progress, ISL_MUT0, rng);
            ga_mutate(c2, progress, ISL_MUT0, rng);
            float* d = nxt + n * IND_WORDS;
            ga_copy5(d, c1);
            d[IND_FIT] = ga_fitness(es, d);
            ++evals; ++n;
            if (n < ISL_POP) {
                d = nxt + n * IND_WORDS;
                ga_copy5(d, c2);
                d[IND_FIT] = ga_fitness(es, d);
                ++evals; ++n;
            }
        }
        for (int i = 0; i < ISL_POP * IND_WORDS; ++i) pop[i] = nxt[i];
        ga_sort_desc(pop, ISL_POP);
        if (pop[IND_FIT] > bestFit) bestFit = pop[IND_FIT];
        hist[histLen++] = bestFit;
        ++gens;

        bool stop = false;
#if ISL_REINJECTION
        // per-island stagnation test (never fires when the epoch is
        // shorter than the window, as with the native parameters)
        if (localGen >= ISL_STAG_WINDOW) {
            const float sd = isl_window_std(hist, histLen, ISL_STAG_WINDOW);
            const int finiteN = histLen < ISL_STAG_WINDOW ? histLen : ISL_STAG_WINDOW;
            if (finiteN > 5 && sd < ISL_STAG_TOL) {
                ++stagCount;
                if (stagCount >= ISL_MAX_STAG) {
                    stop = true;
                } else {
                    const int nInject = (int)(ISL_DIVERSITY_FRAC * (float)ISL_POP);
                    for (int j = ISL_POP - nInject; j < ISL_POP; ++j) {   // replace the worst
                        float* g = pop + j * IND_WORDS;
                        ga_random_individual(rng, g);
                        g[IND_FIT] = ga_fitness(es, g);
                        ++evals;
                    }
                    ga_sort_desc(pop, ISL_POP);
                }
            } else {
                stagCount = 0;
            }
        }
#endif
        if (stop) break;
    }
}

// Island GA: one epoch of island `island` (runs on its own worker):
// build it (first epoch of a cycle) or take its migrants, then evolve
// ISL_MIG_INTERVAL generations, then report its ISL_MIG_SIZE best.
static inline void island_epoch_task(const EdgeSet& es, uint32_t seed, const IslEpochCmd& cmd,
                                     int island, GaIsland& isl, IslEpochReply& rep) {
    GaRng rng(ga_stream(seed, STREAM_ISLAND + (uint32_t)(cmd.restart * ISL_EPOCHS + cmd.epoch),
                        (uint32_t)island));
    int evals = 0, gens = 0;
    float* pop = &isl.pop[0][0];
    if (cmd.init) {
        isl_init(pop, es, rng, cmd.haveSeed != 0, cmd.seed, evals);
    } else if (cmd.migrate) {
        // the migrants replace the worst, as in migrate()
        for (int j = 0; j < ISL_MIG_SIZE; ++j)
            ga_copy_ind(isl.pop[ISL_POP - ISL_MIG_SIZE + j], cmd.migrants[island][j]);
        ga_sort_desc(pop, ISL_POP);
    }
    isl_evolve(pop, &isl.next[0][0], isl.hist, cmd.epoch * ISL_MIG_INTERVAL, es, rng, evals, gens);
    for (int j = 0; j < ISL_MIG_SIZE; ++j) ga_copy_ind(rep.top[j], isl.pop[j]);
    for (int g = 0; g < ISL_MIG_INTERVAL; ++g)              // pad an early stop
        rep.hist[g] = g < gens ? isl.hist[g] : (gens > 0 ? isl.hist[gens - 1] : isl.pop[0][IND_FIT]);
    rep.evals = evals;
    rep.gens = gens;
}

// ============================================================
// Master side
// ============================================================

// Result for an image with too few edge points (the reference returns
// an empty, undetected result).
static inline void ga_set_empty(GaResult& r, int variant, int image, int flags) {
    r.variant = variant;
    r.image = image;
    r.detected = 0;
    for (int d = 0; d < 5; ++d) r.ell[d] = 0.0f;
    r.fitness = 0.0f;
    r.evals = 0;
    r.gens = 0;
    r.flags = flags;
    for (int i = 0; i < GA_HIST_MAX; ++i) r.hist[i] = INVALID_FITNESS;
}

// Pad the history after `n` recorded values with the last one.
static inline void ga_pad_hist(GaResult& r, int n, int len) {
    const float last = n > 0 ? r.hist[n - 1] : INVALID_FITNESS;
    for (int i = n; i < len; ++i) r.hist[i] = last;
}

// Normalise the best chromosome, validate it and fill the result.
static inline void ga_finish(GaResult& r, const EdgeSet& es, int variant, int image,
                             float* best, float bestFit, int evals, int gens, int flags) {
    ga_normalize(best);
    r.variant = variant;
    r.image = image;
    r.detected = ga_is_valid(es, best) ? 1 : 0;
    for (int d = 0; d < 5; ++d) r.ell[d] = best[d];
    r.fitness = bestFit;
    r.evals = evals;
    r.gens = gens;
    r.flags = flags;
}

// ------------------------------------------------------------
// Conventional GA: each generation keeps the CONV_ELITISM best and fills
// the population with CONV_CHILDREN children built in parallel. The
// population stays unsorted; order holds the ranking.
// ------------------------------------------------------------
template <typename Engine>
static inline void ga_detect_conv(const EdgeSet& es, int image, int flags, uint32_t seed,
                                  ConvGaState& st, float* popA, float* popB,
                                  Engine& eng, GaResult& res) {
    eng.begin(es, seed);
    const int P = CONV_POP;
    float* cur = popA;
    float* nxt = popB;

    eng.conv_init(cur);                                   // parallel
    int evals = P;
    ga_order_desc(cur, st.order, P);
    float best[5];
    ga_copy5(best, cur + st.order[0] * IND_WORDS);
    float bestFit = cur[st.order[0] * IND_WORDS + IND_FIT];

    for (int gen = 0; gen < CONV_GENS; ++gen) {
        for (int n = 0; n < CONV_ELITE; ++n)
            ga_copy_ind(nxt + n * IND_WORDS, cur + st.order[n] * IND_WORDS);
        eng.conv_children(cur, gen, nxt + CONV_ELITE * IND_WORDS);   // parallel
        evals += CONV_CHILDREN;

        float* t = cur; cur = nxt; nxt = t;
        ga_order_desc(cur, st.order, P);
        const float* top = cur + st.order[0] * IND_WORDS;
        if (top[IND_FIT] > bestFit) { bestFit = top[IND_FIT]; ga_copy5(best, top); }
        if (gen < CONV_HIST_LEN) res.hist[gen] = bestFit;
    }
    ga_pad_hist(res, CONV_GENS < CONV_HIST_LEN ? CONV_GENS : CONV_HIST_LEN, CONV_HIST_LEN);
    ga_finish(res, es, VI_CONV, image, best, bestFit, evals, CONV_GENS, flags);
}

// ------------------------------------------------------------
// Micro GA (Krishnakumar, 1989): a tiny population evolves (elitism 1)
// until its fitness std-dev drops below MICRO_CONV_TOL, then restarts
// with the global best plus fresh random individuals, until
// MICRO_MAX_EVALS evaluations are spent.
// ------------------------------------------------------------
template <typename Engine>
static inline void ga_detect_micro(const EdgeSet& es, int image, int flags, uint32_t seed,
                                   MicroGaState& st, Engine& eng, GaResult& res) {
    eng.begin(es, seed);
    const int POP = MICRO_POP;
    float* pop = &st.pop[0][0];
    float* nxt = &st.next[0][0];

    bool  haveBest = false;
    float best[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float bestFit = -1.0e9f;
    int   evals = 0, step = 0, cycle = 0;

    while (evals < MICRO_MAX_EVALS) {
        // (re)initialise: global best + fresh random individuals (parallel)
        int n0 = 0;
        if (haveBest) {
            ga_copy5(pop, best);
            pop[IND_FIT] = bestFit;
            n0 = 1;
        }
        eng.micro_random(pop, n0, POP - n0, cycle++);
        evals += POP - n0;
        ga_sort_desc(pop, POP);

        // evolve until convergence
        while (evals < MICRO_MAX_EVALS) {
            if (pop[IND_FIT] > bestFit) { bestFit = pop[IND_FIT]; ga_copy5(best, pop); haveBest = true; }
            if (step < MICRO_HIST_LEN) res.hist[step] = bestFit;
            ++step;
            if (ga_fitness_std(pop, POP) < MICRO_CONV_TOL) break;   // converged -> restart

            const float progress = (float)evals / (float)MICRO_MAX_EVALS;
            ga_copy_ind(nxt, pop);                                   // elitism
            eng.micro_children(pop, step, progress, nxt + IND_WORDS);  // parallel
            evals += MICRO_CHILDREN;
            float* t = pop; pop = nxt; nxt = t;
            ga_sort_desc(pop, POP);
        }
    }

    if (!haveBest) { ga_set_empty(res, VI_MICRO, image, flags); return; }
    ga_pad_hist(res, step < MICRO_HIST_LEN ? step : MICRO_HIST_LEN, MICRO_HIST_LEN);
    ga_finish(res, es, VI_MICRO, image, best, bestFit, evals, step, flags);
}

// ------------------------------------------------------------
// Island Model GA: ISL_ISLANDS islands evolve in isolation for one epoch
// of ISL_MIG_INTERVAL generations (all islands at the same time), then
// each island's ISL_MIG_SIZE best replace the worst of the next island
// on the ring. The cycle runs ISL_RESTARTS times; later cycles are
// seeded with the best chromosome found so far.
// ------------------------------------------------------------
template <typename Engine>
static inline void ga_detect_island(const EdgeSet& es, int image, int flags, uint32_t seed,
                                    IslandGaState& st, Engine& eng, GaResult& res) {
    eng.begin(es, seed);
    int   evals = 0, gens = 0;
    bool  haveBest = false;
    float best[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float bestFit = -1.0e9f;
    IslEpochCmd& cmd = st.cmd;
    float runBest = INVALID_FITNESS;       // best-so-far over all islands, for the history
    int   h = 0;

    for (int restart = 0; restart < ISL_RESTARTS; ++restart) {
        for (int epoch = 0; epoch < ISL_EPOCHS; ++epoch) {
            cmd.restart = restart;
            cmd.epoch = epoch;
            cmd.init = (epoch == 0) ? 1 : 0;
            cmd.haveSeed = haveBest ? 1 : 0;
            ga_copy5(cmd.seed, best);
            // ring migration after every epoch but the last of a cycle:
            // island i's emigrants (its best before migration) go to i + 1
            cmd.migrate = (ISL_MIGRATION && ISL_ISLANDS > 1 && epoch > 0) ? 1 : 0;
            if (cmd.migrate) {
                for (int src = 0; src < ISL_ISLANDS; ++src) {
                    const int dst = (src + 1) % ISL_ISLANDS;
                    for (int j = 0; j < ISL_MIG_SIZE; ++j)
                        ga_copy_ind(cmd.migrants[dst][j], st.rep[src].top[j]);
                }
                // the reference re-scores the migrants; the fitness is
                // deterministic, so only the evaluation count is kept
                evals += ISL_ISLANDS * ISL_MIG_SIZE;
            }
            eng.island_epoch(cmd, st.rep);                   // all islands at once
            for (int i = 0; i < ISL_ISLANDS; ++i) { evals += st.rep[i].evals; gens += st.rep[i].gens; }
            for (int g = 0; g < ISL_MIG_INTERVAL && h < ISL_HIST_LEN; ++g, ++h) {
                for (int i = 0; i < ISL_ISLANDS; ++i)
                    if (st.rep[i].hist[g] > runBest) runBest = st.rep[i].hist[g];
                res.hist[h] = runBest;
            }
        }
        for (int i = 0; i < ISL_ISLANDS; ++i)
            if (st.rep[i].top[0][IND_FIT] > bestFit) {
                bestFit = st.rep[i].top[0][IND_FIT];
                ga_copy5(best, st.rep[i].top[0]);
                haveBest = true;
            }
    }
    ga_pad_hist(res, h, ISL_HIST_LEN);
    ga_finish(res, es, VI_ISLAND, image, best, bestFit, evals, gens, flags);
}

// ------------------------------------------------------------
// Detect the ellipse of one image record with the variant `vi`.
// seedOffset = rep * 15485863 gives an independent seed set
// (0 = the main study, seed = imageID*7919 + vi*104729 + 1).
// ------------------------------------------------------------
template <typename Engine>
static inline void ga_detect_image(const uint8_t* rec, int vi, EdgeSet& es, void* state,
                                   float* popA, float* popB, Engine& eng, GaResult& res,
                                   uint32_t seedOffset = 0) {
    int flags;
    const int image = ga_load_image(rec, es, flags);
    if (image < 0 || es.n < MIN_INLIERS_FOR_DETECTION) { ga_set_empty(res, vi, image, flags); return; }
    const uint32_t seed = ga_seed(image, vi) + seedOffset;
    if (vi == VI_CONV)
        ga_detect_conv(es, image, flags, seed, *(ConvGaState*)state, popA, popB, eng, res);
    else if (vi == VI_MICRO)
        ga_detect_micro(es, image, flags, seed, *(MicroGaState*)state, eng, res);
    else
        ga_detect_island(es, image, flags, seed, *(IslandGaState*)state, eng, res);
}

#endif
