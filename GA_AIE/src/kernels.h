#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

// ============================================================
// kernels.h  —  AIE kernels of the parallel GA ellipse detector
//
//   GaMaster : one per graph. Takes one image per graph iteration and
//              runs the GA variant built in (GA_VARIANT). Every parallel
//              phase of a generation is broadcast to the workers, and
//              their answers are collected.
//   GaWorker : GA_WORKERS of them, one tile each. They build and score
//              their share of the children (task k -> worker k % W), or
//              evolve one island each (Island GA, workers 0..7).
//
// Master -> workers: one broadcast int32 stream of commands
//   CMD_BEGIN       seed, n, px[n], py[n], boosted[n], bright[n]  (new image)
//   CMD_CONV_INIT                                  initial population
//   CMD_CONV_GEN    gen, population (CONV_POP x 6) children of one generation
//   CMD_MICRO_RAND  count, cycle                   random individuals
//   CMD_MICRO_GEN   step, progress, population (MICRO_POP x 6)
//   CMD_ISL_EPOCH   IslEpochCmd (ISL_CMD_WORDS)    one epoch of every island
//   CMD_END                                        image done
// Workers -> master: one packet per worker and phase (pktmerge):
//   id, count, count x item       (item = 1 individual, or 1 child pair = 2)
//   id, evals, gens, top[ISL_MIG_SIZE], hist[ISL_MIG_INTERVAL] (island reply)
// ============================================================

#include <adf.h>
#include "kernels/include.h"
#include "kernels/ga_types.h"

#define CMD_END        0
#define CMD_BEGIN      1
#define CMD_CONV_INIT  2
#define CMD_CONV_GEN   3
#define CMD_MICRO_RAND 4
#define CMD_MICRO_GEN  5
#define CMD_ISL_EPOCH  6

class GaMaster {
public:
#if GA_VARIANT == VI_CONV
    // The two population arrays (19.2 KB each) are graph-side parameters,
    // so the compiler can place them in neighbouring tile memories.
    GaMaster(float (&popA)[CONV_POP * IND_WORDS], float (&popB)[CONV_POP * IND_WORDS]);
    static void registerKernelClass() {
        REGISTER_FUNCTION(GaMaster::run);
        REGISTER_PARAMETER(popA_);
        REGISTER_PARAMETER(popB_);
    }
#else
    GaMaster();
    static void registerKernelClass() { REGISTER_FUNCTION(GaMaster::run); }
#endif
    void run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
             output_stream<int32>* toWorkers, adf::output_buffer<int32>& result);

private:
    EdgeSet       es_;
    GaMasterState st_;          // ConvGaState, MicroGaState or IslandGaState
#if GA_VARIANT == VI_CONV
    float (&popA_)[CONV_POP * IND_WORDS];
    float (&popB_)[CONV_POP * IND_WORDS];
#endif
};

class GaWorker {
public:
#if GA_VARIANT == VI_CONV
    // copy of the current population (19.2 KB), graph-side parameter
    GaWorker(int id, float (&pop)[CONV_POP * IND_WORDS]);
    static void registerKernelClass() {
        REGISTER_FUNCTION(GaWorker::run);
        REGISTER_PARAMETER(pop_);
    }
#else
    explicit GaWorker(int id);
    static void registerKernelClass() { REGISTER_FUNCTION(GaWorker::run); }
#endif
    void run(input_stream<int32>* fromMaster, output_pktstream* toMaster);

private:
    int      id_;
    uint32_t seed_;
    EdgeSet  es_;
#if GA_VARIANT == VI_CONV
    float (&pop_)[CONV_POP * IND_WORDS];
#elif GA_VARIANT == VI_MICRO
    float    pop_[MICRO_POP * IND_WORDS];
#else
    GaIsland      isl_;         // the island of this worker (id < ISL_ISLANDS)
    IslEpochCmd   cmd_;
    IslEpochReply rep_;
#endif
};

#endif
