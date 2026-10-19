#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

// ============================================================
// kernels.h  —  AIE kernels of the parallel ACO ellipse detector
//
//   AcoMaster : one per graph. Takes one image per graph iteration and
//               runs the colony of the algorithm built in (ACO_VARIANT):
//               it broadcasts every parallel phase of a colony iteration
//               to the workers and collects their answers.
//   AcoWorker : ACO_WORKERS of them, one tile each. Worker d < 5 owns
//               FPED pheromone trail d (h, k, a, b, theta); every worker
//               scores its share of the ants (ant i -> worker i % W).
//
// Master -> workers: one broadcast int32 stream of commands
//   CMD_BEGIN  seed, n, px[n], py[n], boosted[n], bright[n]   (new image)
//   CMD_TRAILS hasUpdate, 5 x FPED_UPDATE_WORDS               (FPED trail phase)
//   CMD_SCORE  n, n x 5 floats                                (score phase)
//   CMD_END                                                   (image done)
// Workers -> master: one packet per worker and phase, merged (pktmerge):
//   trail reply  id, bins[FPED_ANTS], vals[FPED_ANTS]
//   score reply  id, k, fit[k]          (the k ants of this worker, in order)
// ============================================================

#include <adf.h>
#include "kernels/include.h"
#include "kernels/aco_types.h"

#define CMD_END    0
#define CMD_BEGIN  1
#define CMD_TRAILS 2
#define CMD_SCORE  3

class AcoMaster {
public:
    AcoMaster();
    void run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
             output_stream<int32>* toWorkers, adf::output_buffer<int32>& result);
    static void registerKernelClass() { REGISTER_FUNCTION(AcoMaster::run); }

private:
    AcoEdgeSet     es_;
    AcoMasterState st_;          // FpedState or StdAcoState (include.h ACO_VARIANT)
};

class AcoWorker {
public:
    explicit AcoWorker(int id);
    void run(input_stream<int32>* fromMaster, output_pktstream* toMaster);
    static void registerKernelClass() { REGISTER_FUNCTION(AcoWorker::run); }

private:
    int        id_;
    AcoEdgeSet es_;
    FpedTrail  trail_;           // used by workers 0..4 in the FPED build
    int16_t    bins_[FPED_ANTS];
    float      vals_[FPED_ANTS];
};

#endif
