#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

// ============================================================
// kernels.h  —  AIE kernels of the parallel PSO ellipse detector
//
//   PsoMaster : one per graph. Takes one image per graph iteration,
//               extracts the edges (+ DBSCAN for even vi), seeds the
//               swarm and runs the PSO of the variant built in
//               (PSO_VARIANT): every iteration it broadcasts the swarm
//               leader to the workers and reduces their answers.
//   PsoWorker : PSO_WORKERS of them, one tile each. Worker w owns the
//               particles w, w + W, ... (position, velocity, personal
//               best) and moves + scores them every iteration.
//
// Master -> workers: one broadcast int32 stream of commands
//   CMD_BEGIN   seed, n, px[n], py[n], boosted[n], bright[n]     (new image)
//   CMD_INIT    P, P x 5 floats                                  (initial swarm)
//   CMD_STEP    it, P, gbest[5]                                  (Std / APSO iteration)
//   CMD_STEP_SP it, P x (pbestScore, pbest[5], reset)            (SP-PSO iteration)
//   CMD_END                                                      (image done)
// Workers -> master: one packet per worker and command, merged (pktmerge):
//   init reply     id, k, score[k]                (own particles, ascending)
//   step reply     id, idx, score, pos[5]         (best own particle, idx -1 = none)
//   step_sp reply  id, k, k x (pbestScore, pbest[5])
// ============================================================

#include <adf.h>
#include "kernels/include.h"
#include "kernels/pso_types.h"

#define CMD_END     0
#define CMD_BEGIN   1
#define CMD_INIT    2
#define CMD_STEP    3
#define CMD_STEP_SP 4

class PsoMaster {
public:
    PsoMaster();
    void run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
             output_stream<int32>* toWorkers, adf::output_buffer<int32>& result);
    static void registerKernelClass() { REGISTER_FUNCTION(PsoMaster::run); }

private:
    PsoMasterState m_;
};

class PsoWorker {
public:
    explicit PsoWorker(int id);
    void run(input_stream<int32>* fromMaster, output_pktstream* toMaster);
    static void registerKernelClass() { REGISTER_FUNCTION(PsoWorker::run); }

private:
    int      id_;
    uint32_t seed_;
    EdgeSet  es_;
    PsoShard shard_;
    // SP-PSO ring table received every iteration
    float    tabP_[SP_PARTICLES][5];
    float    tabS_[SP_PARTICLES];
    uint8_t  reset_[SP_PARTICLES];
    float    newP_[SP_PARTICLES][5];
    float    newS_[SP_PARTICLES];
};

#endif
