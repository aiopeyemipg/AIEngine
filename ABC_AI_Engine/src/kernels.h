#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

// ============================================================
// kernels.h  —  AIE kernels of the parallel ABC ellipse detector
//
//   AbcMaster : one per graph. Takes one image per graph iteration,
//               extracts the points (HA: lit pixels; SR: Canny edges)
//               and runs the colony of the algorithm built in
//               (ABC_VARIANT). It broadcasts every phase of every ABC
//               iteration to the workers and collects their answers.
//   AbcWorker : ABC_WORKERS of them, one tile each. Worker w handles the
//               bee tasks t = w, w + W, ... of every phase (one bee per
//               worker with 30 workers): build the candidate, score it.
//
// Master -> workers: one broadcast int32 stream of commands
//   CMD_BEGIN  seed, n, nPad, px[nPad], py[nPad], w[nPad]     (new image)
//   CMD_PHASE  kind, iter, bees[30][5], sel[30]               (one phase)
//   CMD_END                                                   (image done)
// Workers -> master: one packet per worker and phase (pktmerge):
//   id, k, then per task: cand[5], fitness, evaluations
// ============================================================

#include <adf.h>
#include "kernels/include.h"
#include "kernels/abc_types.h"
#include "kernels/abc_canny.h"

#define CMD_END    0
#define CMD_BEGIN  1
#define CMD_PHASE  2

// SR-ABC's Canny buffers (25.6 KB) are graph-side parameters, so the
// compiler can place them in neighbouring tile memories; HA-ABC does
// not use them (size 8).
#define CANNY_BUF ((ABC_VARIANT == VI_SR) ? CANNY_PIX : 8)

class AbcMaster {
public:
    AbcMaster(uint8 (&blur)[CANNY_BUF], int16 (&mag)[CANNY_BUF], int8 (&dir)[CANNY_BUF]);
    void run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
             output_stream<int32>* toWorkers, adf::output_buffer<int32>& result);
    static void registerKernelClass() {
        REGISTER_FUNCTION(AbcMaster::run);
        REGISTER_PARAMETER(blur_);
        REGISTER_PARAMETER(mag_);
        REGISTER_PARAMETER(dir_);
    }

private:
    AbcPoints pts_;
    AbcState  st_;
    uint8 (&blur_)[CANNY_BUF];
    int16 (&mag_)[CANNY_BUF];
    int8  (&dir_)[CANNY_BUF];
};

class AbcWorker {
public:
    explicit AbcWorker(int id);
    void run(input_stream<int32>* fromMaster, output_pktstream* toMaster);
    static void registerKernelClass() { REGISTER_FUNCTION(AbcWorker::run); }

private:
    int       id_;
    AbcPoints pts_;
    AbcPhase  msg_;
};

#endif
