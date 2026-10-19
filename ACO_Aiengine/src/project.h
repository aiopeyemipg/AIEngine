#ifndef PROJECT_H
#define PROJECT_H

// ============================================================
// project.h  —  ADF graph of the parallel ACO ellipse detector
//
// One graph iteration = one image. Inside it, every colony iteration
// is spread over ACO_WORKERS + 1 AIE tiles:
//
//                         +--> AcoWorker 0  (trail h, scores ants 0, 30, ..) --+
//                         +--> AcoWorker 1  (trail k)                        --+
//   images --> AcoMaster -+--> AcoWorker 2  (trail a)                        --+--> pktmerge --+
//   (PLIO)     |          +--> AcoWorker 3  (trail b)                        --+               |
//              |          +--> AcoWorker 4  (trail theta)                    --+               |
//              |          +--> AcoWorker 5 .. 29  (score ants)               --+               |
//              |    (one broadcast stream: commands + data)                                    |
//              +<----------------------------------------------------------------------------+
//              |    (worker answers, packet-merged into one stream)
//              +--> results (PLIO, 13 words per image)
//
// Per colony iteration: trail phase (5 trails at once) -> score phase
// (30 ants at once) -> master bookkeeping. runtime<ratio> = 0.9 gives
// every kernel its own tile, so the phases really run in parallel.
// The algorithm (FPED or StdACO) is chosen at build time: ACO_VARIANT.
// ============================================================

#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

static_assert(ACO_WORKERS >= 5 && ACO_WORKERS <= 32, "5 trail owners, at most 32 packet ids");

class AcoEllipseGraph : public adf::graph {
private:
    kernel                master;
    kernel                worker[ACO_WORKERS];
    pktmerge<ACO_WORKERS> merge;

public:
    input_plio  in;
    output_plio out;

    AcoEllipseGraph() {
        in  = input_plio::create("images", plio_32_bits, "/home/embeddedai002/Desktop/AI_engine/ACO_Aiengine/data/Ellipses_1000_images.csv");
        out = output_plio::create("results", plio_32_bits, "data/aco_results.csv");

        master = kernel::create_object<AcoMaster>();
        source(master) = "kernels/aco_detectors.cc";
        runtime<ratio>(master) = 0.9;
        connect(in.out[0], master.in[0]);
        dimensions(master.in[0]) = { IMG_RECORD_BYTES };
        connect(master.out[1], out.in[0]);
        dimensions(master.out[1]) = { RESULT_WORDS };

        merge = pktmerge<ACO_WORKERS>::create();
        connect(merge.out[0], master.in[1]);

        for (int w = 0; w < ACO_WORKERS; ++w) {
            worker[w] = kernel::create_object<AcoWorker>(w);
            source(worker[w]) = "kernels/aco_detectors.cc";
            runtime<ratio>(worker[w]) = 0.9;          // one tile per worker
            connect(master.out[0], worker[w].in[0]);  // broadcast stream
            connect(worker[w].out[0], merge.in[w]);
        }
    }
};

#endif
