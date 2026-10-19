#ifndef PROJECT_H
#define PROJECT_H

// ============================================================
// project.h  —  ADF graph of the parallel PSO ellipse detector
//
// One graph iteration = one image. Inside it, every PSO iteration is
// spread over PSO_WORKERS + 1 AIE tiles:
//
//                         +--> PsoWorker 0   (particles 0, 25, 50, 75) --+
//                         +--> PsoWorker 1   (particles 1, 26, 51, 76) --+
//   images --> PsoMaster -+--> ...                                     --+--> pktmerge --+
//   (PLIO)     |          +--> PsoWorker 24  (particles 24, 49, 74, 99) --+              |
//              |    (one broadcast stream: gbest / ring table + commands)               |
//              +<-----------------------------------------------------------------------+
//              |    (worker answers, packet-merged into one stream)
//              +--> results (PLIO, RESULT_WORDS words per image:
//                   13 scalars + gbest history)
//
// Per PSO iteration: the master broadcasts the swarm leader, all workers
// move + score their particles at the same time, the master reduces the
// answers to the new gbest. runtime<ratio> = 0.9 gives every kernel its
// own tile, so the particles really move in parallel.
// The variant is chosen at build time: PSO_VARIANT (vi 0..5).
// ============================================================

#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

static_assert(PSO_WORKERS >= 1 && PSO_WORKERS <= 32, "at most 32 packet ids");

class PsoEllipseGraph : public adf::graph {
private:
    kernel                master;
    kernel                worker[PSO_WORKERS];
    pktmerge<PSO_WORKERS> merge;

public:
    input_plio  in;
    output_plio out;

    PsoEllipseGraph() {
        in  = input_plio::create("images", plio_32_bits, "data/pso_input.txt");
        out = output_plio::create("results", plio_32_bits, "data/pso_results.txt");

        master = kernel::create_object<PsoMaster>();
        source(master) = "kernels/pso_detectors.cc";
        runtime<ratio>(master) = 0.9;
        connect(in.out[0], master.in[0]);
        dimensions(master.in[0]) = { IMG_RECORD_BYTES };
        connect(master.out[1], out.in[0]);
        dimensions(master.out[1]) = { RESULT_WORDS };

        merge = pktmerge<PSO_WORKERS>::create();
        connect(merge.out[0], master.in[1]);

        for (int w = 0; w < PSO_WORKERS; ++w) {
            worker[w] = kernel::create_object<PsoWorker>(w);
            source(worker[w]) = "kernels/pso_detectors.cc";
            runtime<ratio>(worker[w]) = 0.9;          // one tile per worker
            connect(master.out[0], worker[w].in[0]);  // broadcast stream
            connect(worker[w].out[0], merge.in[w]);
        }
    }
};

#endif
