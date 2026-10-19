#ifndef PROJECT_H
#define PROJECT_H

// ============================================================
// project.h  —  ADF graph of the parallel GA ellipse detector
//
// One graph iteration = one image. Inside it, every generation is
// spread over GA_WORKERS + 1 AIE tiles:
//
//                        +--> GaWorker 0   (children 0, 30, 60 .. / island 0) --+
//                        +--> GaWorker 1   (children 1, 31, ..    / island 1) --+
//   images --> GaMaster -+--> ...                                            --+--> pktmerge --+
//   (PLIO)     |         +--> GaWorker 29  (children 29, 59, ..)             --+               |
//              |   (one broadcast stream: commands + population)                              |
//              +<------------------------------------------------------------------------------+
//              |   (worker answers, packet-merged into one stream)
//              +--> results (PLIO, 13 words + convergence history per image)
//
// Per generation: the master broadcasts the population, every worker
// builds and scores its children at the same time, the master keeps the
// elite, ranks and tracks the best (Island GA: one epoch per island and
// worker, migration on the master). runtime<ratio> = 0.9 gives every
// kernel its own tile, so the workers really run in parallel.
// The variant is chosen at build time: GA_VARIANT.
// ============================================================

#include <adf.h>
#include <vector>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

static_assert(GA_WORKERS >= ISL_ISLANDS && GA_WORKERS <= 32,
              "one worker per island, at most 32 packet ids");

class GaEllipseGraph : public adf::graph {
private:
    kernel               master;
    kernel               worker[GA_WORKERS];
    pktmerge<GA_WORKERS> merge;

public:
    input_plio  in;
    output_plio out;

    GaEllipseGraph() {
        in  = input_plio::create("images", plio_32_bits, "data/ga_input.txt");
        out = output_plio::create("results", plio_32_bits, "data/ga_results.txt");

#if GA_VARIANT == VI_CONV
        master = kernel::create_object<GaMaster>(std::vector<float>(CONV_POP * IND_WORDS, 0.0f),
                                                 std::vector<float>(CONV_POP * IND_WORDS, 0.0f));
#else
        master = kernel::create_object<GaMaster>();
#endif
        source(master) = "kernels/ga_detectors.cc";
        runtime<ratio>(master) = 0.9;
        connect(in.out[0], master.in[0]);
        dimensions(master.in[0]) = { IMG_RECORD_BYTES };
        connect(master.out[1], out.in[0]);
        dimensions(master.out[1]) = { RESULT_WORDS };

        merge = pktmerge<GA_WORKERS>::create();
        connect(merge.out[0], master.in[1]);

        for (int w = 0; w < GA_WORKERS; ++w) {
#if GA_VARIANT == VI_CONV
            worker[w] = kernel::create_object<GaWorker>(w, std::vector<float>(CONV_POP * IND_WORDS, 0.0f));
#else
            worker[w] = kernel::create_object<GaWorker>(w);
#endif
            source(worker[w]) = "kernels/ga_detectors.cc";
            runtime<ratio>(worker[w]) = 0.9;          // one tile per worker
            connect(master.out[0], worker[w].in[0]);  // broadcast stream
            connect(worker[w].out[0], merge.in[w]);
        }
    }
};

#endif
