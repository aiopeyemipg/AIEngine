#ifndef PROJECT_H
#define PROJECT_H

// ============================================================
// project.h  —  ADF graph of the parallel ABC ellipse detector
//
// One graph iteration = one image. Inside it, every phase of every ABC
// iteration (employed, onlooker, scout) is spread over ABC_WORKERS
// worker tiles, one bee per worker:
//
//                         +--> AbcWorker 0   (bee 0)   --+
//   images --> AbcMaster -+--> AbcWorker 1   (bee 1)   --+--> pktmerge --+
//   (PLIO)     |          +--> ...                     --+               |
//              |          +--> AbcWorker 29  (bee 29)  --+               |
//              |    (one broadcast stream: commands + data)              |
//              +<--------------------------------------------------------+
//              |    (worker answers, packet-merged into one stream)
//              +--> results (PLIO, 13 words + 150-value history per image)
//
// runtime<ratio> = 0.9 gives every kernel its own tile, so the bees of
// a phase really are built and scored at the same time.
// The algorithm (HA-ABC or SR-ABC) is chosen at build time: ABC_VARIANT.
// ============================================================

#include <adf.h>
#include <vector>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

static_assert(ABC_WORKERS >= 1 && ABC_WORKERS <= 32, "at most 32 packet ids");

class AbcEllipseGraph : public adf::graph {
private:
    kernel                master;
    kernel                worker[ABC_WORKERS];
    pktmerge<ABC_WORKERS> merge;

public:
    input_plio  in;
    output_plio out;

    AbcEllipseGraph() {
        in  = input_plio::create("images", plio_32_bits, "data/abc_input.txt");
        out = output_plio::create("results", plio_32_bits, "data/abc_results.txt");

        master = kernel::create_object<AbcMaster>(std::vector<uint8>(CANNY_BUF, 0),
                                                  std::vector<int16>(CANNY_BUF, 0),
                                                  std::vector<int8>(CANNY_BUF, 0));
        source(master) = "kernels/abc_detectors.cc";
        runtime<ratio>(master) = 0.9;
        connect(in.out[0], master.in[0]);
        dimensions(master.in[0]) = { IMG_RECORD_BYTES };
        connect(master.out[1], out.in[0]);
        dimensions(master.out[1]) = { RESULT_WORDS };

        merge = pktmerge<ABC_WORKERS>::create();
        connect(merge.out[0], master.in[1]);

        for (int w = 0; w < ABC_WORKERS; ++w) {
            worker[w] = kernel::create_object<AbcWorker>(w);
            source(worker[w]) = "kernels/abc_detectors.cc";
            runtime<ratio>(worker[w]) = 0.9;          // one tile per worker
            connect(master.out[0], worker[w].in[0]);  // broadcast stream
            connect(worker[w].out[0], merge.in[w]);
        }
    }
};

#endif
