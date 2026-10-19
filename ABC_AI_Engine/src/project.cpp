#include <adf.h>
#include "kernels.h"
#include "project.h"
#include <iostream>

using namespace adf;

AbcEllipseGraph mygraph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    mygraph.init();
    // one graph iteration = one image; the bees of every ABC phase run
    // on the ABC_WORKERS worker tiles in parallel
    mygraph.run(NUM_ITERATIONS);
    mygraph.end();
    std::cout << "ABC ellipse detection finished: " << (ABC_VARIANT == VI_HA ? "HA-ABC" : "SR-ABC")
              << ", " << NUM_IMAGES << " image(s), one at a time, 1 master + "
              << ABC_WORKERS << " worker tiles per image" << std::endl;
    return 0;
}
#endif
