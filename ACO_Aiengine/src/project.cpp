#include <adf.h>
#include "kernels.h"
#include "project.h"
#include <iostream>

using namespace adf;

AcoEllipseGraph mygraph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    mygraph.init();
    // one graph iteration = one image; its colony iterations run on
    // the master + ACO_WORKERS worker tiles in parallel
    mygraph.run(NUM_ITERATIONS);
    mygraph.end();
    std::cout << "ACO ellipse detection finished: " << (ACO_VARIANT == VI_FPED ? "FPED" : "StdACO")
              << ", " << NUM_IMAGES << " image(s), one at a time, 1 master + "
              << ACO_WORKERS << " worker tiles per image" << std::endl;
    return 0;
}
#endif
