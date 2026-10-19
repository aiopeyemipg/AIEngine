#include <adf.h>
#include "kernels.h"
#include "project.h"
#include <iostream>

using namespace adf;

GaEllipseGraph mygraph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    mygraph.init();
    // one graph iteration = one image; its generations run on the
    // master + GA_WORKERS worker tiles in parallel
    mygraph.run(NUM_ITERATIONS);
    mygraph.end();
    static const char* names[GA_VARIANTS] = { "ConventionalGA", "MicroGA", "IslandGA" };
    std::cout << "GA ellipse detection finished: " << names[GA_VARIANT] << ", " << NUM_IMAGES
              << " image(s), one at a time, 1 master + " << GA_WORKERS
              << " worker tiles per image" << std::endl;
    return 0;
}
#endif
