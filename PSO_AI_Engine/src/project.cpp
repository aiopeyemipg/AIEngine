#include <adf.h>
#include "kernels.h"
#include "project.h"
#include <iostream>

using namespace adf;

PsoEllipseGraph mygraph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    static const char* names[PSO_VARIANTS] = { "StandardPSO_DBSCAN", "StandardPSO_noDBSCAN",
        "APSO_DBSCAN", "APSO_noDBSCAN", "SPPSO_DBSCAN", "SPPSO_noDBSCAN" };
    mygraph.init();
    // one graph iteration = one image; its PSO iterations run on the
    // master + PSO_WORKERS worker tiles in parallel
    mygraph.run(NUM_ITERATIONS);
    mygraph.end();
    std::cout << "PSO ellipse detection finished: " << names[PSO_VARIANT] << ", " << NUM_IMAGES
              << " image(s), one at a time, 1 master + " << PSO_WORKERS
              << " worker tiles per image" << std::endl;
    return 0;
}
#endif
