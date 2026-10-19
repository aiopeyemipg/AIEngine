// ============================================================
// reference_cpp.cpp  —  timing run of the original ("normal") C++ code
//
// Compiles the unmodified reference detectors of
// Particle-Swarm-Optimisation-for-Multiple-Ellipse-Detection--variantPSO/
// cppCode/scriptCode (double precision, std::mt19937, asynchronous gbest)
// with the parameters of its runAll.cpp, and times them per image:
// edge extraction + DBSCAN (even vi) + PSO. PNG loading, evaluation and
// rendering are excluded (runAll.cpp also times the evaluation; it is
// left out here so both sides time the detection only). Single-threaded.
// Writes per image (detections in image space, time in ms):
//   <out>/<Variant>_reference.csv
//
// Usage: reference_cpp --variant <vi 0..5> [--images N] --out <dir> [--data <dir>]
// ============================================================
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../third_party/lodepng.cpp"
#include "ApsoDetector.h"
#include "Dbscan.h"
#include "GrayImage.h"
#include "PsoDetector.h"
#include "SpPsoDetector.h"

using namespace epso;

static const char* NAMES[6] = { "StandardPSO_DBSCAN", "StandardPSO_noDBSCAN", "APSO_DBSCAN",
                                "APSO_noDBSCAN", "SPPSO_DBSCAN", "SPPSO_noDBSCAN" };

int main(int argc, char** argv) {
    std::string outDir = "results/timing",
                dataDir = "../Particle-Swarm-Optimisation-for-Multiple-Ellipse-Detection--variantPSO/data";
    int vi = 0, images = 1000;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") vi = std::stoi(v);
        else if (k == "--images") images = std::stoi(v);
        else if (k == "--out") outDir = v;
        else if (k == "--data") dataDir = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    if (vi < 0 || vi > 5) { std::cerr << "bad --variant\n"; return 1; }
    const int algo = vi / 2;
    const bool dbscan = (vi % 2) == 0;

    std::ofstream csv(outDir + "/" + NAMES[vi] + "_reference.csv");
    if (!csv) { std::cerr << "cannot write to " << outDir << "\n"; return 1; }
    csv << "imageID,detected,cx,cy,a,b,theta,fitness,time_ms\n";

    double total = 0.0;
    for (int id = 0; id < images; ++id) {
        GrayImage img;
        if (!img.load(dataDir + "/Ellipses/id_" + std::to_string(id) + ".png")) {
            std::cerr << "cannot read image " << id << "\n";
            return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        auto pts = img.edgePoints();
        if (dbscan) pts = Dbscan().filter(pts);
        auto pwi = img.withIntensity(pts);
        unsigned seed = static_cast<unsigned>(id * 7919u + vi * 104729u + 1u);
        DetectionResult res;
        if (algo == 0)
            res = PsoDetector(pts, pwi, img.width(), img.height(), PsoParams{100, 100, 0.5, 1.5, 2.0}, seed).detect();
        else if (algo == 1)
            res = ApsoDetector(pts, pwi, img.width(), img.height(), ApsoParams{}, seed).detect();
        else
            res = SpPsoDetector(pts, pwi, img.width(), img.height(), SpParams{}, seed).detect();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        total += ms;
        char line[256];
        std::snprintf(line, sizeof line, "%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f\n", id,
                      res.detected ? 1 : 0, res.best.cx, res.best.cy, res.best.a, res.best.b,
                      res.best.theta, res.detected ? res.finalFitness : 0.0, ms);
        csv << line;
    }
    std::printf("%s (reference C++)  images %d  detection %.3f s  (%.4f ms/image)\n",
                NAMES[vi], images, total * 1e-3, total / images);
    return 0;
}
