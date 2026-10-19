// ============================================================
// reference_cpp.cpp  —  timing run of the original ("normal") C++ code
//
// Compiles the unmodified reference detectors of ACO/cppCode/scriptCode
// (double precision, std::mt19937_64, std::discrete_distribution) and
// times them per image exactly like its runAll.cpp: hit extraction +
// moment seed + colony run, excluding PNG loading. Single-threaded.
// Writes per image (detections in image space, time in ms):
//   <out>/<Variant>_reference.csv
//
// Usage: reference_cpp --variant FPED|StdACO [--images N] --out <dir>
// (run from anywhere; the ACO dataset is found from ACO/data)
// ============================================================
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "Config.h"
#include "Ellipse.h"
#include "GrayImage.h"
#include "Fitness.h"
#include "Moments.h"
#include "Detectors.h"
#include "../third_party/lodepng.cpp"

using namespace eaco;

int main(int argc, char** argv) {
    std::string variant = "FPED", outDir = "results/reference_cpp", dataDir = "../ACO/data";
    int images = 1000;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") variant = v;
        else if (k == "--images") images = std::stoi(v);
        else if (k == "--out") outDir = v;
        else if (k == "--data") dataDir = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    const bool fped = variant == "FPED";
    const int vi = fped ? 0 : 1;            // variant index of runAll.cpp

    std::ofstream csv(outDir + "/" + variant + "_reference.csv");
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
        auto pts = img.withIntensity(img.edgePoints());
        std::mt19937_64 rng((unsigned long long)id * 7919ULL +
                            (unsigned long long)vi * 104729ULL + 1ULL);
        DetectionResult res;
        if (fped) { FpedParams p; res = FpedDetector(pts, img.width(), img.height(), p, rng).detect(); }
        else      { StdAcoParams p; res = StdAcoDetector(pts, img.width(), img.height(), p, rng).detect(); }
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
                variant.c_str(), images, total * 1e-3, total / images);
    return 0;
}
