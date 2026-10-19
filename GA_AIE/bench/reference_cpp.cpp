// ============================================================
// reference_cpp.cpp  —  timing run of the original ("normal") C++ code
//
// Compiles the unmodified reference detectors of
// ~/Desktop/GaCppPython/cppCode/scriptCode (double precision,
// std::mt19937) and times them per image like its runAll.cpp:
// edge extraction + GA run, excluding PNG loading, the evaluation and
// the figure rendering. Single-threaded. Writes per image (detection in
// image space, time in ms):
//   <out>/<Variant>_reference.csv
//
// Usage: reference_cpp --variant ConventionalGA|MicroGA|IslandGA
//                      [--images N] [--data <GaCppPython/data>] --out <dir>
// ============================================================
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "ConventionalGaDetector.h"
#include "MicroGaDetector.h"
#include "IslandGaDetector.h"
#include "GrayImage.h"
#include "../third_party/lodepng.cpp"

using namespace ega;

int main(int argc, char** argv) {
    std::string variant = "ConventionalGA", outDir = "results/timing";
    std::string dataDir = std::string(getenv("HOME") ? getenv("HOME") : "") + "/Desktop/GaCppPython/data";
    int images = 1000;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") variant = v;
        else if (k == "--images") images = std::stoi(v);
        else if (k == "--out") outDir = v;
        else if (k == "--data") dataDir = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    const int vi = variant == "ConventionalGA" ? 0 : variant == "MicroGA" ? 1
                 : variant == "IslandGA" ? 2 : -1;      // variant index of runAll.cpp
    if (vi < 0) { std::cerr << "bad --variant\n"; return 1; }

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
        auto pwi = img.withIntensity(img.edgePoints());
        const unsigned seed = static_cast<unsigned>(id * 7919u + vi * 104729u + 1u);
        DetectionResult res;
        if (vi == 0)      res = ConventionalGaDetector(pwi, img.width(), img.height(), ConvGaParams{}, seed).detect();
        else if (vi == 1) res = MicroGaDetector(pwi, img.width(), img.height(), MicroGaParams{}, seed).detect();
        else              res = IslandGaDetector(pwi, img.width(), img.height(), IslandGaParams{}, seed).detect();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        total += ms;
        char line[256];
        std::snprintf(line, sizeof line, "%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f\n", id,
                      res.detected ? 1 : 0, res.best.cx, res.best.cy, res.best.a, res.best.b,
                      res.best.theta, res.finalFitness, ms);
        csv << line;
    }
    std::printf("%s (reference C++)  images %d  detection %.3f s  (%.4f ms/image)\n",
                variant.c_str(), images, total * 1e-3, total / images);
    return 0;
}
