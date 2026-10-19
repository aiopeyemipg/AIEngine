// ============================================================
// reference_cpp.cpp  —  the "normal C++" ABC: a faithful port of
// ../ABC/hollowness-aware-ABC.py and ../ABC/standard-residual-ABC.py
//
// The research has no C++ version, so this file is the C++ baseline:
// the same algorithm, line by line, in double precision, one thread,
// with the colony updated bee by bee exactly as in run_abc() (a later
// bee sees the earlier bees' new positions; onlookers act one after
// another). Random numbers: std::mt19937_64 with the per-image seed
// imageID*7919 + vi*104729 + 1 (+ rep*15485863) instead of Python's one
// global stream, so every frame is reproducible on its own.
// The SR edge points come from src/kernels/abc_canny.h, which matches
// OpenCV exactly on all 1000 frames (scripts/check_canny.py).
//
// Times per image (point extraction + colony, no PNG decoding) and writes
//   <out>/<HA|SR>_reference.csv  imageID,detected,cx,cy,a,b,theta,fitness,time_ms
//
// Usage: reference_cpp --variant HA|SR [--images N] [--rep R]
//                      [--input data/abc_input.txt] --out <dir>
// ============================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../src/kernels/include.h"
#include "../src/kernels/abc_canny.h"

namespace {

const double PI = 3.14159265358979323846;
const double NEG_INF = -std::numeric_limits<double>::infinity();

struct Pt { double x, y, w; };
typedef std::vector<double> Bee;

struct Rand {                                   // Python's random module, per image
    std::mt19937_64 g;
    explicit Rand(unsigned long long s) : g(s) {}
    double random() { return std::uniform_real_distribution<double>(0.0, 1.0)(g); }
    double uniform(double a, double b) { return a + (b - a) * random(); }
    int randint(int a, int b) { return std::uniform_int_distribution<int>(a, b)(g); }
};

// hollowness_fitness
double fitness_ha(const Bee& e, const std::vector<Pt>& pts) {
    const double h = e[0], k = e[1], a = e[2], b = e[3], th = e[4];
    if (a < 0.5 || b < 0.5) return NEG_INF;
    const double c = std::cos(th), s = std::sin(th);
    const double bw = std::sqrt((a * c) * (a * c) + (b * s) * (b * s));
    const double bh = std::sqrt((a * s) * (a * s) + (b * c) * (b * c));
    if (h - bw < 0 || h + bw > IMG_W || k - bh < 0 || k + bh > IMG_H) return NEG_INF;
    double per = 0, in = 0;
    int nPer = 0;
    for (const Pt& p : pts) {
        const double t1 = std::pow((p.x - h) * c + (p.y - k) * s, 2) / (a * a);
        const double t2 = std::pow((p.x - h) * s - (p.y - k) * c, 2) / (b * b);
        const double eq = t1 + t2;
        if (std::fabs(eq - 1.0) < 0.35) { per += p.w; ++nPer; }
        else if (eq < 1.0 - 0.35) in += p.w;
    }
    if (!(3 <= nPer && nPer <= 500)) return NEG_INF;
    return per - 0.5 * in;
}

// compute_fitness (standard residual)
double fitness_sr(const Bee& e, const std::vector<Pt>& pts) {
    const double cx = e[0], cy = e[1], a = e[2], b = e[3], th = e[4];
    if (a <= 0 || b <= 0 || pts.empty()) return 0.0;
    const double c = std::cos(th), s = std::sin(th);
    double err = 0;
    for (const Pt& p : pts) {
        const double dx = p.x - cx, dy = p.y - cy;
        const double xr = dx * c + dy * s, yr = -dx * s + dy * c;
        err += std::fabs(xr * xr / (a * a) + yr * yr / (b * b) - 1.0);
    }
    return 1.0 / (err / pts.size() + 1e-6);
}

Bee random_bee(Rand& r, bool ha) {
    const double axMin = ha ? 1.0 : 1.5;
    double cx = r.uniform(1.0, IMG_W - 1.0);
    double cy = r.uniform(1.0, IMG_H - 1.0);
    double a  = r.uniform(axMin, IMG_W / 2.0);
    double b  = r.uniform(axMin, IMG_H / 2.0);
    double t  = r.uniform(0.0, 2 * PI);
    return { cx, cy, a, b, t };
}

void clamp_bee(Bee& e) {
    e[0] = std::max(0.5, std::min(IMG_W - 0.5, e[0]));
    e[1] = std::max(0.5, std::min(IMG_H - 0.5, e[1]));
    e[2] = std::max(0.5, std::min(IMG_W / 2.0, e[2]));
    e[3] = std::max(0.5, std::min(IMG_H / 2.0, e[3]));
    e[4] = std::fmod(e[4], 2 * PI);
    if (e[4] < 0) e[4] += 2 * PI;                       // Python % is non-negative
}

// run_abc of either script
Bee run_abc(const std::vector<Pt>& pts, bool ha, Rand& r, double& bestFit, bool& valid) {
    auto fit = [&](const Bee& e) { return ha ? fitness_ha(e, pts) : fitness_sr(e, pts); };
    const int N = NUM_BEES;
    if ((int)pts.size() < (ha ? 3 : 5)) {
        valid = false; bestFit = 0;
        return { IMG_W / 2.0, IMG_H / 2.0, IMG_W / 4.0, IMG_H / 4.0, 0.0 };
    }
    std::vector<Bee> bees(N);
    std::vector<double> f(N);
    std::vector<int> trials(N, 0);
    for (int i = 0; i < N; ++i) bees[i] = random_bee(r, ha);
    for (int i = 0; i < N; ++i) f[i] = fit(bees[i]);
    if (ha)
        for (int i = 0; i < N; ++i)
            for (int att = 0; f[i] == NEG_INF && att < 50; ++att) { bees[i] = random_bee(r, ha); f[i] = fit(bees[i]); }

    auto argbest = [&]() {
        int bi = -1;
        for (int i = 0; i < N; ++i)
            if (f[i] != NEG_INF && (bi < 0 || f[i] > f[bi])) bi = i;
        return bi;
    };
    int bi = argbest();
    Bee best = bees[bi >= 0 ? bi : 0];
    bestFit = bi >= 0 ? f[bi] : NEG_INF;

    auto try_bee = [&](int i) {
        int k = r.randint(0, N - 1);
        while (k == i) k = r.randint(0, N - 1);
        Bee nb = bees[i];
        const int j = r.randint(0, 4);
        const double phi = r.uniform(-1.0, 1.0);
        nb[j] += phi * (bees[i][j] - bees[k][j]);
        clamp_bee(nb);
        const double nf = fit(nb);
        if (nf > f[i]) { bees[i] = nb; f[i] = nf; trials[i] = 0; }
        else ++trials[i];
    };

    std::vector<double> cumul(N);
    for (int it = 0; it < MAX_ITERATIONS; ++it) {
        for (int i = 0; i < N; ++i) try_bee(i);                      // employed
        double total = 0;
        for (int i = 0; i < N; ++i) total += ha ? std::max(f[i], 0.0) : f[i];
        total += 1e-12;
        double acc = 0;
        for (int i = 0; i < N; ++i) { acc += (ha ? std::max(f[i], 0.0) : f[i]) / total; cumul[i] = acc; }
        for (int o = 0; o < N; ++o) {                                 // onlookers
            const double rr = r.random();
            int si = 0;
            while (si < N && cumul[si] < rr) ++si;                    // searchsorted
            si = std::min(si, N - 1);
            try_bee(si);
        }
        for (int i = 0; i < N; ++i)                                   // scouts
            if (trials[i] >= SCOUT_LIMIT) { bees[i] = random_bee(r, ha); f[i] = fit(bees[i]); trials[i] = 0; }
        const int cb = argbest();
        if (cb >= 0 && f[cb] > bestFit) { bestFit = f[cb]; best = bees[cb]; }
    }
    valid = bestFit != NEG_INF;
    if (!valid) bestFit = 0;
    return best;
}

} // namespace

int main(int argc, char** argv) {
    std::string variant = "HA", input = "data/abc_input.txt", outDir = "results/reference_cpp";
    int images = NUM_IMAGES, rep = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") variant = v;
        else if (k == "--images") images = std::stoi(v);
        else if (k == "--rep") rep = std::stoi(v);
        else if (k == "--input") input = v;
        else if (k == "--out") outDir = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    const bool ha = variant == "HA";
    if (!ha && variant != "SR") { std::cerr << "bad --variant (HA|SR)\n"; return 1; }
    const int vi = ha ? VI_HA : VI_SR;

    std::ifstream in(input);
    std::vector<uint8_t> recs;
    int v;
    while ((int)recs.size() < images * IMG_RECORD_BYTES && in >> v) recs.push_back((uint8_t)v);
    if ((int)recs.size() != images * IMG_RECORD_BYTES) { std::cerr << "cannot read " << input << "\n"; return 1; }

    std::ofstream csv(outDir + "/" + variant + "_reference.csv");
    if (!csv) { std::cerr << "cannot write to " << outDir << "\n"; return 1; }
    csv << "imageID,detected,cx,cy,a,b,theta,fitness,time_ms\n";

    static CannyWork cw;
    static float fx[CANNY_PIX], fy[CANNY_PIX];
    double total = 0;
    for (int n = 0; n < images; ++n) {
        const uint8_t* rec = &recs[(size_t)n * IMG_RECORD_BYTES];
        const int id = rec[0] | (rec[1] << 8);
        const uint8_t* pix = rec + IMG_HEADER_BYTES;
        auto t0 = std::chrono::steady_clock::now();

        std::vector<Pt> pts;
        if (ha) {                                    // load_points_with_intensity
            int mx = 0;
            for (int i = 0; i < IMG_PIXELS; ++i) mx = std::max(mx, (int)pix[i]);
            const double maxv = mx > 0 ? mx : 1.0;
            for (int y = 0; y < IMG_H; ++y)
                for (int x = 0; x < IMG_W; ++x)
                    if (pix[y * IMG_W + x] > 0) pts.push_back({ (double)x, (double)y, pix[y * IMG_W + x] / maxv });
        } else {                                     // extract_points
            int fb;
            const int cnt = canny_points(pix, cw.blur, cw.mag, cw.dir, fx, fy, CANNY_PIX, &fb);
            for (int i = 0; i < cnt; ++i) pts.push_back({ fx[i], fy[i], 1.0 });
        }
        Rand r((unsigned long long)id * 7919ULL + (unsigned long long)vi * 104729ULL + 1ULL
               + (unsigned long long)rep * 15485863ULL);
        double bestFit;
        bool valid;
        const Bee best = run_abc(pts, ha, r, bestFit, valid);

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total += ms;
        char line[256];
        std::snprintf(line, sizeof line, "%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f\n", id, valid ? 1 : 0,
                      best[0], best[1], best[2], best[3], best[4], bestFit, ms);
        csv << line;
    }
    std::printf("%s-ABC (reference C++)  images %d  detection %.3f s  (%.4f ms/image)\n",
                variant.c_str(), images, total * 1e-3, total / images);
    return 0;
}
