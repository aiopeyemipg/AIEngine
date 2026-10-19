// ============================================================
// aco_bench.cpp  —  the AIE detection code, compiled natively, timed
//
// Runs exactly the code of src/kernels/aco_core.h (-DACO_NATIVE) over
// the same image records the AIE graph reads (data/aco_input.txt),
// one image after another, in one of two modes:
//
//   --threads 1   serial: one thread does all the work of every colony
//                 iteration (the "normal C++" structure)
//   --threads T   concurrent: inside every colony iteration the five
//                 trails, then the ants, are processed by T threads at
//                 the same time (master thread + T-1 workers), like the
//                 master tile and the worker tiles of the AIE graph
//
// Results are bit-identical in both modes (the fitness is pure). The
// per-image detection time (edge extraction + moment seed + colony run,
// no I/O, as in the reference runAll.cpp) is written to
//   <out>/aco_results.txt   packets in the x86sim output format
//                           (scored by scripts/evaluate.py)
//
// Usage: aco_bench --variant FPED|StdACO --threads T [--images N] [--rep R]
//                  [--input data/aco_input.txt] --out <dir>
//   --rep R  seed set R of the repeat study (seed + R * 15485863)
// ============================================================
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif

#include "../src/kernels/aco_core.h"

// ---- engines (see aco_core.h) ----------------------------------------

// Everything on the calling thread: the serial baseline.
struct SerialEngine {
    const AcoEdgeSet* es = nullptr;
    FpedLocalTrails t;
    void begin(const AcoEdgeSet& e, uint32_t seed, bool fped) { if (fped) t.init(e, seed); }
    void trails(const FpedUpdate* up, int16_t (*bins)[FPED_ANTS], float (*vals)[FPED_ANTS]) {
        for (int d = 0; d < 5; ++d) fped_trail_step(t.tr[d], d, up ? &up[d] : nullptr, bins[d], vals[d]);
    }
    void score(const float (*cand)[5], int n, float* fit) { aco_score_range(*es, cand, 0, n, 1, fit); }
};

// T threads run every phase together: the caller (master) is thread 0,
// T-1 spinning workers are threads 1..T-1. Task i of a phase goes to
// thread i % T: in the trail phase trail d is task d, in the score
// phase ant i is task i. A phase returns when every thread is done
// (the barrier between two phases).
class PoolEngine {
public:
    const AcoEdgeSet* es = nullptr;
    FpedLocalTrails t;

    explicit PoolEngine(int threads) : T_(threads) {
        for (int w = 1; w < T_; ++w) pool_.emplace_back([this, w] { worker(w); });
    }
    ~PoolEngine() {
        quit_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& th : pool_) th.join();
    }

    void begin(const AcoEdgeSet& e, uint32_t seed, bool fped) { if (fped) t.init(e, seed); }

    void trails(const FpedUpdate* up, int16_t (*bins)[FPED_ANTS], float (*vals)[FPED_ANTS]) {
        phase_ = PH_TRAILS; up_ = up; bins_ = bins; vals_ = vals;
        run();
    }
    void score(const float (*cand)[5], int n, float* fit) {
        phase_ = PH_SCORE; cand_ = cand; n_ = n; fit_ = fit;
        run();
    }

private:
    enum { PH_TRAILS, PH_SCORE };

    void share(int w) {
        if (phase_ == PH_TRAILS) {
            for (int d = w; d < 5; d += T_)
                fped_trail_step(t.tr[d], d, up_ ? &up_[d] : nullptr, bins_[d], vals_[d]);
        } else {
            aco_score_range(*es, cand_, w, n_, T_, fit_);
        }
    }
    void run() {
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);          // start the workers
        share(0);                                              // master's share
        while (done_.load(std::memory_order_acquire) < T_ - 1) CPU_RELAX();
    }
    void worker(int w) {
        unsigned seen = 0;
        for (;;) {
            unsigned g;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) CPU_RELAX();
            seen = g;
            if (quit_.load(std::memory_order_acquire)) return;
            share(w);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    int T_;
    std::vector<std::thread> pool_;
    std::atomic<unsigned> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> quit_{false};
    int phase_ = PH_SCORE;
    const FpedUpdate* up_ = nullptr;
    int16_t (*bins_)[FPED_ANTS] = nullptr;
    float (*vals_)[FPED_ANTS] = nullptr;
    const float (*cand_)[5] = nullptr;
    int n_ = 0;
    float* fit_ = nullptr;
};

// ---- I/O --------------------------------------------------------------

static bool read_records(const std::string& path, int images, std::vector<uint8_t>& recs) {
    std::ifstream f(path);
    if (!f) return false;
    recs.clear();
    int v;
    while ((int)recs.size() < images * IMG_RECORD_BYTES && f >> v) recs.push_back((uint8_t)v);
    return (int)recs.size() == images * IMG_RECORD_BYTES;
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// One packet in the x86sim PLIO text format: header word, data words,
// "TLAST" before the last word.
static void write_packet(std::ostream& o, int vi, const AcoResult& r, float timeUs) {
    uint32_t w[RESULT_WORDS];
    w[RES_VARIANT] = (uint32_t)vi;
    w[RES_IMAGE] = (uint32_t)r.image;
    w[RES_DETECTED] = (uint32_t)r.detected;
    for (int d = 0; d < 5; ++d) w[RES_CX + d] = bits(r.ell[d]);
    w[RES_FITNESS] = bits(r.fitness);
    w[RES_EVALS] = (uint32_t)r.evals;
    w[RES_ITERS] = (uint32_t)r.iters;
    w[RES_FLAGS] = (uint32_t)r.flags;
    w[RES_TIME_US] = bits(timeUs);
    for (int i = 0; i < ACO_HIST_LEN; ++i) w[RES_HIST + i] = bits(r.hist[i]);
    o << 0u << '\n';
    for (int i = 0; i < RESULT_WORDS; ++i) {
        if (i == RESULT_WORDS - 1) o << "TLAST\n";
        o << w[i] << '\n';
    }
}

template <typename Engine>
static double run_all(Engine& eng, int vi, const std::vector<uint8_t>& recs, int images,
                      uint32_t seedOffset, std::ostream& out) {
    static AcoEdgeSet es;
    static FpedState fped;
    static StdAcoState stdaco;
    eng.es = &es;
    double total = 0.0;
    for (int i = 0; i < images; ++i) {
        AcoResult r;
        auto t0 = std::chrono::steady_clock::now();
        aco_detect_image(&recs[(size_t)i * IMG_RECORD_BYTES], vi, es, &fped, &stdaco, eng, r, seedOffset);
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count();
        total += us;
        write_packet(out, vi, r, (float)us);
    }
    return total;
}

int main(int argc, char** argv) {
    std::string variant = "FPED", input = "data/aco_input.txt", outDir = "results/bench";
    int threads = 1, images = NUM_IMAGES, rep = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") variant = v;
        else if (k == "--threads") threads = std::stoi(v);
        else if (k == "--images") images = std::stoi(v);
        else if (k == "--rep") rep = std::stoi(v);
        else if (k == "--input") input = v;
        else if (k == "--out") outDir = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    const int vi = (variant == "FPED") ? VI_FPED : (variant == "StdACO") ? VI_STDACO : -1;
    if (vi < 0 || threads < 1) { std::cerr << "bad --variant / --threads\n"; return 1; }

    std::vector<uint8_t> recs;
    if (!read_records(input, images, recs)) { std::cerr << "cannot read " << images << " records from " << input << "\n"; return 1; }

    std::ofstream out(outDir + "/aco_results.txt");
    if (!out) { std::cerr << "cannot write to " << outDir << "\n"; return 1; }

    auto t0 = std::chrono::steady_clock::now();
    double detectUs;
    if (threads == 1) { static SerialEngine e; detectUs = run_all(e, vi, recs, images, (uint32_t)rep * 15485863u, out); }
    else              { PoolEngine e(threads); detectUs = run_all(e, vi, recs, images, (uint32_t)rep * 15485863u, out); }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("%s  threads %d  images %d  detection %.3f s  (%.4f ms/image)  wall %.3f s\n",
                variant.c_str(), threads, images, detectUs * 1e-6, detectUs * 1e-3 / images, wall);
    return 0;
}
