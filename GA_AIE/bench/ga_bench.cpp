// ============================================================
// ga_bench.cpp  —  the AIE detection code, compiled natively, timed
//
// Runs exactly the code of src/kernels/ga_algorithms.h (-DGA_NATIVE)
// over the same image records the AIE graph reads (data/ga_input.txt),
// one image after another, in one of two modes:
//
//   --threads 1   serial: one thread does all the work of every
//                 generation (the "normal C++" structure)
//   --threads T   concurrent: inside every generation the children
//                 (Conventional, Micro) or the islands (Island) are
//                 processed by T threads at the same time (master
//                 thread + T-1 workers), like the master tile and the
//                 worker tiles of the AIE graph
//
// Results are bit-identical in both modes (every task has its own
// random stream). The per-image detection time (edge extraction + GA,
// no I/O, as in the reference runAll.cpp) is written with the results:
//   <out>/ga_results.txt   packets in the x86sim output text format: header
//                          word, 13 scalars + convergence history (include.h),
//                          TLAST before the last word
//
// Usage: ga_bench --variant ConventionalGA|MicroGA|IslandGA --threads T
//                 [--images N] [--rep R] [--input data/ga_input.txt] --out <dir>
//   --rep R  seed set R (seed + R * 15485863)
// ============================================================
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif

#include "../src/kernels/ga_algorithms.h"

// ---- executors: run f(0..n-1) ----------------------------------------

struct SerialExec {
    void run(int n, const std::function<void(int)>& f) { for (int i = 0; i < n; ++i) f(i); }
};

// T threads: the caller (master) is thread 0, T-1 spinning workers are
// threads 1..T-1; task i goes to thread i % T. run() returns when every
// thread is done (the barrier at the end of a parallel phase).
class PoolExec {
public:
    explicit PoolExec(int threads) : T_(threads) {
        for (int w = 1; w < T_; ++w) pool_.emplace_back([this, w] { worker(w); });
    }
    ~PoolExec() {
        quit_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& th : pool_) th.join();
    }
    void run(int n, const std::function<void(int)>& f) {
        n_ = n; f_ = &f;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);          // start the workers
        share(0);                                              // master's share
        while (done_.load(std::memory_order_acquire) < T_ - 1) CPU_RELAX();
    }

private:
    void share(int w) { for (int i = w; i < n_; i += T_) (*f_)(i); }
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
    int n_ = 0;
    const std::function<void(int)>* f_ = nullptr;
};

// ---- engine (see ga_algorithms.h): the parallel phases on an executor --

template <typename Exec>
struct LocalEngine {
    Exec ex;
    const EdgeSet* es = nullptr;
    uint32_t seed = 0;
    GaIsland isl[ISL_ISLANDS];          // one island per worker

    template <typename... A> explicit LocalEngine(A... a) : ex(a...) {}

    void begin(const EdgeSet& e, uint32_t s) { es = &e; seed = s; }
    void conv_init(float* pop) {
        ex.run(CONV_POP, [&](int i) { conv_init_task(*es, seed, i, pop + i * IND_WORDS); });
    }
    void conv_children(const float* cur, int gen, float* out) {
        ex.run(CONV_PAIRS, [&](int k) { conv_pair_task(*es, seed, gen, k, cur, out + 2 * k * IND_WORDS); });
    }
    void micro_random(float* pop, int first, int count, int cycle) {
        ex.run(count, [&](int i) { micro_random_task(*es, seed, cycle, i, pop + (first + i) * IND_WORDS); });
    }
    void micro_children(const float* pop, int step, float progress, float* out) {
        ex.run(MICRO_PAIRS, [&](int k) { micro_pair_task(*es, seed, step, k, pop, progress, out + 2 * k * IND_WORDS); });
    }
    void island_epoch(const IslEpochCmd& cmd, IslEpochReply* rep) {
        ex.run(ISL_ISLANDS, [&](int i) { island_epoch_task(*es, seed, cmd, i, isl[i], rep[i]); });
    }
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
// The record has RESULT_SCALARS words + the history of the variant.
static void write_packet(std::ostream& o, const GaResult& r, float timeUs) {
    const int words = RESULT_SCALARS + ga_hist_len(r.variant);
    uint32_t w[RESULT_SCALARS + GA_HIST_MAX];
    w[RES_VARIANT] = (uint32_t)r.variant;
    w[RES_IMAGE] = (uint32_t)r.image;
    w[RES_DETECTED] = (uint32_t)r.detected;
    for (int d = 0; d < 5; ++d) w[RES_CX + d] = bits(r.ell[d]);
    w[RES_FITNESS] = bits(r.fitness);
    w[RES_EVALS] = (uint32_t)r.evals;
    w[RES_GENS] = (uint32_t)r.gens;
    w[RES_FLAGS] = (uint32_t)r.flags;
    w[RES_TIME_US] = bits(timeUs);
    for (int i = 0; i < ga_hist_len(r.variant); ++i) w[RES_HIST + i] = bits(r.hist[i]);
    o << 0u << '\n';
    for (int i = 0; i < words; ++i) {
        if (i == words - 1) o << "TLAST\n";
        o << w[i] << '\n';
    }
}

template <typename Engine>
static double run_all(Engine& eng, int vi, const std::vector<uint8_t>& recs, int images,
                      uint32_t seedOffset, std::ostream& out) {
    static EdgeSet es;
    static ConvGaState conv;
    static MicroGaState micro;
    static IslandGaState island;
    static float popA[CONV_POP * IND_WORDS], popB[CONV_POP * IND_WORDS];
    void* state = vi == VI_CONV ? (void*)&conv : vi == VI_MICRO ? (void*)&micro : (void*)&island;
    double total = 0.0;
    for (int i = 0; i < images; ++i) {
        GaResult r;
        auto t0 = std::chrono::steady_clock::now();
        ga_detect_image(&recs[(size_t)i * IMG_RECORD_BYTES], vi, es, state, popA, popB, eng, r, seedOffset);
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count();
        total += us;
        write_packet(out, r, (float)us);
    }
    return total;
}

int main(int argc, char** argv) {
    std::string variant = "ConventionalGA", input = "data/ga_input.txt", outDir = "results/bench";
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
    const int vi = variant == "ConventionalGA" ? VI_CONV : variant == "MicroGA" ? VI_MICRO
                 : variant == "IslandGA" ? VI_ISLAND : -1;
    if (vi < 0 || threads < 1) { std::cerr << "bad --variant / --threads\n"; return 1; }

    std::vector<uint8_t> recs;
    if (!read_records(input, images, recs)) { std::cerr << "cannot read " << images << " records from " << input << "\n"; return 1; }

    std::ofstream out(outDir + "/ga_results.txt");
    if (!out) { std::cerr << "cannot write to " << outDir << "\n"; return 1; }

    const uint32_t offset = (uint32_t)rep * 15485863u;
    auto t0 = std::chrono::steady_clock::now();
    double detectUs;
    if (threads == 1) { static LocalEngine<SerialExec> e; detectUs = run_all(e, vi, recs, images, offset, out); }
    else              { LocalEngine<PoolExec> e(threads); detectUs = run_all(e, vi, recs, images, offset, out); }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("%s  threads %d  images %d  detection %.3f s  (%.4f ms/image)  wall %.3f s\n",
                variant.c_str(), threads, images, detectUs * 1e-6, detectUs * 1e-3 / images, wall);
    return 0;
}
