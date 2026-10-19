// ============================================================
// pso_bench.cpp  —  the AIE detection code, compiled natively, timed
//
// Runs exactly the code of src/kernels/pso_core.h (-DPSO_NATIVE) over
// the same image records the AIE graph reads (data/pso_input.txt),
// one image after another, in one of two modes:
//
//   --threads 1   serial: one thread moves and scores every particle
//                 of every PSO iteration (the "normal C++" structure)
//   --threads T   concurrent: inside every PSO iteration the particles
//                 are moved and scored by T threads at the same time
//                 (master thread + T-1 workers, particle i on thread
//                 i % T), like the master tile and the worker tiles of
//                 the AIE graph
//
// Results are identical in both modes (per-particle random streams,
// first-index tie breaks). The per-image detection time (edges + DBSCAN
// + seeding + swarm run, no I/O) is written with the detections to
//   <out>/pso_results.txt   packets: header word + RESULT_WORDS words
//                           (scored by scripts/evaluate.py)
//
// Usage: pso_bench --variant <vi 0..5> --threads T [--images N] [--rep R]
//                  [--input data/pso_input.txt] --out <dir>
//   --rep R  seed set R (seed + R * 15485863)
// ============================================================
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
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

#include "../src/kernels/pso_core.h"

// ---- engines (see pso_core.h) ----------------------------------------

// T shards; shard w owns particles w, w + T, ... . run(f) executes
// f(w) for every shard: on the calling thread only (T = 1), or on the
// master thread (w = 0) and T-1 spinning workers at the same time.
// run() returns when every shard is done (the barrier of one phase).
class Engine {
public:
    int algo = ALGO_STD;

    explicit Engine(int threads) : T_(threads), shards_(threads), bests_(threads) {
        for (int w = 1; w < T_; ++w) pool_.emplace_back([this, w] { worker(w); });
    }
    ~Engine() {
        quit_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& th : pool_) th.join();
    }

    void begin(const EdgeSet& es, uint32_t seed) { es_ = &es; seed_ = seed; }

    void init(const float (*pos)[5], int P, float* score) {
        ph_ = PH_INIT; pos_ = pos; P_ = P; score_ = score;
        run();
    }
    void step(int it, int P, const float* gbest, PsoBest& b) {
        ph_ = PH_STEP; it_ = it; P_ = P; gbest_ = gbest;
        run();
        b = bests_[0];
        for (int w = 1; w < T_; ++w) pso_best_merge(b, bests_[w]);
    }
    void step_sp(int it, const float (*tabP)[5], const float* tabS, const uint8_t* reset,
                 float (*newP)[5], float* newS) {
        ph_ = PH_STEP_SP; it_ = it; tabP_ = tabP; tabS_ = tabS; reset_ = reset;
        newP_ = newP; newS_ = newS;
        run();
    }

private:
    enum { PH_INIT, PH_STEP, PH_STEP_SP };

    void share(int w) {
        PsoShard& s = shards_[w];
        if (ph_ == PH_INIT)
            pso_shard_init(s, *es_, w, T_, pos_, P_, score_);
        else if (ph_ == PH_STEP)
            pso_shard_step(s, *es_, algo, seed_, it_, w, T_, P_, gbest_, bests_[w]);
        else
            pso_shard_step_sp(s, *es_, seed_, it_, w, T_, tabP_, tabS_, reset_, newP_, newS_);
    }
    void run() {
        if (T_ == 1) { share(0); return; }
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);          // start the workers
        share(0);                                              // master's shard
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
    std::vector<PsoShard> shards_;
    std::vector<PsoBest> bests_;
    std::vector<std::thread> pool_;
    std::atomic<unsigned> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> quit_{false};

    int ph_ = PH_INIT;
    const EdgeSet* es_ = nullptr;
    uint32_t seed_ = 0;
    int it_ = 0, P_ = 0;
    const float (*pos_)[5] = nullptr;
    float* score_ = nullptr;
    const float* gbest_ = nullptr;
    const float (*tabP_)[5] = nullptr;
    const float* tabS_ = nullptr;
    const uint8_t* reset_ = nullptr;
    float (*newP_)[5] = nullptr;
    float* newS_ = nullptr;
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
static void write_packet(std::ostream& o, const PsoResult& r, float timeUs) {
    const bool detected = pso_valid(r.score);
    uint32_t w[RESULT_WORDS];
    w[RES_VARIANT] = (uint32_t)r.variant;
    w[RES_IMAGE] = (uint32_t)r.image;
    w[RES_DETECTED] = detected ? 1u : 0u;
    for (int d = 0; d < 5; ++d) w[RES_CX + d] = bits(r.best[d]);
    w[RES_FITNESS] = bits(detected ? r.score : PSO_NEG_INF);
    w[RES_EVALS] = (uint32_t)r.evals;
    w[RES_ITERS] = (uint32_t)r.iters;
    w[RES_FLAGS] = (uint32_t)r.flags;
    w[RES_TIME_US] = bits(timeUs);
    for (int it = 0; it < PSO_HIST_LEN; ++it) w[RES_HIST + it] = bits(r.hist[it]);
    o << 0u << '\n';
    for (int i = 0; i < RESULT_WORDS; ++i) {
        if (i == RESULT_WORDS - 1) o << "TLAST\n";
        o << w[i] << '\n';
    }
}

int main(int argc, char** argv) {
    std::string input = "data/pso_input.txt", outDir = "results/bench";
    int vi = 0, threads = 1, images = NUM_IMAGES, rep = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") vi = std::stoi(v);
        else if (k == "--threads") threads = std::stoi(v);
        else if (k == "--images") images = std::stoi(v);
        else if (k == "--rep") rep = std::stoi(v);
        else if (k == "--input") input = v;
        else if (k == "--out") outDir = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    if (vi < 0 || vi >= PSO_VARIANTS || threads < 1) { std::cerr << "bad --variant / --threads\n"; return 1; }

    std::vector<uint8_t> recs;
    if (!read_records(input, images, recs)) {
        std::cerr << "cannot read " << images << " records from " << input << "\n";
        return 1;
    }
    std::ofstream out(outDir + "/pso_results.txt");
    if (!out) { std::cerr << "cannot write to " << outDir << "\n"; return 1; }

    static PsoMasterState m;
    Engine eng(threads);
    eng.algo = vi / 2;
    double total = 0.0;
    auto tw = std::chrono::steady_clock::now();
    for (int i = 0; i < images; ++i) {
        PsoResult r;
        auto t0 = std::chrono::steady_clock::now();
        pso_detect(vi, &recs[(size_t)i * IMG_RECORD_BYTES], m, eng, r, (uint32_t)rep * 15485863u);
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count();
        total += us;
        write_packet(out, r, (float)us);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    std::printf("vi %d  threads %d  images %d  detection %.3f s  (%.4f ms/image)  wall %.3f s\n",
                vi, threads, images, total * 1e-6, total * 1e-3 / images, wall);
    return 0;
}
