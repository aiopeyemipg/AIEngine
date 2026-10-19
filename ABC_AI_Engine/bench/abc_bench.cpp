// ============================================================
// abc_bench.cpp  —  the AIE detection code, compiled natively, timed
//
// Runs exactly the code of src/kernels/abc_core.h (-DABC_NATIVE) over
// the same image records the AIE graph reads (data/abc_input.txt), one
// image after another, in one of two modes:
//
//   --threads 1   serial: one thread does every bee task of every phase
//                 (the "normal C++" structure)
//   --threads T   concurrent: inside every ABC iteration the bee tasks of
//                 each phase (employed, onlooker, scout) run on T threads
//                 at the same time (master thread + T-1 workers), like
//                 the master tile and the worker tiles of the AIE graph
//
// Detections are bit-identical in both modes. The per-image detection
// time (point extraction incl. Canny + colony run, no I/O) is written to
//   <out>/abc_results.txt   records in the packet text format
//                           (scored by scripts/evaluate.py)
//
// Usage: abc_bench --variant HA|SR --threads T [--images N] [--rep R]
//                  [--input data/abc_input.txt] --out <dir>
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

#include "../src/kernels/abc_core.h"

// ---- engines (see abc_core.h) ----------------------------------------

// Everything on the calling thread: the serial baseline.
struct SerialEngine {
    const AbcPoints* p = nullptr;
    int vi = 0;
    uint32_t seed = 0;
    void begin(const AbcPoints& pts, uint32_t s) { p = &pts; seed = s; }
    void phase(const AbcPhase& m, int n, float (*cand)[5], float* fit, int* evals) {
        abc_task_range(*p, vi, seed, m, 0, n, 1, cand, fit, evals);
    }
};

// T threads run every phase together: the caller (master) is thread 0,
// T-1 spinning workers are threads 1..T-1; bee task t goes to thread
// t % T. A phase returns when every thread is done.
class PoolEngine {
public:
    const AbcPoints* p = nullptr;
    int vi = 0;
    uint32_t seed = 0;

    explicit PoolEngine(int threads) : T_(threads) {
        for (int w = 1; w < T_; ++w) pool_.emplace_back([this, w] { worker(w); });
    }
    ~PoolEngine() {
        quit_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& th : pool_) th.join();
    }

    void begin(const AbcPoints& pts, uint32_t s) { p = &pts; seed = s; }

    void phase(const AbcPhase& m, int n, float (*cand)[5], float* fit, int* evals) {
        m_ = &m; n_ = n; cand_ = cand; fit_ = fit; evals_ = evals;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);          // start the workers
        share(0);                                              // master's share
        while (done_.load(std::memory_order_acquire) < T_ - 1) CPU_RELAX();
    }

private:
    void share(int w) { abc_task_range(*p, vi, seed, *m_, w, n_, T_, cand_, fit_, evals_); }
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
    const AbcPhase* m_ = nullptr;
    int n_ = 0;
    float (*cand_)[5] = nullptr;
    float* fit_ = nullptr;
    int* evals_ = nullptr;
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

// One record in packet text format: header word, data words, "TLAST"
// before the last word.
static void write_packet(std::ostream& o, int vi, const AbcResult& r, float timeUs) {
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
    for (int i = 0; i < ABC_HIST_LEN; ++i) w[RES_HIST + i] = bits(r.hist[i]);
    o << 0u << '\n';
    for (int i = 0; i < RESULT_WORDS; ++i) {
        if (i == RESULT_WORDS - 1) o << "TLAST\n";
        o << w[i] << '\n';
    }
}

template <typename Engine>
static double run_all(Engine& eng, int vi, const std::vector<uint8_t>& recs, int images,
                      uint32_t seedOffset, std::ostream& out) {
    static AbcPoints pts;
    static CannyWork cw;
    static AbcState st;
    eng.vi = vi;
    double total = 0.0;
    for (int i = 0; i < images; ++i) {
        AbcResult r;
        auto t0 = std::chrono::steady_clock::now();
        abc_detect_image(&recs[(size_t)i * IMG_RECORD_BYTES], vi, pts, cw.blur, cw.mag, cw.dir, st, eng, r, seedOffset);
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count();
        total += us;
        write_packet(out, vi, r, (float)us);
    }
    return total;
}

int main(int argc, char** argv) {
    std::string variant = "HA", input = "data/abc_input.txt", outDir = "results/bench";
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
    const int vi = (variant == "HA") ? VI_HA : (variant == "SR") ? VI_SR : -1;
    if (vi < 0 || threads < 1) { std::cerr << "bad --variant (HA|SR) / --threads\n"; return 1; }

    std::vector<uint8_t> recs;
    if (!read_records(input, images, recs)) { std::cerr << "cannot read " << images << " records from " << input << "\n"; return 1; }

    std::ofstream out(outDir + "/abc_results.txt");
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
