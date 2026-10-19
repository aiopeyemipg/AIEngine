// ============================================================
// ga_detectors.cc  —  master and worker tiles of the GA detector
//
// One graph iteration = one image. The master runs the GA variant built
// in (ga_algorithms.h ga_detect_conv / _micro / _island) with an engine
// that sends every parallel phase of a generation to the worker tiles:
//   Conventional : workers build + score their child pairs of a generation
//   Micro        : workers build + score their child pairs of a step
//   Island       : worker i evolves island i for one epoch
// The detections match the native serial and threaded runs of the same
// code (bench/ga_bench.cpp); x86sim may differ in the last float bit
// (aie::sqrt emulation vs std::sqrt).
// ============================================================
#include <adf.h>
#include "include.h"
#include "kernels.h"
#include "ga_algorithms.h"
#if defined(__X86SIM__)
#include <chrono>
#endif

static inline int32 ga_bits(float f) {
    union { float f; int32 i; } u;
    u.f = f;
    return u.i;
}

static inline float ga_float(int32 i) {
    union { int32 i; float f; } u;
    u.i = i;
    return u.f;
}

// Time stamp in microseconds: the tile cycle counter on the AI Engine,
// the host clock in x86sim (a functional model, not the device speed).
static inline double ga_now_us() {
#if defined(__X86SIM__)
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    return (double)aie::tile::current().cycles() / (double)AIE_CLOCK_MHZ;
#endif
}

static inline void send_inds(output_stream<int32>* out, const float* p, int n) {
    for (int i = 0; i < n * IND_WORDS; ++i) writeincr(out, ga_bits(p[i]));
}

// ------------------------------------------------------------
// Master-side engine: every call is one broadcast + one gather.
// ------------------------------------------------------------
struct AieEngine {
    input_pktstream*      in;
    output_stream<int32>* out;

    // Gather one "id, count, items" packet per worker; item j of worker w
    // is task w + j * GA_WORKERS and holds `itemWords` words, of which the
    // first `keep(task)` go to dst + task * itemWords.
    template <typename Keep>
    void gather(float* dst, int itemWords, Keep keep) {
        for (int k = 0; k < GA_WORKERS; ++k) {
            readincr(in);                                    // packet header
            const int w = readincr(in);
            const int cnt = readincr(in);
            for (int j = 0; j < cnt; ++j) {
                const int task = w + j * GA_WORKERS;
                const int kw = keep(task);
                for (int q = 0; q < itemWords; ++q) {
                    const float v = ga_float(readincr(in));
                    if (q < kw) dst[task * itemWords + q] = v;
                }
            }
        }
    }

    void begin(const EdgeSet& es, uint32_t seed) {
        writeincr(out, (int32)CMD_BEGIN);
        writeincr(out, (int32)seed);
        writeincr(out, (int32)es.n);
        for (int i = 0; i < es.n; ++i) writeincr(out, ga_bits(es.px[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, ga_bits(es.py[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, ga_bits(es.boosted[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, ga_bits(es.bright[i]));
    }

    void conv_init(float* pop) {
        writeincr(out, (int32)CMD_CONV_INIT);
        gather(pop, IND_WORDS, [](int) { return IND_WORDS; });
    }

    void conv_children(const float* cur, int gen, float* outPop) {
        writeincr(out, (int32)CMD_CONV_GEN);
        writeincr(out, (int32)gen);
        send_inds(out, cur, CONV_POP);
        gather(outPop, 2 * IND_WORDS,
               [](int k) { return (2 * k + 1 < CONV_CHILDREN ? 2 : 1) * IND_WORDS; });
    }

    void micro_random(float* pop, int first, int count, int cycle) {
        writeincr(out, (int32)CMD_MICRO_RAND);
        writeincr(out, (int32)count);
        writeincr(out, (int32)cycle);
        gather(pop + first * IND_WORDS, IND_WORDS, [](int) { return IND_WORDS; });
    }

    void micro_children(const float* pop, int step, float progress, float* outPop) {
        writeincr(out, (int32)CMD_MICRO_GEN);
        writeincr(out, (int32)step);
        writeincr(out, ga_bits(progress));
        send_inds(out, pop, MICRO_POP);
        gather(outPop, 2 * IND_WORDS,
               [](int k) { return (2 * k + 1 < MICRO_CHILDREN ? 2 : 1) * IND_WORDS; });
    }

    void island_epoch(const IslEpochCmd& cmd, IslEpochReply* rep) {
        writeincr(out, (int32)CMD_ISL_EPOCH);
        writeincr(out, (int32)cmd.restart);
        writeincr(out, (int32)cmd.epoch);
        writeincr(out, (int32)cmd.init);
        writeincr(out, (int32)cmd.haveSeed);
        for (int d = 0; d < 5; ++d) writeincr(out, ga_bits(cmd.seed[d]));
        writeincr(out, (int32)cmd.migrate);
        send_inds(out, &cmd.migrants[0][0][0], ISL_ISLANDS * ISL_MIG_SIZE);
        for (int k = 0; k < ISL_ISLANDS; ++k) {
            readincr(in);                                    // packet header
            const int i = readincr(in);
            rep[i].evals = readincr(in);
            rep[i].gens = readincr(in);
            float* t = &rep[i].top[0][0];
            for (int q = 0; q < ISL_MIG_SIZE * IND_WORDS; ++q) t[q] = ga_float(readincr(in));
            for (int g = 0; g < ISL_MIG_INTERVAL; ++g) rep[i].hist[g] = ga_float(readincr(in));
        }
    }
};

// ============================================================
// GaMaster
// ============================================================
#if GA_VARIANT == VI_CONV
GaMaster::GaMaster(float (&popA)[CONV_POP * IND_WORDS], float (&popB)[CONV_POP * IND_WORDS])
    : popA_(popA), popB_(popB) {}
#else
GaMaster::GaMaster() {}
#endif

void GaMaster::run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
                   output_stream<int32>* toWorkers, adf::output_buffer<int32>& result) {
    const double t0 = ga_now_us();

    AieEngine eng = { fromWorkers, toWorkers };
    GaResult res;
#if GA_VARIANT == VI_CONV
    ga_detect_image(image.data(), GA_VARIANT, es_, &st_, popA_, popB_, eng, res);
#else
    ga_detect_image(image.data(), GA_VARIANT, es_, &st_, (float*)0, (float*)0, eng, res);
#endif
    writeincr(toWorkers, (int32)CMD_END);                   // workers finish the image

    const float us = (float)(ga_now_us() - t0);
    int32* r = result.data();
    r[RES_VARIANT]  = GA_VARIANT;
    r[RES_IMAGE]    = res.image;
    r[RES_DETECTED] = res.detected;
    for (int d = 0; d < 5; ++d) r[RES_CX + d] = ga_bits(res.ell[d]);
    r[RES_FITNESS]  = ga_bits(res.fitness);
    r[RES_EVALS]    = res.evals;
    r[RES_GENS]     = res.gens;
    r[RES_FLAGS]    = res.flags;
    r[RES_TIME_US]  = ga_bits(us);
    for (int i = 0; i < GA_HIST_LEN; ++i) r[RES_HIST + i] = ga_bits(res.hist[i]);
}

// ============================================================
// GaWorker
// ============================================================
#if GA_VARIANT == VI_CONV
GaWorker::GaWorker(int id, float (&pop)[CONV_POP * IND_WORDS]) : id_(id), seed_(0), pop_(pop) {}
#else
GaWorker::GaWorker(int id) : id_(id), seed_(0) {}
#endif

static inline void read_inds(input_stream<int32>* in, float* p, int n) {
    for (int i = 0; i < n * IND_WORDS; ++i) p[i] = ga_float(readincr(in));
}

// Reply "id, count, items": the tasks id, id + W, ... < nTasks; `task`
// writes the itemWords words of one task into buf.
template <typename Task>
static inline void reply_tasks(output_pktstream* out, int id, int nTasks, int itemWords, Task task) {
    const int cnt = nTasks > id ? (nTasks - id + GA_WORKERS - 1) / GA_WORKERS : 0;
    writeHeader(out, 0, getPacketid(out, 0));
    writeincr(out, (int32)id);
    writeincr(out, (int32)cnt, cnt == 0);
    float buf[2 * IND_WORDS];
    for (int j = 0; j < cnt; ++j) {
        task(id + j * GA_WORKERS, buf);
        for (int q = 0; q < itemWords; ++q)
            writeincr(out, ga_bits(buf[q]), j == cnt - 1 && q == itemWords - 1);
    }
}

void GaWorker::run(input_stream<int32>* fromMaster, output_pktstream* toMaster) {
    for (;;) {
        const int cmd = readincr(fromMaster);
        if (cmd == CMD_END) return;

        if (cmd == CMD_BEGIN) {
            seed_ = (uint32_t)readincr(fromMaster);
            const int n = readincr(fromMaster);
            es_.n = n;
            es_.nPad = (n + 7) & ~7;
            for (int i = 0; i < n; ++i) es_.px[i] = ga_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.py[i] = ga_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.boosted[i] = ga_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.bright[i] = ga_float(readincr(fromMaster));
            for (int i = n; i < es_.nPad; ++i) {
                es_.px[i] = 1.0e4f; es_.py[i] = 1.0e4f; es_.boosted[i] = 0.0f; es_.bright[i] = 0.0f;
            }
        }
#if GA_VARIANT == VI_CONV
        else if (cmd == CMD_CONV_INIT) {
            reply_tasks(toMaster, id_, CONV_POP, IND_WORDS,
                        [&](int i, float* b) { conv_init_task(es_, seed_, i, b); });
        } else if (cmd == CMD_CONV_GEN) {
            const int gen = readincr(fromMaster);
            read_inds(fromMaster, pop_, CONV_POP);
            reply_tasks(toMaster, id_, CONV_PAIRS, 2 * IND_WORDS,
                        [&](int k, float* b) { conv_pair_task(es_, seed_, gen, k, pop_, b); });
        }
#elif GA_VARIANT == VI_MICRO
        else if (cmd == CMD_MICRO_RAND) {
            const int count = readincr(fromMaster);
            const int cycle = readincr(fromMaster);
            reply_tasks(toMaster, id_, count, IND_WORDS,
                        [&](int i, float* b) { micro_random_task(es_, seed_, cycle, i, b); });
        } else if (cmd == CMD_MICRO_GEN) {
            const int step = readincr(fromMaster);
            const float progress = ga_float(readincr(fromMaster));
            read_inds(fromMaster, pop_, MICRO_POP);
            reply_tasks(toMaster, id_, MICRO_PAIRS, 2 * IND_WORDS,
                        [&](int k, float* b) { micro_pair_task(es_, seed_, step, k, pop_, progress, b); });
        }
#else
        else if (cmd == CMD_ISL_EPOCH) {
            cmd_.restart = readincr(fromMaster);
            cmd_.epoch = readincr(fromMaster);
            cmd_.init = readincr(fromMaster);
            cmd_.haveSeed = readincr(fromMaster);
            for (int d = 0; d < 5; ++d) cmd_.seed[d] = ga_float(readincr(fromMaster));
            cmd_.migrate = readincr(fromMaster);
            read_inds(fromMaster, &cmd_.migrants[0][0][0], ISL_ISLANDS * ISL_MIG_SIZE);
            if (id_ < ISL_ISLANDS) {
                island_epoch_task(es_, seed_, cmd_, id_, isl_, rep_);
                writeHeader(toMaster, 0, getPacketid(toMaster, 0));
                writeincr(toMaster, (int32)id_);
                writeincr(toMaster, (int32)rep_.evals);
                writeincr(toMaster, (int32)rep_.gens);
                const float* t = &rep_.top[0][0];
                for (int q = 0; q < ISL_MIG_SIZE * IND_WORDS; ++q) writeincr(toMaster, ga_bits(t[q]));
                for (int g = 0; g < ISL_MIG_INTERVAL; ++g)
                    writeincr(toMaster, ga_bits(rep_.hist[g]), g == ISL_MIG_INTERVAL - 1);
            }
        }
#endif
    }
}
