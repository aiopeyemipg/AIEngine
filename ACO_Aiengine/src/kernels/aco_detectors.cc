// ============================================================
// aco_detectors.cc  —  master and worker tiles of the ACO detector
//
// One graph iteration = one image. The master runs the colony of the
// algorithm built in (aco_core.h fped_detect / stdaco_detect) with an
// engine that sends every parallel phase of a colony iteration to the
// worker tiles:
//   trail phase : workers 0..4 update and sample FPED trails h, k, a, b,
//                 theta at the same time
//   score phase : all ACO_WORKERS workers score their ants at the same time
// The detections are bit-identical to the native serial and threaded
// runs of the same code (bench/aco_bench.cpp).
// ============================================================
#include <adf.h>
#include "include.h"
#include "kernels.h"
#include "aco_core.h"
#include "aco_math.h"
#if defined(__X86SIM__)
#include <chrono>
#endif

static inline int32 aco_bits(float f) {
    union { float f; int32 i; } u;
    u.f = f;
    return u.i;
}

static inline float aco_float(int32 i) {
    union { int32 i; float f; } u;
    u.i = i;
    return u.f;
}

// Time stamp in microseconds: the tile cycle counter on the AI Engine,
// the host clock in x86sim (a functional model, not the device speed).
static inline double aco_now_us() {
#if defined(__X86SIM__)
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    return (double)aie::tile::current().cycles() / (double)AIE_CLOCK_MHZ;
#endif
}

// ------------------------------------------------------------
// Master-side engine: every call is one broadcast + one gather.
// ------------------------------------------------------------
struct AieEngine {
    input_pktstream*      in;
    output_stream<int32>* out;

    // new image: every worker gets the hits (to score), trail owners
    // also derive their trail from them
    void begin(const AcoEdgeSet& es, uint32_t seed, bool /*fped: workers know ACO_VARIANT*/) {
        writeincr(out, (int32)CMD_BEGIN);
        writeincr(out, (int32)seed);
        writeincr(out, (int32)es.n);
        for (int i = 0; i < es.n; ++i) writeincr(out, aco_bits(es.px[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, aco_bits(es.py[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, aco_bits(es.boosted[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, aco_bits(es.bright[i]));
    }

    // trail phase: five trails at once on workers 0..4
    void trails(const FpedUpdate* up, int16_t (*bins)[FPED_ANTS], float (*vals)[FPED_ANTS]) {
        writeincr(out, (int32)CMD_TRAILS);
        writeincr(out, (int32)(up ? 1 : 0));
        if (up) {
            for (int d = 0; d < 5; ++d) {
                writeincr(out, (int32)up[d].ne);
                writeincr(out, (int32)up[d].bestBin);
                writeincr(out, (int32)up[d].refine);
                writeincr(out, aco_bits(up[d].centre));
                for (int r = 0; r < FPED_ELITE; ++r) writeincr(out, (int32)up[d].bins[r]);
            }
        }
        for (int k = 0; k < 5; ++k) {
            readincr(in);                                  // packet header
            const int d = readincr(in);
            for (int a = 0; a < FPED_ANTS; ++a) bins[d][a] = (int16_t)readincr(in);
            for (int a = 0; a < FPED_ANTS; ++a) vals[d][a] = aco_float(readincr(in));
        }
    }

    // score phase: ant i on worker i % ACO_WORKERS, all at once
    void score(const float (*cand)[5], int n, float* fit) {
        writeincr(out, (int32)CMD_SCORE);
        writeincr(out, (int32)n);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < 5; ++d) writeincr(out, aco_bits(cand[i][d]));
        for (int k = 0; k < ACO_WORKERS; ++k) {
            readincr(in);                                  // packet header
            const int w = readincr(in);
            const int cnt = readincr(in);
            for (int j = 0; j < cnt; ++j) fit[w + j * ACO_WORKERS] = aco_float(readincr(in));
        }
    }
};

// ============================================================
// AcoMaster
// ============================================================
AcoMaster::AcoMaster() {}

void AcoMaster::run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
                    output_stream<int32>* toWorkers, adf::output_buffer<int32>& result) {
    const double t0 = aco_now_us();

    AieEngine eng = { fromWorkers, toWorkers };
    AcoResult res;
#if ACO_VARIANT == VI_FPED
    aco_detect_image(image.data(), ACO_VARIANT, es_, &st_, (StdAcoState*)0, eng, res);
#else
    aco_detect_image(image.data(), ACO_VARIANT, es_, (FpedState*)0, &st_, eng, res);
#endif
    writeincr(toWorkers, (int32)CMD_END);                 // workers finish the image

    const float us = (float)(aco_now_us() - t0);
    int32* r = result.data();
    r[RES_VARIANT]  = ACO_VARIANT;
    r[RES_IMAGE]    = res.image;
    r[RES_DETECTED] = res.detected;
    for (int d = 0; d < 5; ++d) r[RES_CX + d] = aco_bits(res.ell[d]);
    r[RES_FITNESS]  = aco_bits(res.fitness);
    r[RES_EVALS]    = res.evals;
    r[RES_ITERS]    = res.iters;
    r[RES_FLAGS]    = res.flags;
    r[RES_TIME_US]  = aco_bits(us);
    for (int i = 0; i < ACO_HIST_LEN; ++i) r[RES_HIST + i] = aco_bits(res.hist[i]);
}

// ============================================================
// AcoWorker
// ============================================================
AcoWorker::AcoWorker(int id) : id_(id) {}

void AcoWorker::run(input_stream<int32>* fromMaster, output_pktstream* toMaster) {
    for (;;) {
        const int cmd = readincr(fromMaster);
        if (cmd == CMD_END) return;

        if (cmd == CMD_BEGIN) {
            const uint32_t seed = (uint32_t)readincr(fromMaster);
            const int n = readincr(fromMaster);
            es_.n = n;
            es_.nPad = (n + 7) & ~7;
            for (int i = 0; i < n; ++i) es_.px[i] = aco_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.py[i] = aco_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.boosted[i] = aco_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.bright[i] = aco_float(readincr(fromMaster));
            for (int i = n; i < es_.nPad; ++i) {
                es_.px[i] = 1.0e4f; es_.py[i] = 1.0e4f; es_.boosted[i] = 0.0f; es_.bright[i] = 0.0f;
            }
#if ACO_VARIANT == VI_FPED
            if (id_ < 5) {
                AcoSeed s;
                aco_moments(es_, s);
                fped_trail_init(trail_, id_, s, seed);
            }
#else
            (void)seed;
#endif
        } else if (cmd == CMD_TRAILS) {
            const int hasUp = readincr(fromMaster);
            FpedUpdate mine = {};
            if (hasUp) {
                for (int d = 0; d < 5; ++d) {
                    FpedUpdate u;
                    u.ne = readincr(fromMaster);
                    u.bestBin = readincr(fromMaster);
                    u.refine = readincr(fromMaster);
                    u.centre = aco_float(readincr(fromMaster));
                    for (int r = 0; r < FPED_ELITE; ++r) u.bins[r] = readincr(fromMaster);
                    if (d == id_) mine = u;
                }
            }
            if (id_ < 5) {
                fped_trail_step(trail_, id_, hasUp ? &mine : (const FpedUpdate*)0, bins_, vals_);
                writeHeader(toMaster, 0, getPacketid(toMaster, 0));
                writeincr(toMaster, (int32)id_);
                for (int a = 0; a < FPED_ANTS; ++a) writeincr(toMaster, (int32)bins_[a]);
                for (int a = 0; a < FPED_ANTS; ++a)
                    writeincr(toMaster, aco_bits(vals_[a]), a == FPED_ANTS - 1);
            }
        } else if (cmd == CMD_SCORE) {
            const int n = readincr(fromMaster);
            float fit[(FPED_ANTS > STD_ANTS ? FPED_ANTS : STD_ANTS) / ACO_WORKERS + 1];
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                float c[5];
                for (int d = 0; d < 5; ++d) c[d] = aco_float(readincr(fromMaster));
                if (i % ACO_WORKERS == id_) aco_fitness(es_, c, fit[cnt++]);
            }
            writeHeader(toMaster, 0, getPacketid(toMaster, 0));
            writeincr(toMaster, (int32)id_);
            writeincr(toMaster, (int32)cnt, cnt == 0);
            for (int j = 0; j < cnt; ++j) writeincr(toMaster, aco_bits(fit[j]), j == cnt - 1);
        }
    }
}
