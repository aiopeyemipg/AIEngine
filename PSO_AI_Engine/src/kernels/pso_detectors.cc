// ============================================================
// pso_detectors.cc  —  master and worker tiles of the PSO detector
//
// One graph iteration = one image. The master runs the PSO of the
// variant built in (pso_core.h pso_detect) with an engine that sends
// every parallel step to the worker tiles: inside each PSO iteration
// all PSO_WORKERS workers move and score their particles at the same
// time, and the master reduces their bests to the new gbest.
// The detections match the native serial and threaded runs of the same
// code (bench/pso_bench.cpp) up to the rounding of the square root.
// ============================================================
#include <adf.h>
#include "include.h"
#include "kernels.h"
#include "pso_core.h"
#include "pso_math.h"
#if defined(__X86SIM__)
#include <chrono>
#endif

static inline int32 pso_bits(float f) {
    union { float f; int32 i; } u;
    u.f = f;
    return u.i;
}

static inline float pso_float(int32 i) {
    union { int32 i; float f; } u;
    u.i = i;
    return u.f;
}

// Time stamp in microseconds: the tile cycle counter on the AI Engine,
// the host clock in x86sim (a functional model, not the device speed).
static inline double pso_now_us() {
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

    void begin(const EdgeSet& es, uint32_t seed) {
        writeincr(out, (int32)CMD_BEGIN);
        writeincr(out, (int32)seed);
        writeincr(out, (int32)es.n);
        for (int i = 0; i < es.n; ++i) writeincr(out, pso_bits(es.px[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, pso_bits(es.py[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, pso_bits(es.boosted[i]));
        for (int i = 0; i < es.n; ++i) writeincr(out, pso_bits(es.bright[i]));
    }

    // initial swarm: every worker scores its particles
    void init(const float (*pos)[5], int P, float* score) {
        writeincr(out, (int32)CMD_INIT);
        writeincr(out, (int32)P);
        for (int i = 0; i < P; ++i)
            for (int d = 0; d < 5; ++d) writeincr(out, pso_bits(pos[i][d]));
        for (int k = 0; k < PSO_WORKERS; ++k) {
            readincr(in);                                  // packet header
            const int w = readincr(in);
            const int cnt = readincr(in);
            for (int j = 0; j < cnt; ++j) score[w + j * PSO_WORKERS] = pso_float(readincr(in));
        }
    }

    // Std / APSO iteration: all particles at once, reduce to the best
    void step(int it, int P, const float* gbest, PsoBest& b) {
        writeincr(out, (int32)CMD_STEP);
        writeincr(out, (int32)it);
        writeincr(out, (int32)P);
        for (int d = 0; d < 5; ++d) writeincr(out, pso_bits(gbest[d]));
        b.idx = -1;
        b.score = PSO_NEG_INF;
        for (int k = 0; k < PSO_WORKERS; ++k) {
            readincr(in);                                  // packet header
            readincr(in);                                  // worker id
            PsoBest wb;
            wb.idx = readincr(in);
            wb.score = pso_float(readincr(in));
            for (int d = 0; d < 5; ++d) wb.pos[d] = pso_float(readincr(in));
            pso_best_merge(b, wb);
        }
    }

    // SP-PSO iteration: send the ring table, collect the new personal bests
    void step_sp(int it, const float (*tabP)[5], const float* tabS, const uint8_t* reset,
                 float (*newP)[5], float* newS) {
        writeincr(out, (int32)CMD_STEP_SP);
        writeincr(out, (int32)it);
        for (int i = 0; i < SP_PARTICLES; ++i) {
            writeincr(out, pso_bits(tabS[i]));
            for (int d = 0; d < 5; ++d) writeincr(out, pso_bits(tabP[i][d]));
            writeincr(out, (int32)reset[i]);
        }
        for (int k = 0; k < PSO_WORKERS; ++k) {
            readincr(in);                                  // packet header
            const int w = readincr(in);
            const int cnt = readincr(in);
            for (int j = 0; j < cnt; ++j) {
                const int i = w + j * PSO_WORKERS;
                newS[i] = pso_float(readincr(in));
                for (int d = 0; d < 5; ++d) newP[i][d] = pso_float(readincr(in));
            }
        }
    }
};

// ============================================================
// PsoMaster
// ============================================================
PsoMaster::PsoMaster() {}

void PsoMaster::run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
                    output_stream<int32>* toWorkers, adf::output_buffer<int32>& result) {
    const double t0 = pso_now_us();

    AieEngine eng = { fromWorkers, toWorkers };
    PsoResult res;
    pso_detect(PSO_VARIANT, image.data(), m_, eng, res);
    writeincr(toWorkers, (int32)CMD_END);                 // workers finish the image

    const float us = (float)(pso_now_us() - t0);
    const bool detected = pso_valid(res.score);
    int32* r = result.data();
    r[RES_VARIANT]  = PSO_VARIANT;
    r[RES_IMAGE]    = res.image;
    r[RES_DETECTED] = detected ? 1 : 0;
    for (int d = 0; d < 5; ++d) r[RES_CX + d] = pso_bits(res.best[d]);
    r[RES_FITNESS]  = pso_bits(detected ? res.score : PSO_NEG_INF);
    r[RES_EVALS]    = res.evals;
    r[RES_ITERS]    = res.iters;
    r[RES_FLAGS]    = res.flags;
    r[RES_TIME_US]  = pso_bits(us);
    for (int it = 0; it < PSO_HIST_LEN; ++it) r[RES_HIST + it] = pso_bits(res.hist[it]);
}

// ============================================================
// PsoWorker
// ============================================================
PsoWorker::PsoWorker(int id) : id_(id), seed_(0) {}

void PsoWorker::run(input_stream<int32>* fromMaster, output_pktstream* toMaster) {
    const int W = PSO_WORKERS;
    for (;;) {
        const int cmd = readincr(fromMaster);
        if (cmd == CMD_END) return;

        if (cmd == CMD_BEGIN) {
            seed_ = (uint32_t)readincr(fromMaster);
            const int n = readincr(fromMaster);
            es_.n = n;
            es_.nPad = (n + 7) & ~7;
            for (int i = 0; i < n; ++i) es_.px[i] = pso_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.py[i] = pso_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.boosted[i] = pso_float(readincr(fromMaster));
            for (int i = 0; i < n; ++i) es_.bright[i] = pso_float(readincr(fromMaster));
            for (int i = n; i < es_.nPad; ++i) {
                es_.px[i] = 1.0e4f; es_.py[i] = 1.0e4f; es_.boosted[i] = 0.0f; es_.bright[i] = 0.0f;
            }
        } else if (cmd == CMD_INIT) {
            const int P = readincr(fromMaster);
            for (int i = 0; i < P; ++i)                    // keep the own particles
                for (int d = 0; d < 5; ++d) {
                    const float v = pso_float(readincr(fromMaster));
                    if (i % W == id_) shard_.pos[i][d] = v;
                }
            writeHeader(toMaster, 0, getPacketid(toMaster, 0));
            writeincr(toMaster, (int32)id_);
            int cnt = 0;
            for (int i = id_; i < P; i += W) ++cnt;
            writeincr(toMaster, (int32)cnt, cnt == 0);
            int j = 0;
            for (int i = id_; i < P; i += W, ++j)
                writeincr(toMaster, pso_bits(pso_particle_init(shard_, es_, i)), j == cnt - 1);
        } else if (cmd == CMD_STEP) {
            const int it = readincr(fromMaster);
            const int P = readincr(fromMaster);
            float gbest[5];
            for (int d = 0; d < 5; ++d) gbest[d] = pso_float(readincr(fromMaster));
            PsoBest b;
            pso_shard_step(shard_, es_, PSO_VARIANT / 2, seed_, it, id_, W, P, gbest, b);
            writeHeader(toMaster, 0, getPacketid(toMaster, 0));
            writeincr(toMaster, (int32)id_);
            writeincr(toMaster, (int32)b.idx);
            writeincr(toMaster, pso_bits(b.score));
            for (int d = 0; d < 5; ++d) writeincr(toMaster, pso_bits(b.pos[d]), d == 4);
        } else if (cmd == CMD_STEP_SP) {
            const int it = readincr(fromMaster);
            for (int i = 0; i < SP_PARTICLES; ++i) {
                tabS_[i] = pso_float(readincr(fromMaster));
                for (int d = 0; d < 5; ++d) tabP_[i][d] = pso_float(readincr(fromMaster));
                reset_[i] = (uint8_t)readincr(fromMaster);
            }
            pso_shard_step_sp(shard_, es_, seed_, it, id_, W, tabP_, tabS_, reset_, newP_, newS_);
            writeHeader(toMaster, 0, getPacketid(toMaster, 0));
            writeincr(toMaster, (int32)id_);
            int cnt = 0;
            for (int i = id_; i < SP_PARTICLES; i += W) ++cnt;
            writeincr(toMaster, (int32)cnt, cnt == 0);
            int j = 0;
            for (int i = id_; i < SP_PARTICLES; i += W, ++j) {
                writeincr(toMaster, pso_bits(newS_[i]));
                for (int d = 0; d < 5; ++d)
                    writeincr(toMaster, pso_bits(newP_[i][d]), j == cnt - 1 && d == 4);
            }
        }
    }
}
