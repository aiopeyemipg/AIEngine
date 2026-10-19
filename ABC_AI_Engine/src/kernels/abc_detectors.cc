// ============================================================
// abc_detectors.cc  —  master and worker tiles of the ABC detector
//
// One graph iteration = one image. The master runs the colony
// (abc_core.h abc_detect) with an engine that sends every phase of an
// ABC iteration (initialisation, employed, onlooker, scout) to the worker
// tiles, which build and score their bees at the same time. The
// detections equal the native serial and threaded runs of the same code
// (bench/abc_bench.cpp), up to the last bit of aie::sqrt in x86sim.
// ============================================================
#include <adf.h>
#include "include.h"
#include "kernels.h"
#include "abc_core.h"
#if defined(__X86SIM__)
#include <chrono>
#endif

static inline int32 abc_bits(float f) {
    union { float f; int32 i; } u;
    u.f = f;
    return u.i;
}

static inline float abc_float(int32 i) {
    union { int32 i; float f; } u;
    u.i = i;
    return u.f;
}

// Time stamp in microseconds: the tile cycle counter on the AI Engine,
// the host clock in x86sim (a functional model, not the device speed).
static inline double abc_now_us() {
#if defined(__X86SIM__)
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    return (double)aie::tile::current().cycles() / (double)AIE_CLOCK_MHZ;
#endif
}

// ------------------------------------------------------------
// Master-side engine: every phase is one broadcast + one gather.
// ------------------------------------------------------------
struct AieEngine {
    input_pktstream*      in;
    output_stream<int32>* out;

    // new image: every worker gets the points and the seed
    void begin(const AbcPoints& p, uint32_t seed) {
        writeincr(out, (int32)CMD_BEGIN);
        writeincr(out, (int32)seed);
        writeincr(out, (int32)p.n);
        writeincr(out, (int32)p.nPad);
        for (int i = 0; i < p.nPad; ++i) writeincr(out, abc_bits(p.px[i]));
        for (int i = 0; i < p.nPad; ++i) writeincr(out, abc_bits(p.py[i]));
        for (int i = 0; i < p.nPad; ++i) writeincr(out, abc_bits(p.w[i]));
    }

    // one phase: bee task t on worker t % ABC_WORKERS, all at once
    void phase(const AbcPhase& m, int n, float (*cand)[5], float* fit, int* evals) {
        writeincr(out, (int32)CMD_PHASE);
        writeincr(out, (int32)m.kind);
        writeincr(out, (int32)m.iter);
        for (int i = 0; i < NUM_BEES; ++i)
            for (int d = 0; d < 5; ++d) writeincr(out, abc_bits(m.bees[i][d]));
        for (int i = 0; i < NUM_BEES; ++i) writeincr(out, (int32)m.sel[i]);
        (void)n;                                            // always NUM_BEES tasks
        for (int k = 0; k < ABC_WORKERS; ++k) {
            readincr(in);                                   // packet header
            const int w = readincr(in);
            const int cnt = readincr(in);
            for (int j = 0; j < cnt; ++j) {
                const int t = w + j * ABC_WORKERS;
                for (int d = 0; d < 5; ++d) cand[t][d] = abc_float(readincr(in));
                fit[t] = abc_float(readincr(in));
                evals[t] = readincr(in);
            }
        }
    }
};

// ============================================================
// AbcMaster
// ============================================================
AbcMaster::AbcMaster(uint8 (&blur)[CANNY_BUF], int16 (&mag)[CANNY_BUF], int8 (&dir)[CANNY_BUF])
    : blur_(blur), mag_(mag), dir_(dir) {}

void AbcMaster::run(adf::input_buffer<uint8>& image, input_pktstream* fromWorkers,
                    output_stream<int32>* toWorkers, adf::output_buffer<int32>& result) {
    const double t0 = abc_now_us();

    AieEngine eng = { fromWorkers, toWorkers };
    AbcResult res;
    abc_detect_image(image.data(), ABC_VARIANT, pts_, (uint8_t*)blur_, (int16_t*)mag_, (int8_t*)dir_,
                     st_, eng, res);
    writeincr(toWorkers, (int32)CMD_END);                 // workers finish the image

    const float us = (float)(abc_now_us() - t0);
    int32* r = result.data();
    r[RES_VARIANT]  = ABC_VARIANT;
    r[RES_IMAGE]    = res.image;
    r[RES_DETECTED] = res.detected;
    for (int d = 0; d < 5; ++d) r[RES_CX + d] = abc_bits(res.ell[d]);
    r[RES_FITNESS]  = abc_bits(res.fitness);
    r[RES_EVALS]    = res.evals;
    r[RES_ITERS]    = res.iters;
    r[RES_FLAGS]    = res.flags;
    r[RES_TIME_US]  = abc_bits(us);
    for (int i = 0; i < ABC_HIST_LEN; ++i) r[RES_HIST + i] = abc_bits(res.hist[i]);
}

// ============================================================
// AbcWorker
// ============================================================
AbcWorker::AbcWorker(int id) : id_(id) {}

void AbcWorker::run(input_stream<int32>* fromMaster, output_pktstream* toMaster) {
    uint32_t seed = 0;
    for (;;) {
        const int cmd = readincr(fromMaster);
        if (cmd == CMD_END) return;

        if (cmd == CMD_BEGIN) {
            seed = (uint32_t)readincr(fromMaster);
            pts_.n = readincr(fromMaster);
            pts_.nPad = readincr(fromMaster);
            for (int i = 0; i < pts_.nPad; ++i) pts_.px[i] = abc_float(readincr(fromMaster));
            for (int i = 0; i < pts_.nPad; ++i) pts_.py[i] = abc_float(readincr(fromMaster));
            for (int i = 0; i < pts_.nPad; ++i) pts_.w[i] = abc_float(readincr(fromMaster));
        } else if (cmd == CMD_PHASE) {
            msg_.kind = readincr(fromMaster);
            msg_.iter = readincr(fromMaster);
            for (int i = 0; i < NUM_BEES; ++i)
                for (int d = 0; d < 5; ++d) msg_.bees[i][d] = abc_float(readincr(fromMaster));
            for (int i = 0; i < NUM_BEES; ++i) msg_.sel[i] = readincr(fromMaster);

            const int cnt = (NUM_BEES - id_ + ABC_WORKERS - 1) / ABC_WORKERS;   // tasks id, id + W, ...
            writeHeader(toMaster, 0, getPacketid(toMaster, 0));
            writeincr(toMaster, (int32)id_);
            writeincr(toMaster, (int32)cnt, cnt == 0);
            for (int j = 0; j < cnt; ++j) {
                const int t = id_ + j * ABC_WORKERS;
                float cand[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, fit = 0.0f;
                int evals = 0;
                abc_task(pts_, ABC_VARIANT, seed, msg_, t, cand, fit, evals);
                for (int d = 0; d < 5; ++d) writeincr(toMaster, abc_bits(cand[d]));
                writeincr(toMaster, abc_bits(fit));
                writeincr(toMaster, (int32)evals, j == cnt - 1);
            }
        }
    }
}
