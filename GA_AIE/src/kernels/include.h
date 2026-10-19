#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// ============================================================
// include.h  —  sizes and algorithm constants shared by the
// graph (project.h), every AIE kernel and the Python scripts
// (make_input.py / evaluate.py read the #defines from this file).
//
// Values are the native (tuned) configuration of the reference
// C++ code: GaCppPython/cppCode/scriptCode/Config.h and
// cppCode/results/experiments/tuned_params.json.
// ============================================================

// ---- images and parallel workers ----
// One build runs ONE GA variant (GA_VARIANT, below). The images are
// detected one after another: one graph iteration = one image. The
// speed-up is inside the detection: every generation is spread over
// GA_WORKERS worker tiles that build and score their share of the
// children (Conventional, Micro) or evolve one island each (Island)
// at the same time; the master tile keeps the population bookkeeping.
// The variants are built and run one after another (scripts/run_x86sim.sh)
// and only compared for accuracy and time.
#ifndef NUM_IMAGES
#define NUM_IMAGES 1000
#endif
#define NUM_ITERATIONS  NUM_IMAGES
// Worker tiles per detection (at most 32: 5-bit packet id of the merged
// return stream; at least ISL_ISLANDS so every island has its own tile).
#ifndef GA_WORKERS
#define GA_WORKERS 30
#endif
// AI Engine clock used to turn the tile cycle counter into time
// (VC1902, -2MP / -3HP speed grades).
#define AIE_CLOCK_MHZ 1250

// ---- image / buffer geometry ----
#define IMG_W           20
#define IMG_H           20
#define IMG_PIXELS      (IMG_W * IMG_H)        // 400 grayscale bytes per image
// Image record in the input stream: 16-byte header + 400 pixel bytes.
//   byte 0-1  image id (little endian)
//   byte 2    1 = real image
//   byte 3-15 reserved (0)
#define IMG_HEADER_BYTES 16
#define IMG_RECORD_BYTES (IMG_HEADER_BYTES + IMG_PIXELS)            // 416
// Edge points kept per image. The 1000 frames have 44..97 lit pixels;
// a frame with more is truncated and flagged (RES_FLAGS bit 0).
#define MAX_EDGES       128

// ---- result record (one per image) ----
// 13 scalar words, then the convergence history: the best-so-far fitness
// after every generation (GA_HIST_LEN floats, padded with the last value).
// Integer fields are int32, the others are float bit patterns.
//   Conventional : one value per generation           (CONV_GENS = 74)
//   Micro        : one value per step until the evaluation budget ends
//                  (at most MICRO_MAX_EVALS / 13 + 1 steps)
//   Island       : one value per island-model generation, best over the
//                  8 islands (ISL_RESTARTS x ISL_EPOCHS x ISL_MIG_INTERVAL)
#define CONV_HIST_LEN   74
#define MICRO_HIST_LEN  390
#define ISL_HIST_LEN    240
#define GA_HIST_MAX     MICRO_HIST_LEN
#define RESULT_SCALARS  13
#define RES_VARIANT  0    // int   variant index (vi)
#define RES_IMAGE    1    // int   image id
#define RES_DETECTED 2    // int   1 = valid detection
#define RES_CX       3    // float best ellipse, image space (20x20)
#define RES_CY       4
#define RES_A        5
#define RES_B        6
#define RES_THETA    7
#define RES_FITNESS  8    // float final best fitness
#define RES_EVALS    9    // int   fitness evaluations spent
#define RES_GENS     10   // int   generations run
#define RES_FLAGS    11   // int   bit 0: edge list truncated to MAX_EDGES
#define RES_TIME_US  12   // float detection time of this image in microseconds:
                          //       hw = master tile cycle counter / AIE_CLOCK_MHZ,
                          //       x86sim = host wall clock (functional model only)
#define RES_HIST     13   // float[GA_HIST_LEN] best-so-far fitness per generation

// ---- fitness / detection constants (Config.h) ----
#define INVALID_FITNESS           (-1.0e6f)
#define INLIER_TOLERANCE          0.4f
#define MIN_INLIERS_FOR_DETECTION 10
#define MAX_INLIERS_FOR_DETECTION 70
#define HOLLOWNESS_PENALTY        2.0f
#define HIGH_INTENSITY_THRESHOLD  0.5f
#define BRIGHT_POINT_BONUS        0.0f
// INTENSITY_BOOST_POWER is 2, so the boosted intensity is intensity^2.

// ---- shared mutation constants (GaCommon) ----
#define MUT_FINAL_RATE  0.05f
#define MUT_CX_FRAC     0.07f
#define MUT_CY_FRAC     0.07f
#define MUT_AXIS_SCALE  2.50f
#define MUT_THETA_SCALE 0.22f

// ---- Conventional GA (ConvGaParams) ----
#define CONV_POP        800
#define CONV_GENS       74
#define CONV_ELITISM    20
#define CONV_TOURNAMENT 8
#define CONV_XRATE      0.7f
#define CONV_BLX        0.3f
#define CONV_MUT0       0.6f

// ---- Micro GA (MicroGaParams) ----
#define MICRO_POP        14
#define MICRO_MAX_EVALS  5000
#define MICRO_TOURNAMENT 2
#define MICRO_CONV_TOL   0.001f
#define MICRO_XRATE      1.0f
#define MICRO_BLX        0.7f
#define MICRO_MUT0       0.6f

// ---- Island Model GA (IslandGaParams) ----
#define ISL_GENERATIONS   350     // only scales the mutation schedule
#define ISL_ELITISM       10
#define ISL_TOURNAMENT    5
#define ISL_STAG_WINDOW   25
#define ISL_STAG_TOL      0.001f
#define ISL_DIVERSITY_FRAC 0.05f
#define ISL_MAX_STAG      3
#define ISL_ISLANDS       8
#define ISL_POP           30
#define ISL_MIG_INTERVAL  15
#define ISL_MIG_SIZE      6
#define ISL_EPOCHS        8
#define ISL_RESTARTS      2
#define ISL_MIGRATION     1       // ablation switches, both on = IMGA as studied
#define ISL_REINJECTION   1
#define ISL_XRATE         0.7f
#define ISL_BLX           0.1f
#define ISL_MUT0          0.6f

// ---- variant index, same order as runAll.cpp (used in the RNG seed) ----
#define GA_VARIANTS 3
#define VI_CONV     0
#define VI_MICRO    1
#define VI_ISLAND   2

// ---- the variant built into the graph ----
// Set by the build: cmake -DGA_VARIANT=<0|1|2> (CMakeLists.txt passes
// --Xpreproc=-DGA_VARIANT=<n> to the AIE compiler). Without it the
// Conventional GA is built (e.g. a plain Vitis IDE build).
#ifndef GA_VARIANT
#define GA_VARIANT VI_CONV
#endif

// history length and record size of the variant built in
#if GA_VARIANT == VI_CONV
#define GA_HIST_LEN CONV_HIST_LEN
#elif GA_VARIANT == VI_MICRO
#define GA_HIST_LEN MICRO_HIST_LEN
#else
#define GA_HIST_LEN ISL_HIST_LEN
#endif
#define RESULT_WORDS (RESULT_SCALARS + GA_HIST_LEN)

#endif
