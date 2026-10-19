#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// ============================================================
// include.h  —  sizes and algorithm constants shared by the
// graph (project.h), the AIE kernels, the native bench
// (bench/abc_bench.cpp), the reference C++ port and the Python
// scripts (make_input.py / evaluate.py read the #defines).
//
// Values are those of the Python study in ../ABC:
//   hollowness-aware-ABC.py   (HA-ABC)
//   standard-residual-ABC.py  (SR-ABC)
// ============================================================

// ---- which ABC algorithm this build runs ----
// One build = one algorithm; the two are built and run one after the
// other only to compare accuracy and time.
// Select with -DABC_VARIANT=<n> (CMake: -DABC_VARIANT=<n>).
#define VI_HA        0     // Hollowness-Aware ABC (intensity-weighted hollowness fitness)
#define VI_SR        1     // Standard-Residual ABC (Canny edges, residual fitness)
#define ABC_VARIANTS 2
#ifndef ABC_VARIANT
#define ABC_VARIANT VI_HA
#endif

// ---- images and parallel workers ----
// The images are detected one after another: one graph iteration = one
// image. The speed-up is inside the detection: in every ABC iteration
// (employed, onlooker and scout phases) the bees are worked on by
// ABC_WORKERS worker tiles at the same time (one bee per worker).
#ifndef NUM_IMAGES
#define NUM_IMAGES 1000
#endif
#define NUM_ITERATIONS  NUM_IMAGES
#ifndef ABC_WORKERS
#define ABC_WORKERS 30
#endif
#define AIE_CLOCK_MHZ 1250

// ---- image / record geometry ----
#define IMG_W           20
#define IMG_H           20
#define IMG_PIXELS      (IMG_W * IMG_H)
// Image record: 16-byte header (image id in bytes 0-1, valid flag in
// byte 2) + 400 grayscale bytes, row-major.
#define IMG_HEADER_BYTES 16
#define IMG_RECORD_BYTES (IMG_HEADER_BYTES + IMG_PIXELS)            // 416
// Input points per image: HA uses the lit pixels (44..97 in the dataset);
// SR uses the Canny edge pixels of the 4x upscaled frame. A frame with
// more is truncated and flagged (RES_FLAGS bit 0).
#define MAX_POINTS      800

// ---- result record (one per image) ----
// 13 scalar words, then the convergence history (best-so-far fitness
// after every ABC iteration, ABC_HIST_LEN floats).
#define ABC_HIST_LEN 150   // = MAX_ITERATIONS
#define RESULT_WORDS (13 + ABC_HIST_LEN)
#define RES_VARIANT  0    // int   variant index (vi)
#define RES_IMAGE    1    // int   image id
#define RES_DETECTED 2    // int   1 = a valid best bee was found
#define RES_CX       3    // float best ellipse, image pixels (20x20)
#define RES_CY       4
#define RES_A        5
#define RES_B        6
#define RES_THETA    7
#define RES_FITNESS  8    // float final best fitness
#define RES_EVALS    9    // int   fitness evaluations spent
#define RES_ITERS    10   // int   ABC iterations run
#define RES_FLAGS    11   // int   bit 0: points truncated to MAX_POINTS,
                          //       bit 1: SR fallback to the lit pixels (< 5 edges),
                          //       bit 2: too few points, fallback ellipse returned
#define RES_TIME_US  12   // float detection time in microseconds (hw: tile cycles /
                          //       AIE_CLOCK_MHZ; x86sim: host wall clock)
#define RES_HIST     13   // float[ABC_HIST_LEN] best-so-far fitness per iteration

// ---- ABC hyper-parameters (both scripts) ----
#define NUM_BEES        30
#define MAX_ITERATIONS  150
#define SCOUT_LIMIT     20

// ---- HA-ABC fitness ----
#define HA_INLIER_TOLERANCE 0.35f
#define HA_HOLLOWNESS       0.50f
#define HA_MIN_INLIERS      3
#define HA_MAX_INLIERS      500
#define HA_MIN_AXIS         0.5f     // a or b below -> invalid
#define HA_INIT_RESTARTS    50       // re-draws of an invalid initial bee
#define HA_BEE_AXIS_MIN     1.0f     // random_bee: a, b ~ U(1.0, size/2)

// ---- SR-ABC fitness ----
#define SR_EPS              1.0e-6f  // fitness = 1 / (mean |E - 1| + eps)
#define SR_MIN_POINTS       5        // fewer edge points -> fallback ellipse
#define SR_BEE_AXIS_MIN     1.5f     // random_bee: a, b ~ U(1.5, size/2)

#endif
