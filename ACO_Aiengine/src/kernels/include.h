#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// ============================================================
// include.h  —  sizes and algorithm constants shared by the
// graph (project.h), the AIE kernels, the native bench
// (bench/aco_bench.cpp) and the Python scripts
// (make_input.py / evaluate.py read the #defines from this file).
//
// Values are the native (grid-selected) configuration of the
// reference C++ code: ACO/cppCode/scriptCode/Config.h.
// ============================================================

// ---- which ACO algorithm this build runs ----
// One build = one algorithm. The two algorithms are built and run one
// after the other (scripts/run_x86sim.sh) only to compare accuracy and
// time.
// Select with -DACO_VARIANT=<n> (CMake: -DACO_VARIANT=<n>).
#define VI_FPED     0     // Factorized-Pheromone Ellipse Detection
#define VI_STDACO   1     // joint-pheromone Standard ACO baseline
#define ACO_VARIANTS 2
#ifndef ACO_VARIANT
#define ACO_VARIANT VI_FPED
#endif

// ---- images and parallel workers ----
// The images are detected one after another: one graph iteration = one
// image. The speed-up is inside the detection: in every colony iteration
// the master tile broadcasts the ants to ACO_WORKERS worker tiles, which
// score them at the same time (one ant per worker with 30 workers).
#ifndef NUM_IMAGES
#define NUM_IMAGES 1000
#endif
#define NUM_ITERATIONS  NUM_IMAGES
// Worker tiles that score the ants of an iteration in parallel.
// At most 32 (5-bit packet id of the merged return stream).
#ifndef ACO_WORKERS
#define ACO_WORKERS 30
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
//   byte 2    1 = real image, 0 = padding (last batch not full)
//   byte 3-15 reserved (0)
#define IMG_HEADER_BYTES 16
#define IMG_RECORD_BYTES (IMG_HEADER_BYTES + IMG_PIXELS)            // 416
// Hit pixels kept per image. The 1000 frames have 44..97 lit pixels;
// a frame with more is truncated and flagged (RES_FLAGS bit 0).
#define MAX_EDGES       128

// ---- result record (one per image) ----
// 13 scalar words, then the convergence history: the best-so-far
// fitness after every colony iteration (ACO_HIST_LEN floats; the
// invalid-fitness marker -1e30 until the first valid candidate). Integer fields are int32, the
// others are float bit patterns.
#define ACO_HIST_LEN 100   // = FPED_ITERS = STD_ITERS
#define RESULT_WORDS (13 + ACO_HIST_LEN)
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
#define RES_ITERS    10   // int   colony iterations run
#define RES_FLAGS    11   // int   bit 0: hit list truncated to MAX_EDGES
#define RES_TIME_US  12   // float detection time of this image in microseconds:
                          //       hw = master tile cycle counter / AIE_CLOCK_MHZ,
                          //       x86sim = host wall clock (functional model only)
#define RES_HIST     13   // float[ACO_HIST_LEN] best-so-far fitness per colony iteration

// ---- fitness / detection constants (Config.h) ----
#define INLIER_TOLERANCE          0.40f   // tau: rim band half-width in E
#define MIN_INLIERS_FOR_DETECTION 10
#define MAX_INLIERS_FOR_DETECTION 70
#define HOLLOWNESS_PENALTY        2.0f    // lambda
#define HIGH_INTENSITY_THRESHOLD  0.5f
#define BRIGHT_POINT_BONUS        1.5f    // gamma
// INTENSITY_BOOST_POWER is 2, so the boosted intensity is intensity^2.

// ---- search space (Detectors.h buildSearchSpace) ----
#define ACO_BINS          60      // grid bins per parameter dimension
#define CENTER_HALFSPAN   7.0f    // px around the moment seed
#define AXIS_HALFSPAN     4.0f    // px around the moment seed
#define ETA_FLOOR         0.15f
#define ETA_SIGMA_C       2.0f
#define ETA_SIGMA_AX      1.5f
#define ETA_SIGMA_TH      0.5f
#define NEAR_CIRCULAR     1.15f   // a0/b0 below this -> flat theta heuristic

// ---- FPED (FpedParams) ----
#define FPED_ANTS         30
#define FPED_ITERS        100
#define FPED_ALPHA        1.5f
#define FPED_BETA         3.0f
#define FPED_RHO          0.30f
#define FPED_ELITE        10
#define FPED_Q            1.0f
#define FPED_QBEST        0.5f
#define FPED_SMOOTH_SIGMA 1.5f
#define FPED_SMOOTH_RAD   6       // max(1, ceil(4 * sigma))
#define FPED_TAU_MIN      0.05f
#define FPED_TAU_MAX      25.0f
#define FPED_REFINE_AT    50      // progressive trail refinement (0 = off)
#define FPED_REFINE_C     2.0f    // px half-span of the refined centre grids
#define FPED_REFINE_AX    1.5f    // px half-span of the refined axis grids
#define FPED_REFINE_TH    0.30f   // rad half-span of the refined theta grid

// ---- Standard ACO baseline (StdAcoParams) ----
#define STD_ANTS          30
#define STD_ITERS         100
#define STD_TUPLES        150     // fixed joint candidate archive
#define STD_ALPHA         1.5f
#define STD_RHO           0.30f
#define STD_ELITE         10
#define STD_Q             1.0f
#define STD_QBEST         0.5f
#define STD_TAU_MIN       0.05f
#define STD_TAU_MAX       25.0f

#endif
