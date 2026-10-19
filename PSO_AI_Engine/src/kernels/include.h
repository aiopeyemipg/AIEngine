
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// ============================================================
// include.h  —  sizes and algorithm constants shared by the
// graph (project.h), every AIE kernel and the Python scripts
// (make_input.py / evaluate.py read the #defines from this file).
//
// Values are the v7 constants of the reference C++ code
// (cppCode/scriptCode/Config.h and helpers/runAll.cpp).
// ============================================================

// ---- images and parallel workers ----
// One graph runs ONE PSO variant (PSO_VARIANT, see the end of this file)
// and detects the images one after another: one graph iteration = one
// image. The speed-up is inside the detection: in every PSO iteration
// the master tile broadcasts the swarm leader (gbest, or the ring table
// of SP-PSO) and PSO_WORKERS worker tiles move and score their share of
// the particles at the same time (particle i on worker i % PSO_WORKERS).
// The variants are built and run one after another; they are only
// compared for accuracy and time.
#ifndef NUM_IMAGES
#define NUM_IMAGES 1000
#endif
#define NUM_ITERATIONS  NUM_IMAGES
// Worker tiles per image. At most 32 (5-bit packet id of the merged
// return stream). 25 divides both swarm sizes (100 and 25).
#ifndef PSO_WORKERS
#define PSO_WORKERS 25
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
// RESULT_WORDS int32 words on the output PLIO: 13 scalars, then the
// convergence history (best-so-far fitness after every PSO iteration,
// PSO_HIST_LEN floats). Integer fields are int32, the
// others are float bit patterns. Coordinates are in image pixels
// (20x20), before the x2 scaling of the evaluation.
#define PSO_HIST_LEN 200  // the longest run (SP-PSO, 200 iterations)
#define RESULT_WORDS (13 + PSO_HIST_LEN)
#define RES_VARIANT  0    // int   variant index (vi)
#define RES_IMAGE    1    // int   image id
#define RES_DETECTED 2    // int   1 = valid detection
#define RES_CX       3    // float best ellipse
#define RES_CY       4
#define RES_A        5
#define RES_B        6
#define RES_THETA    7
#define RES_FITNESS  8    // float final best fitness
#define RES_EVALS    9    // int   fitness evaluations spent
#define RES_ITERS    10   // int   PSO iterations run
#define RES_FLAGS    11   // int   bit 0: edge list truncated to MAX_EDGES
#define RES_TIME_US  12   // float detection time of this image in microseconds:
                          //       hw = master tile cycle counter / AIE_CLOCK_MHZ,
                          //       x86sim = host wall clock (functional model only)
#define RES_HIST     13   // float x PSO_HIST_LEN: best-so-far (gbest) fitness after
                          //       every PSO iteration, padded with the last value
                          //       after RES_ITERS iterations; PSO_NEG_INF while
                          //       no valid candidate was found

// ---- fitness / detection constants ----
#define INLIER_TOLERANCE          0.40f
#define MIN_INLIERS_FOR_DETECTION 10
#define MAX_INLIERS_FOR_DETECTION 70
#define HOLLOWNESS_PENALTY        2.0f
#define HIGH_INTENSITY_THRESHOLD  0.5f
#define BRIGHT_POINT_BONUS        1.5f
// INTENSITY_BOOST_POWER is 2 in the reference code, so the boosted
// intensity is computed as intensity * intensity.

// ---- DBSCAN noise removal ----
#define DBSCAN_EPS2        4      // eps = 2 px, compared squared on integer pixels
#define DBSCAN_MIN_SAMPLES 5

// ---- 5-point seeding ----
#define SEED_ATTEMPTS_PER_PARTICLE 30

// ---- Standard PSO (runAll.cpp: PsoParams{100, 100, 0.5, 1.5, 2.0}) ----
#define STD_PARTICLES 100
#define STD_ITERS     100
#define STD_W         0.5f
#define STD_C1        1.5f
#define STD_C2        2.0f

// ---- Accelerated PSO (ApsoParams defaults) ----
#define APSO_PARTICLES 100
#define APSO_ITERS     100
#define APSO_ALPHA0    1.0f
#define APSO_ALPHA_MIN 0.001f
#define APSO_BETA      0.5f

// ---- Small-Population PSO (SpParams defaults) ----
#define SP_PARTICLES      25
#define SP_ITERS          200
#define SP_W              0.7f
#define SP_C1             1.5f
#define SP_C2             1.5f
#define SP_NEIGHBORHOOD   3
#define SP_REINIT_INTERVAL 20
#define SP_REINIT_COUNT   5

// Largest swarm (sizes the per-worker particle arrays).
#define PSO_MAX_PARTICLES 100

// ---- variant index, same order as runAll.cpp (used in the RNG seed) ----
// even vi = with DBSCAN, odd vi = without
#define PSO_VARIANTS  6
#define VI_STD_DB     0
#define VI_STD_NODB   1
#define VI_APSO_DB    2
#define VI_APSO_NODB  3
#define VI_SPPSO_DB   4
#define VI_SPPSO_NODB 5

#define ALGO_STD   0      // vi / 2
#define ALGO_APSO  1
#define ALGO_SPPSO 2

// Variant built into the graph. Set by the build:
//   cmake -DPSO_VARIANT=<vi>   (passes --aie.Xpreproc=-DPSO_VARIANT=<vi>)
// The default (0, Standard PSO with DBSCAN) is used by a plain IDE build.
#ifndef PSO_VARIANT
#define PSO_VARIANT VI_STD_DB
#endif
#if PSO_VARIANT < 0 || PSO_VARIANT >= PSO_VARIANTS
#error "PSO_VARIANT must be 0..5"
#endif

#endif
