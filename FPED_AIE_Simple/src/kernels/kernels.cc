#include <adf.h>
#include "kernels.h"

#include <cmath>
#include <cstdint>

using namespace adf;

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

static constexpr float ALPHA = 1.5f;
static constexpr float BETA  = 1.0f;
static constexpr float RHO   = 0.15f;

static constexpr float GAUSSIAN_SIGMA = 1.5f;

static constexpr float PI = 3.14159265358979323846f;


// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------

static float absf(float x)
{
    return (x < 0.0f) ? -x : x;
}


static float clampf(float x, float lo, float hi)
{
    if (x < lo)
        return lo;

    if (x > hi)
        return hi;

    return x;
}


// ------------------------------------------------------------
// Angle wrapping
// Equivalent to Python:
//
// diff = abs(a1-a2) % pi
// return min(diff, pi-diff)
// ------------------------------------------------------------

static float angle_wrap_error(float a1, float a2)
{
    float diff = absf(a1 - a2);

    while (diff >= PI)
        diff -= PI;

    float other = PI - diff;

    return (diff < other) ? diff : other;
}


// ------------------------------------------------------------
// Gaussian smoothing
//
// Approximation of scipy.ndimage.gaussian_filter1d()
//
// Fixed 3-point Gaussian smoothing is used because the AIE
// kernel should avoid dynamic convolution operations.
// ------------------------------------------------------------

static void gaussian_smooth(
    float* pheromone,
    int length)
{
    float temp[N_BINS];

    const float sigma = GAUSSIAN_SIGMA;

    const float w0 =
        1.0f / (std::sqrt(2.0f * PI) * sigma);

    const float w1 =
        std::exp(-1.0f / (2.0f * sigma * sigma));

    const float w2 =
        std::exp(-4.0f / (2.0f * sigma * sigma));

    float norm = w0 + 2.0f * w1 + 2.0f * w2;

    float c0 = w0 / norm;
    float c1 = w1 / norm;
    float c2 = w2 / norm;

    for (int i = 0; i < length; i++)
    {
        float value = c0 * pheromone[i];

        int i1 = i - 1;
        int i2 = i + 1;

        int i3 = i - 2;
        int i4 = i + 2;

        if (i1 < 0)
            i1 = 0;

        if (i2 >= length)
            i2 = length - 1;

        if (i3 < 0)
            i3 = 0;

        if (i4 >= length)
            i4 = length - 1;

        value += c1 * pheromone[i1];
        value += c1 * pheromone[i2];

        value += c2 * pheromone[i3];
        value += c2 * pheromone[i4];

        temp[i] = value;
    }

    for (int i = 0; i < length; i++)
    {
        pheromone[i] = temp[i];
    }
}


// ------------------------------------------------------------
// Ellipse IoU approximation
//
// The original Python code rasterizes two ellipses using
// cv2.ellipse() and calculates:
//
// intersection / union
//
// For AIE we replace the OpenCV rasterization with a fixed
// grid approximation.
// ------------------------------------------------------------

static float ellipse_iou(
    const Ellipse& e1,
    const Ellipse& e2)
{
    constexpr int GRID = 20;

    int intersection = 0;
    int union_count  = 0;

    for (int y = 0; y < GRID; y++)
    {
        for (int x = 0; x < GRID; x++)
        {
            float px =
                ((float)x + 0.5f) * 40.0f / GRID;

            float py =
                ((float)y + 0.5f) * 40.0f / GRID;

            // -----------------------------
            // Ellipse 1
            // -----------------------------

            float dx1 = px - e1.cx;
            float dy1 = py - e1.cy;

            float c1 = std::cos(e1.theta);
            float s1 = std::sin(e1.theta);

            float xr1 =  c1 * dx1 + s1 * dy1;
            float yr1 = -s1 * dx1 + c1 * dy1;

            float value1 =
                (xr1 * xr1) / (e1.a * e1.a) +
                (yr1 * yr1) / (e1.b * e1.b);

            bool inside1 = value1 <= 1.0f;

            // -----------------------------
            // Ellipse 2
            // -----------------------------

            float dx2 = px - e2.cx;
            float dy2 = py - e2.cy;

            float c2 = std::cos(e2.theta);
            float s2 = std::sin(e2.theta);

            float xr2 =  c2 * dx2 + s2 * dy2;
            float yr2 = -s2 * dx2 + c2 * dy2;

            float value2 =
                (xr2 * xr2) / (e2.a * e2.a) +
                (yr2 * yr2) / (e2.b * e2.b);

            bool inside2 = value2 <= 1.0f;

            if (inside1 && inside2)
                intersection++;

            if (inside1 || inside2)
                union_count++;
        }
    }

    if (union_count == 0)
        return 0.0f;

    return (float)intersection /
           (float)union_count;
}


// ------------------------------------------------------------
// Fitness
//
// Original Python:
//
// fitness = 1.0 - (1.0 - iou)^2
// ------------------------------------------------------------

static float ellipse_fitness(
    const Ellipse& candidate,
    const Ellipse& ground_truth)
{
    float iou =
        ellipse_iou(candidate, ground_truth);

    return 1.0f -
           (1.0f - iou) *
           (1.0f - iou);
}


// ------------------------------------------------------------
// Deterministic pseudo random generator
//
// Avoid rand() inside an AIE kernel.
// ------------------------------------------------------------

static uint32_t rng_state = 123456789u;


static uint32_t fast_rand()
{
    rng_state =
        rng_state * 1664525u + 1013904223u;

    return rng_state;
}


static float random_uniform()
{
    uint32_t r = fast_rand();

    return ((float)(r & 0x00FFFFFFu)) /
           16777216.0f;
}


// ------------------------------------------------------------
// Random choice according to pheromone probability
//
// Equivalent to:
//
// np.random.choice(n_bins, p=probs[d])
// ------------------------------------------------------------

static int pheromone_choice(
    float* pheromone)
{
    float sum = 0.0f;

    float probability[N_BINS];

    for (int i = 0; i < N_BINS; i++)
    {
        float p =
            std::pow(pheromone[i], ALPHA);

        probability[i] = p;

        sum += p;
    }

    if (sum <= 0.0f)
        return 0;

    float r =
        random_uniform() * sum;

    float cumulative = 0.0f;

    for (int i = 0; i < N_BINS; i++)
    {
        cumulative += probability[i];

        if (r <= cumulative)
            return i;
    }

    return N_BINS - 1;
}


// ------------------------------------------------------------
// Build parameter grids
//
// Corresponds to the Python:
//
// self.grids = [
//     cx grid,
//     cy grid,
//     a grid,
//     b grid,
//     theta grid
// ]
// ------------------------------------------------------------

static void create_grids(
    const Ellipse& gt,
    float grids[N_PARAMS][N_BINS])
{
    float init_cx =
        gt.cx + 3.0f *
        (random_uniform() - 0.5f);

    float init_cy =
        gt.cy + 3.0f *
        (random_uniform() - 0.5f);

    float init_a =
        gt.a + 3.0f *
        (random_uniform() - 0.5f);

    float init_b =
        gt.b + 3.0f *
        (random_uniform() - 0.5f);

    float init_theta =
        gt.theta +
        0.4f *
        (random_uniform() - 0.5f);

    // -----------------------------
    // cx
    // -----------------------------

    float cx_start =
        clampf(init_cx - 6.0f, 0.5f, 39.5f);

    float cx_end =
        clampf(init_cx + 6.0f, 0.5f, 39.5f);

    // -----------------------------
    // cy
    // -----------------------------

    float cy_start =
        clampf(init_cy - 6.0f, 0.5f, 39.5f);

    float cy_end =
        clampf(init_cy + 6.0f, 0.5f, 39.5f);

    // -----------------------------
    // a
    // -----------------------------

    float a_start =
        clampf(init_a - 5.0f, 1.0f, 100.0f);

    float a_end =
        init_a + 5.0f;

    // -----------------------------
    // b
    // -----------------------------

    float b_start =
        clampf(init_b - 5.0f, 1.0f, 100.0f);

    float b_end =
        init_b + 5.0f;

    // -----------------------------
    // theta
    // -----------------------------

    float theta_start =
        clampf(
            init_theta - PI / 4.0f,
            0.0f,
            PI);

    float theta_end =
        clampf(
            init_theta + PI / 4.0f,
            0.0f,
            PI);

    // -----------------------------
    // Fill grids
    // -----------------------------

    for (int i = 0; i < N_BINS; i++)
    {
        float t =
            (float)i /
            (float)(N_BINS - 1);

        grids[0][i] =
            cx_start +
            t * (cx_end - cx_start);

        grids[1][i] =
            cy_start +
            t * (cy_end - cy_start);

        grids[2][i] =
            a_start +
            t * (a_end - a_start);

        grids[3][i] =
            b_start +
            t * (b_end - b_start);

        grids[4][i] =
            theta_start +
            t * (theta_end - theta_start);
    }
}


// ------------------------------------------------------------
// Main Factorized ACO
// ------------------------------------------------------------

static Ellipse run_factorized_aco(
    const Ellipse& ground_truth,
    float& best_fitness)
{
    float grids[N_PARAMS][N_BINS];

    float pheromones[N_PARAMS][N_BINS];

    // ---------------------------------------
    // Create search grids
    // ---------------------------------------

    create_grids(
        ground_truth,
        grids);

    // ---------------------------------------
    // Initialize pheromones
    //
    // Python:
    //
    // np.ones((5, n_bins))
    // ---------------------------------------

    for (int d = 0; d < N_PARAMS; d++)
    {
        for (int i = 0; i < N_BINS; i++)
        {
            pheromones[d][i] = 1.0f;
        }
    }

    Ellipse global_best;

    global_best.cx = ground_truth.cx;
    global_best.cy = ground_truth.cy;
    global_best.a  = ground_truth.a;
    global_best.b  = ground_truth.b;
    global_best.theta = ground_truth.theta;

    best_fitness = -1.0f;

    // ---------------------------------------
    // ACO iterations
    // ---------------------------------------

    for (int iteration = 0;
         iteration < N_ITERS;
         iteration++)
    {
        float iteration_best_fitness =
            -1.0f;

        Ellipse iteration_best;

        int selected_indices
            [N_ANTS][N_PARAMS];

        float ant_fitness[N_ANTS];

        // -----------------------------------
        // Generate ants
        // -----------------------------------

        for (int ant = 0;
             ant < N_ANTS;
             ant++)
        {
            // Select five parameters
            for (int d = 0;
                 d < N_PARAMS;
                 d++)
            {
                selected_indices[ant][d] =
                    pheromone_choice(
                        pheromones[d]);
            }

            // --------------------------------
            // Construct ellipse
            // --------------------------------

            Ellipse candidate;

            candidate.cx =
                grids[0][selected_indices[ant][0]];

            candidate.cy =
                grids[1][selected_indices[ant][1]];

            candidate.a =
                grids[2][selected_indices[ant][2]];

            candidate.b =
                grids[3][selected_indices[ant][3]];

            candidate.theta =
                grids[4][selected_indices[ant][4]];

            // Ensure a >= b
            if (candidate.a <
                candidate.b)
            {
                float temp =
                    candidate.a;

                candidate.a =
                    candidate.b;

                candidate.b =
                    temp;
            }

            // --------------------------------
            // Fitness
            // --------------------------------

            float fitness =
                ellipse_fitness(
                    candidate,
                    ground_truth);

            ant_fitness[ant] =
                fitness;

            // --------------------------------
            // Best ant
            // --------------------------------

            if (fitness >
                iteration_best_fitness)
            {
                iteration_best_fitness =
                    fitness;

                iteration_best =
                    candidate;
            }
        }

        // -----------------------------------
        // Pheromone evaporation
        //
        // Python:
        //
        // pheromones *= (1-RHO)
        // -----------------------------------

        for (int d = 0;
             d < N_PARAMS;
             d++)
        {
            for (int i = 0;
                 i < N_BINS;
                 i++)
            {
                pheromones[d][i] *=
                    (1.0f - RHO);
            }
        }

        // -----------------------------------
        // Pheromone deposition
        //
        // Python:
        //
        // pheromone[d, idx] += fitness * 0.5
        // -----------------------------------

        for (int ant = 0;
             ant < N_ANTS;
             ant++)
        {
            for (int d = 0;
                 d < N_PARAMS;
                 d++)
            {
                int idx =
                    selected_indices[ant][d];

                pheromones[d][idx] +=
                    ant_fitness[ant] * 0.5f;
            }
        }

        // -----------------------------------
        // Gaussian smoothing
        // -----------------------------------

        for (int d = 0;
             d < N_PARAMS;
             d++)
        {
            gaussian_smooth(
                pheromones[d],
                N_BINS);
        }

        // -----------------------------------
        // Global best
        // -----------------------------------

        if (iteration_best_fitness >
            best_fitness)
        {
            best_fitness =
                iteration_best_fitness;

            global_best =
                iteration_best;
        }
    }

    return global_best;
}


// ============================================================
// AIE KERNEL
// ============================================================

void factorized_aco(
    adf::input_buffer<cint16>& in,
    adf::output_buffer<cint16>& out)
{
    auto* input =
        in.data();

    auto* output =
        out.data();

    // --------------------------------------------------------
    // Input format
    //
    // We encode the ground-truth ellipse using five cint16
    // values:
    //
    // input[0].real = cx * 100
    // input[1].real = cy * 100
    // input[2].real = a  * 100
    // input[3].real = b  * 100
    // input[4].real = theta * 1000
    //
    // --------------------------------------------------------

    Ellipse gt;

    gt.cx =
        (float)input[0].real /
        100.0f;

    gt.cy =
        (float)input[1].real /
        100.0f;

    gt.a =
        (float)input[2].real /
        100.0f;

    gt.b =
        (float)input[3].real /
        100.0f;

    gt.theta =
        (float)input[4].real /
        1000.0f;

    // --------------------------------------------------------
    // Run ACO
    // --------------------------------------------------------

    float fitness;

    Ellipse result =
        run_factorized_aco(
            gt,
            fitness);

    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    output[0].real =
        (int16_t)(result.cx * 100.0f);

    output[0].imag =
        (int16_t)(result.cy * 100.0f);

    output[1].real =
        (int16_t)(result.a * 100.0f);

    output[1].imag =
        (int16_t)(result.b * 100.0f);

    output[2].real =
        (int16_t)(result.theta * 1000.0f);

    output[2].imag =
        (int16_t)(fitness * 1000.0f);
}