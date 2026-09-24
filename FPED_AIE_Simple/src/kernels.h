#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>

#define N_BINS   100
#define N_ANTS   15
#define N_ITERS  50

// Five ellipse parameters:
// cx, cy, a, b, theta
#define N_PARAMS 5

struct Ellipse
{
    float cx;
    float cy;
    float a;
    float b;
    float theta;
};

void factorized_aco(
    adf::input_buffer<cint16>& in,
    adf::output_buffer<cint16>& out
);

#endif