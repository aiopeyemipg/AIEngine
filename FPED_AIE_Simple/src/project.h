#ifndef PROJECT_H
#define PROJECT_H

#include <adf.h>
#include "kernels.h"

using namespace adf;

class simpleGraph : public adf::graph
{
private:

    kernel aco1;
    kernel aco2;
    kernel aco3;
    kernel aco4;

public:

    input_plio in;
    output_plio out;

    simpleGraph()
    {
        in = input_plio::create(
            plio_32_bits,
            "/home/embeddedai002/Desktop/AI_engine/FPED_AIE_Simple/data/Ellipses_1000_images.csv"
        );

        out = output_plio::create(
            plio_32_bits,
            "data/output.csv"
        );

        // Four independent ACO kernel instances
        aco1 = kernel::create(factorized_aco);
        aco2 = kernel::create(factorized_aco);
        aco3 = kernel::create(factorized_aco);
        aco4 = kernel::create(factorized_aco);

        connect(in.out[0], aco1.in[0]);

        connect(aco1.out[0], aco2.in[0]);

        connect(aco2.out[0], aco3.in[0]);

        connect(aco3.out[0], aco4.in[0]);

        connect(aco4.out[0], out.in[0]);

        dimensions(aco1.in[0]) =
            {5};

        dimensions(aco1.out[0]) =
            {3};

        dimensions(aco2.in[0]) =
            {3};

        dimensions(aco2.out[0]) =
            {3};

        dimensions(aco3.in[0]) =
            {3};

        dimensions(aco3.out[0]) =
            {3};

        dimensions(aco4.in[0]) =
            {3};

        dimensions(aco4.out[0]) =
            {3};

        source(aco1) =
            "kernels/kernels.cc";

        source(aco2) =
            "kernels/kernels.cc";

        source(aco3) =
            "kernels/kernels.cc";

        source(aco4) =
            "kernels/kernels.cc";

        runtime<ratio>(aco1) = 0.8;
        runtime<ratio>(aco2) = 0.8;
        runtime<ratio>(aco3) = 0.8;
        runtime<ratio>(aco4) = 0.8;
    }
};

#endif