#include "project.h"

simpleGraph fped_graph;

int main()
{
    fped_graph.init();

    fped_graph.run(4);

    fped_graph.end();

    return 0;
}