
#include <adf.h>
#include "kernels.h"
#include "project.h"
#include <iostream>

using namespace adf;

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(4);
  mygraph.end();
std::cout<<"Hello world";
  return 0;
}
