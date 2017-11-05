#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>

#include "io_flow.h"
#include "utils.h"

using namespace std;

// COMPILE: g++ -O3 -DNDEBUG -o interp interpolate.cpp -lpng

int32_t main (int32_t argc,char *argv[]) {

  if (argc!=3) {
    cout << "Usage: ./interp in_file out_file" << endl;
    return 1;
  }

  FlowImage flow(argv[1]);
  flow.interpolateBackground();
  flow.write(argv[2]);

  return 0;
}

