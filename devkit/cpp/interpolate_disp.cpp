#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>

#include "io_disp.h"
#include "utils.h"

using namespace std;

// COMPILE: g++ -O3 -DNDEBUG -o interp interpolate.cpp -lpng

int32_t main (int32_t argc,char *argv[]) {

  if (argc!=3) {
    cout << "Usage: ./interp in_file out_file" << endl;
    return 1;
  }

  DisparityImage disp(argv[1]);
  disp.interpolateBackground();
  disp.write(argv[2]);

  return 0;
}

