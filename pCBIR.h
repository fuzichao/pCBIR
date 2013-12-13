#include <mpi.h>
#include <iostream>
#include <cmath>

struct featureT {
  int id;
  int featureN;
  double* features;
}


featurT* readFeatures(char* filename);