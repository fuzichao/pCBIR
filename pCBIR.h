#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
/*
struct featureT {
  int id;
  int featureN;
  double* features;
}*/


double* readFeatures(const char* filename);
