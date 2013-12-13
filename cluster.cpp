#include "pCBIR.h"

const int numDoc = 100;
const int numFea = 3600;
const int filename = "feature_input";
const int TARGET = 0;
const int DATA = 1;
const int DIFF = 2;
double distance(double* a, double* b) {
  double diff = 0;
  for(int i = 0; i < numFea; i++) {
    diff += pow((a[i] - b[i]), 2.0);
  }
  return sqrt(diff);
}

int main(int argc, char** argv) {
  int np, rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double diff[numDoc];
  int div_work = numDoc / np;
  int mas_work = div_work + numDoc % np;
  featureT target;
  double* featureArray;
  
  double* target;
  if ( rank == 0 ) 
  {
    featureArray = readFeatures(filename);
    //for test
    target = featruenArray;
    for (int i = 1; i < np; i ++) {
      MPI_Isend(target, numFea, MPI_DOUBLE, i, TARGET, MPI_COMM_WORLD, 0);
      MPI_Isend(featureArray + (mas_work + (i - 1) * div_work) * numFea, div_work * numFea, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD, 0);
      MPI_Irecv(diff + mas_work + (i - 1)*div_work, div_work, MPI_DOUBLW, i, DIFF, MPI_COMM_WORLD, 0);
    }
    for(int i = 0; i < mas_work; i++) {
      diff[i] = distance(target, featureArray + i*numFea);
    }
  } else {
    featureArray
    
    
