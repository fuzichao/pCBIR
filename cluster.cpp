#include "pCBIR.h"
#include <algorithm>
const int numDoc = 999;
const int numFea = 162;
char* filename = "feature_input";
const int TARGET = 0;
const int DATA = 1;
const int DIFF = 2;
double distance(double* a, double* b) {
  double diff = 0;
  for(int i = 0; i < numFea; i++) {
    if(a[i]+b[i] != 0)
      diff += pow((a[i] - b[i]), 2.0)/(a[i]+b[i]);
  }
  return diff;
}
struct doc {
  int id;
  double score;
};

bool comfunc(doc a, doc b) {
  return (a.score < b.score);
}
int main(int argc, char** argv) {
  int np, rank, count; 
  MPI_Init(&argc, &argv);
  MPI_File infile;
  MPI_Status status;
  MPI_Offset filesize;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
  MPI_File_get_size(infile, &filesize);
  int div_work = numDoc / np;
  int mas_work = div_work + numDoc % np;
  double* target;
  double* featureArray;
  if ( rank == 0 ) 
  {
    MPI_Request recvQ[np], tmp;
    double diff[numDoc];
    featureArray = new double[mas_work * numFea];
    MPI_File_set_view(infile, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(infile, featureArray, mas_work*numFea, MPI_DOUBLE, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    MPI_File_close(&infile);
    
    //for test
    target = featureArray + 0*numFea;
    for (int i = 1; i < np; i ++) {
      MPI_Isend(target, numFea, MPI_DOUBLE, i, TARGET, MPI_COMM_WORLD, &tmp);
//      MPI_Isend(featureArray + (mas_work + (i - 1) * div_work) * numFea, div_work * numFea, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD, &tmp);
      MPI_Irecv(diff + mas_work + (i - 1)*div_work, div_work, MPI_DOUBLE, i, DIFF, MPI_COMM_WORLD, &recvQ[i]);
    }
    for(int i = 0; i < mas_work; i++) {
      diff[i] = distance(target, featureArray + i*numFea);
    }
    for(int i = 1; i < np; i++) {
      MPI_Wait(&recvQ[i], 0);
    }
    doc results[numDoc];
    for(int i =0; i < numDoc; i++) {
       results[i].id = i+1;
       results[i].score = diff[i];
    }
    sort(results, results + numDoc, comfunc);
    for(int j = 0; j < 100; j++) {
      cout << j << " " << results[j].id << " " << results[j].score<<endl;
    }
        
    
  } else {
    featureArray = new double[div_work * numFea];
    int previous = numFea * (mas_work + div_work * (rank - 1));
    MPI_File_set_view(infile, previous, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(infile, featureArray, div_work * numFea, MPI_DOUBLE, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    MPI_File_close(&infile);
    target = new double [numFea];
  //  MPI_Recv(featureArray, div_work * numFea, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD, 0);
    MPI_Recv(target, numFea, MPI_DOUBLE, 0, TARGET, MPI_COMM_WORLD, 0);
    double diff[div_work];
    for(int i = 0; i < div_work; i++) {
      diff[i] = distance(target, featureArray + i * numFea);
    }
    MPI_Send(diff, div_work, MPI_DOUBLE, 0, DIFF, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}

    
    
