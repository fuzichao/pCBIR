#include "pCBIR.h"
#include <algorithm>
#include <cstdlib>
#include <string.h>
#include <time.h>
const int numDoc = 999;
char* filename = "feature_input";
const int TARGET = 0;
const int DATA = 1;
const int DIFF = 2;
const int k = 20;
int numFea;
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

double* initCenter(int offset, int centerN, double* candidate, int numCandidate) {
  double results = new double [k * numFea];
  memset(results, 0, k * numFea * sizeof(double));
  srand(time(0));
  int div_size = numCandidate / centerN;
  for(int i = 0; i < centerN; i++) {
    int pick = rand() % div_size;
    memcpy(results + (i + offset) * numFea, candidate + i * div_size * numFea, numFea);
  }
  return results;
}

int findMatch(double *target, double* center) {
  double tmp, min = distance(target, center);
  int best = 0;
  for(int i = 1; i < k; i ++) {
    tmp = distance(target, center + i * numFea);
    if(tmp < min) {
      min = tmp;
      best = i;
    }
  }
  return best;
}

int main(int argc, char** argv) {
  int np, rank, count; 
  MPI_Init(&argc, &argv);
  numFea = atoi(argv[1]);
  MPI_File infile;
  MPI_Status status;
  MPI_Offset filesize;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char fn[strlen(filename) + 1];
  sprintf(fn, "%s%d",filename, rank);
  MPI_File_open(MPI_COMM_WORLD, fn, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
  MPI_File_get_size(infile, &filesize);
  int div_work = numDoc / np;
  int mas_work = div_work + numDoc % np;
  double* target;
  double* featureArray = new double[numDoc * numFea];
  int* mapper = new int[numDoc];
  double* buf;
  double* center;//#center * numFea
  if ( rank == 0 ) 
  {
    double start = MPI_Wtime();
    MPI_Request recvQ[np], tmp;
    double diff[numDoc];
    buf = featureArray;
    MPI_File_set_view(infile, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(infile, buf, mas_work*numFea, MPI_DOUBLE, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    MPI_File_close(&infile);
    double readt = MPI_Wtime();
    cout << "proc " << rank << " reading time : " << readt - start << endl;
    //std::cout.precision(12);
    //cout <<fixed<< "proc " << rank << " reading time from " << start << "to "<< readt  << endl;
    
    //init centers
    center = initCenter(0, k / np + k % np, buf, mas_work);
    target = featureArray + 0*numFea; //use the first img as test
    MPI_Bcast(target, numFea, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < np; i ++) {
      if( i == 0)
        MPI_Bcast(center, numFea * (k /np + k % np),  MPI_DOUBLE, i, MPI_COMM_WORLD);
      else
        MPI_Bcast(center + numFea * (k/np+k%np + (i-1)*k/np), numFea * k/np, MPI_DOUBLE, i, MPI_COMM_WORLD);
    }
    //init center assignment
    for (int i = 0; i < mas_work, i++) {
      mapper[i] = findMatch(feature + i * numFea, center);
    }
    for (int i = 0; i < np; i ++) {
      if( i == 0)
        MPI_Bcast(mapper, mas_work, MPI_INT, i, MPI_COMM_WORLD);
      else
        MPI_Bcast(mapper + mas_work + (i-1) * div_work, div_work, MPI_INT, i, MPI_COMM_WORLD);
    }
       


    for (int i = 1; i < np; i ++) {
//      MPI_Isend(featureArray + (mas_work + (i - 1) * div_work) * numFea, div_work * numFea, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD, &tmp);
      MPI_Irecv(diff + mas_work + (i - 1)*div_work, div_work, MPI_DOUBLE, i, DIFF, MPI_COMM_WORLD, &recvQ[i]);
    }
    for(int i = 0; i < mas_work; i++) {
      diff[i] = distance(target, featureArray + i*numFea);
    }
    MPI_Waitall(np - 1, recvQ + 1, 0);
    doc results[numDoc];
    for(int i =0; i < numDoc; i++) {
       results[i].id = i+1;
       results[i].score = diff[i];
    }
    sort(results, results + numDoc, comfunc);
    double end = MPI_Wtime();
/*    for(int j = 0; j < 100; j++) {
      cout << j << " " << results[j].id << " " << results[j].score<<endl;
    }*/
    cout << "total time: " << end - start << endl;
        
    
  } else {
    double start = MPI_Wtime();
    featureArray = new double[div_work * numFea];
    int previous = numFea * (mas_work + div_work * (rank - 1));
    MPI_File_set_view(infile, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(infile, featureArray, div_work * numFea, MPI_DOUBLE, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    MPI_File_close(&infile);
    double readt = MPI_Wtime();
    cout << "proc " << rank << " reading time : " << readt - start << endl;
    //std::cout.precision(12);
    ///cout <<fixed<< "proc " << rank << " reading time from " << start << "to "<< readt  << endl;
    target = new double [numFea];
  //  MPI_Recv(featureArray, div_work * numFea, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD, 0);
    MPI_Bcast(target, numFea, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double diff[div_work];
    for(int i = 0; i < div_work; i++) {
      diff[i] = distance(target, featureArray + i * numFea);
    }
    MPI_Send(diff, div_work, MPI_DOUBLE, 0, DIFF, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}

    
    
