#include "pCBIR.h"
#include <algorithm>
#include <cstdlib>
#include <string.h>
#include <time.h>
#include <unistd.h>
const int numDoc = 999;
char* filename = "feature_input";
const int TARGET = 0;
const int DATA = 1;
const int DIFF = 2;
const int k = 10;
int numFea;
double distance(double* a, double* b) {
  double diff = 0;
  for(int i = 0; i < numFea - 1; i++) {
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

void initCenter(double* results, int centerN, double* candidate, int numCandidate) {
  srand(time(0));
  int div_size = numCandidate / centerN;
  for(int i = 0; i < centerN; i++) {
    int pick = rand() % div_size;
    double * in = results + i * numFea;
    double * out = candidate + (i * div_size + pick) * numFea;
    for(int j = 0; j < numFea; j++) {
      in[j] = out[j];
    }
  }
}

int findMatch(double *target, double* center, double* sum) {
  double tmp, min = distance(target, center);
  int best = 0;
  for(int i = 1; i < k; i ++) {
    tmp = distance(target, center + i * numFea);
    if(tmp < min) {
      min = tmp;
      best = i;
    }
  }
  if(sum != NULL) {
    double* sum_ptr = sum + best * numFea;
    for(int i = 0; i < numFea; i++) {
      sum_ptr[i] += target[i];
    }
  }
  return best;
}

int change_check(int* Old, int* New) {
  for(int i = 0; i < numDoc; i++) {
    if(Old[i] != New[i]){
      memcpy(Old, New, sizeof(int) * numDoc);
      return 1;
    }
  }
  memcpy(Old, New, sizeof(int) * numDoc);
  return 0;
}

int main(int argc, char** argv) {
  int np, rank, count; 
  MPI_Init(&argc, &argv);
  numFea = atoi(argv[1]) + 1;
  MPI_File infile;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int work_size[np];
  int center_size[np];
  int center_disp[np];
  int work_disp[np];
  for(int i = 0; i < np; i++) {
    if(i == 0) {
      work_size[i] = numDoc / np + numDoc % np;
      center_size[i]= (k / np + k % np) * numFea;
      center_disp[i] = 0;
      work_disp[i] = 0;
    } else {
      work_size[i] = numDoc / np;
      center_size[i] = k / np * numFea;
      center_disp[i] = (k/np + k%np + k/np*(i - 1)) * numFea;
      work_disp[i] = numDoc/np + numDoc %np + (i - 1) * numDoc / np;
    }
  }
  char fn[strlen(filename) + 1];
  sprintf(fn, "%s%d%d",filename, np, rank);
cout << fn << " "<<rank << endl;
  MPI_File_open(MPI_COMM_WORLD, fn, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
  double* target = new double[numFea];
  int* mapper;
  int* tmp_mapper = new int[work_size[rank]];
  int* ele_cnt = new int[k];
  memset(ele_cnt, 0, k * sizeof(int));
  double* featureArray = new double[numFea * work_size[rank]];
  double* center = new double [k * numFea];
  double* partialSum = new double[k * numFea];
  memset(partialSum, 0, k * numFea * sizeof(double));
  double start = MPI_Wtime();
  double* sum;
  int* ele_cnt_sum;
  MPI_File_set_view(infile, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
  MPI_File_read(infile, featureArray, work_size[rank] * numFea, MPI_DOUBLE, &status);
  MPI_File_close(&infile);
  if(rank == 0) {
    ifstream qfile ("feature_inputquery", ios::in | ios::binary);
    qfile.read((char*)target, numFea * sizeof(double));
    qfile.close();
  }
  double readt = MPI_Wtime();
  cout << "proc " << rank << " reading time : " << readt - start << endl;
  
  //std::cout.precision(12);
  //cout <<fixed<< "proc " << rank << " reading time from " << start << "to "<< readt  << endl;
  
  //init centers
  double* tmp_center = new double[center_size[rank]];
  if(rank == 0) {
    mapper = new int[numDoc];
    ele_cnt_sum = new int [k];
    sum = new double[k * numFea];
    memset(sum, 0, k * numFea * sizeof(double));
    initCenter(tmp_center, center_size[rank] / numFea, featureArray, work_size[rank]);
  } else {
    initCenter(tmp_center, center_size[rank] / numFea, featureArray, work_size[rank]);
    target = new double[numFea];
  }
  
  MPI_Bcast(target, numFea, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(tmp_center, center_size[rank], MPI_DOUBLE, center, center_size, center_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);//gather centers
  MPI_Bcast(center, k * numFea, MPI_DOUBLE, 0, MPI_COMM_WORLD);//broadcast centers

  //init center assignment
  int* tmp_id = new int[work_size[rank]];
  for (int i = 0; i < work_size[rank]; i++) {
    tmp_mapper[i] = findMatch(featureArray + i * numFea, center, partialSum);
    ele_cnt[tmp_mapper[i]]++;
    tmp_id[i] = *(featureArray + (i + 1) * numFea - 1);
  }

  int* id_features;
  if(rank == 0)
    id_features = new int[numDoc];
  MPI_Gatherv(tmp_id, work_size[rank], MPI_INT, id_features, work_size, work_disp, MPI_INT, 0, MPI_COMM_WORLD);
  

  //gather mapper
  MPI_Gatherv(tmp_mapper, work_size[rank], MPI_INT, mapper, work_size, work_disp, MPI_INT, 0, MPI_COMM_WORLD);
  //get  sum and ele_cnt
  MPI_Reduce(partialSum, sum, k * numFea, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(ele_cnt, ele_cnt_sum, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  int changed = 1;
  int* new_mapper;
  if(rank ==0) {
    new_mapper = new int[numDoc];
  }
  int counter = 0;
  do {
    counter ++;
    if(rank == 0) {
      //calculate new center
      for(int j = 0; j < k; j++) {
        for(int i = 0; i < numFea; i ++) {
          center[j * numFea + i] = sum[j * numFea + i] / ele_cnt_sum[j];
        }
      }
    }
    //distribute new center
    MPI_Bcast(center, numFea * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //locally calculate new match
    memset(ele_cnt, 0, k * sizeof(int));
    memset(partialSum, 0, k * numFea * sizeof(double));
    for (int i = 0; i < work_size[rank]; i++) {
      tmp_mapper[i] = findMatch(featureArray + i * numFea, center, partialSum);
      ele_cnt[tmp_mapper[i]]++;
    }
    //gather mapper
    MPI_Gatherv(tmp_mapper, work_size[rank], MPI_INT, new_mapper, work_size, work_disp, MPI_INT, 0, MPI_COMM_WORLD);
    //get  sum and ele_cnt
    MPI_Reduce(partialSum, sum, k * numFea, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ele_cnt, ele_cnt_sum, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
      changed = change_check(mapper, new_mapper);
    }
    MPI_Bcast(&changed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }while(changed);

  
  //determine which cluster
  int cindex;
  if(rank ==0) {
    cindex = findMatch(target, center, 0);
  }
  MPI_Bcast(&cindex, 1, MPI_INT, 0, MPI_COMM_WORLD);


  double* scores = new double[work_size[rank]];
  for(int i = 0; i < work_size[rank]; i++) {
      if(tmp_mapper[i] == cindex)
        scores[i] = distance(target, featureArray + i * numFea);
       else 
        scores[i] = -1;
  }
  double* scores_total = new double[numDoc];
  MPI_Gatherv(scores, work_size[rank], MPI_DOUBLE, scores_total, work_size, work_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(rank == 0) {
    doc* scores_filtered = new doc[ele_cnt_sum[cindex]];
    int cnt= 0;
    for(int i = 0; i < numDoc; i++) {
      if(scores_total[i] >= 0) {
        scores_filtered[cnt].id = id_features[i];
        scores_filtered[cnt].score = scores_total[i];
        cnt ++;
      }
      
    }
    
    sort(scores_filtered, scores_filtered + ele_cnt_sum[cindex], comfunc);

    for (int i = 0; i < 10; i ++) {
     cout << scores_filtered[i].id << " " << scores_filtered[i].score << endl;
    }
  }
  double end = MPI_Wtime();


  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
    cout << "total iteration: " << counter << " \ntotal time: " << end - start << endl;
  MPI_Finalize();
  return 0;
}

    
    
