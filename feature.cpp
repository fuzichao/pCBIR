#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <omp.h>
#include <vector>
using namespace cv;
using namespace std;

const char* out_path = "feature_input";
char* img_path = "/home/zichao/image.vary.jpg/";
const double PI = 3.141592653;
const int H_nbins = 18;
const int S_nbins = 3;
const int V_nbins = 3;
const int CELLSIZE = 16;
const int IMG_SIZE = 150;
int imgN = 9907;
const int totalbins = H_nbins * S_nbins * V_nbins;
//suggested by http://scien.stanford.edu/pages/labsite/2002/psych221/projects/02/sojeong
void writeFeatures(int partN, int numFea, double* features, int index) {
  string tmp = out_path;
  char index_str[10];
  memset(index_str, 0, 10); 
  if(partN == -1)
    sprintf(index_str, "query"); 
  else 
    sprintf(index_str, "%d%d", partN,index); 
  tmp.append(index_str);
  ofstream ofile (tmp.c_str(), ios::out|ios::binary | ios::app);
  ofile.write((char*)features, sizeof(double) * numFea);
  ofile.close();
}


int main(int argc, char** argv)
{
  if (argc != 4) { 
    cout << "usage: "<<argv[0]<<" <useHOG>{1/0} -p <partition#> | -f <query image path>\n";
    return -1;
  }
  bool useHOG = atoi(argv[1]);
  int partN = 9907;
  bool query = false;
  if(argv[2][1] == 'p')
    partN = atoi(argv[3]);
  else {
    query = true;
    imgN = 1;
    img_path = argv[3];
  }
  cout << "partition into " << partN << " files" << endl;
  if(useHOG) {
    int size = (IMG_SIZE + CELLSIZE/2) / CELLSIZE;
    cout << "using HOG feature\nfeature size:" << (size * size * 4 * 9)<< endl;
  } else {
    cout << "using color histogram\nfeature size:" << totalbins<< endl;
  }
  double start_t = omp_get_wtime();
  #pragma omp parallel for
  for(int img_id = 1; img_id <= imgN; img_id++) {
    int index;
    string path = img_path;
    if(!query) {
      //mapping img_id to idex
      if ( img_id <= (imgN / partN + imgN % partN) ){
        index = 0;
      }
      else { 
        index = (img_id - (imgN / partN + imgN % partN))/(imgN / partN) + 1;
        if((img_id - (imgN / partN + imgN % partN))%(imgN / partN)==0)
          index --;
      }
    } else {
        index = -1;
    }
    if(!query) {
      char tmpp[10];
      memset(tmpp, 0, 10);
      sprintf(tmpp, "%d", img_id);
      path.append(tmpp);
      path.append(".jpg");
    }
    Mat src, src_HSV;
    Size dsize = Size(IMG_SIZE, IMG_SIZE);//resize image
    if(!useHOG) {
      src = imread( path.c_str() );
      cvtColor( src, src_HSV, CV_BGR2HSV );
      Mat img1;
      #pragma omp critial
      {
        resize(src_HSV, img1, dsize);
      } 
     //color histogram part
      //init an array for historgam
      double hist[totalbins + 1];
      for(int i = 0; i < totalbins; i++){
        hist[i] =0;
      }
      for(int x = 0; x < IMG_SIZE; x++) {
        for(int y = 0; y < IMG_SIZE; y++) {
          Vec3b pixel = img1.at<Vec3b>(x, y);
          double H = pixel[0] * 2;
          double S = (double) pixel[1] / 255.0;
          double V = (double) pixel[2] / 255.0;
          int H_index = fmin(floor( H / 20 ), 17);
          int S_index = fmin(floor( S * 3  ), 2);    
          int V_index = fmin(floor( V * 3  ), 2);   
          hist[H_index * S_nbins * V_nbins + S_index * V_nbins + V_index]++;
        }
      }
      hist[totalbins] = img_id;
      writeFeatures(partN, totalbins + 1, hist, index);
    } else {
      //HOG feature extraction
      Mat img;
      src = imread( path.c_str(), 0);
      #pragma omp critical
      { 
        resize(src, img, dsize);
      }
      int numOrientations = 9;
      int cellSize = CELLSIZE;
      int imgWidth = IMG_SIZE, imgHeight = IMG_SIZE;
      int hogWidth = (imgWidth + cellSize/2) / cellSize;
      int hogHeight = (imgHeight + cellSize/2) /cellSize;
      double orientX[numOrientations];
      double orientY[numOrientations];
      //use DalalTriggs
      int dimension = 4 * numOrientations;

      for(int i = 0; i < numOrientations; i++) {
        double angle = (double)i * PI / numOrientations;
        orientX[i] = cos(angle);
        orientY[i] = sin(angle); 
      }
      //prepare buffer
      int numChannels = 1;
      int k;
      int hogStride = hogWidth * hogHeight;
      int channelStride = imgWidth * imgHeight;
      double* hog = new double[hogWidth * hogHeight * numOrientations * 2];
      memset(hog, 0, sizeof(double)*hogStride*numOrientations*2);
      double* hogNorm = new double[hogStride];
      memset(hogNorm, 0, sizeof(double)*hogStride);
      //compute gradients and map to HOG cells by bilinear interpolation
      for(int y = 1; y < imgHeight - 1; y++) {
        for(int x = 1; x < imgWidth - 1; x++) {
          double gradx = 0, grady =0, grad;
          double orientationWeights[2] = {0, 0};
          int orientationBins[2] = {-1, -1};
          int orientation = 0;
          double hx, hy, wx1, wx2, wy1, wy2;
          int binx, biny, o;
          //compute gradient at (x,y)
          double grad2 = 0;
          for (k = 0; k < numChannels; k++) {
            double gradx_ = img.at<Vec3b>(x + 1, y)[k] - img.at<Vec3b>(x - 1, y)[k];
            double grady_ = img.at<Vec3b>(x, y + 1)[k] - img.at<Vec3b>(x, y -1)[k];
            double grad2_ = gradx_ * gradx_ + grady_ * grady_;
            if (grad2_ > grad2) {
              gradx = gradx_;
              grady = grady_;
              grad2 = grad2_;
            }  
          }
          grad = sqrtf(grad2);
          gradx /= fmax(grad, 1e-10);
          grady /= fmax(grad, 1e-10);
          for (k = 0; k < numOrientations; k++) {
            double oScore = gradx * orientX[k] + grady * orientY[k];
            int oBin = k;
            if(oScore < 0) {
              oScore = -1 * oScore;
              oBin += numOrientations;
            }
            if (oScore > orientationWeights[0]) {
              orientationBins[1] = orientationBins[0];
              orientationWeights[1] = orientationWeights[0];
              orientationBins[0] = oBin;
              orientationWeights[0] = oScore;
            } else if(oScore > orientationWeights[1]) {
              orientationBins[1] = oBin;
              orientationWeights[1] = oScore;
            }
          }
          
          orientationWeights[0] = 1;
          orientationBins[1] = -1;
          //Accumulate the gradient
          for(o = 0; o < 2; o++) {
            orientation = orientationBins[o];
            if (orientation < 0) continue;
            hx = (x + 0.5) / cellSize - 0.5;
            hy = (y + 0.5) / cellSize - 0.5;
            binx = floor(hx);
            biny = floor(hy);
            wx2 = hx - binx;
            wy2 = hy - biny;
            wx1 = 1.0 - wx2;
            wy1 = 1.0 - wy2;
            wx1 *= orientationWeights[o];
            wx2 *= orientationWeights[o];
            wy1 *= orientationWeights[o];
            wy2 *= orientationWeights[o];
          
            if (binx >= 0 && biny >= 0) {
              hog[binx + biny * hogWidth + orientation * hogStride] += grad * wx1 *wy1;
            }
            if (binx < hogWidth - 1 && biny >= 0) {
              hog[binx + 1+ biny * hogWidth + orientation * hogStride] += grad * wx2 *wy1;
            }
            if (binx < hogWidth -1 && biny < hogHeight - 1) {
              hog[binx + 1+ (biny + 1) * hogWidth + orientation * hogStride] += grad * wx2 *wy2;
            }
            if (binx >= 0 && biny < hogHeight - 1) {
              hog[binx + (1 + biny) * hogWidth + orientation * hogStride] += grad * wx1 *wy2;
            }
          }
        }
      }
      double* hogFeature = new double[1 + hogWidth * hogHeight * dimension];
      //extract results
      //compute the L2 norm
      double * iter = hog;
      for (k =0; k < numOrientations; k++) {
        double* current = hogNorm;
        double* end = hogNorm + hogWidth * hogHeight; 
        int stride = hogWidth * hogHeight * numOrientations;
        while ( current != end) {
          double h1 = *iter;
          double h2 = *(iter + stride);
          *current += (h1 + h2) * (h1 + h2);
          current ++;
          iter ++;
        }
      }
      //block-normlization
      iter = hog;
      for (int y = 0; y < hogHeight; y++) {
        for (int x = 0; x < hogWidth; x++) {
          int xm = fmax ( x - 1, 0);
          int xp = fmin ( x + 1, hogWidth - 1);
          int ym = fmax ( y - 1, 0);
          int yp = fmin ( y + 1, hogHeight - 1);
        
          double norm1 = hogNorm[xm + ym * hogWidth];
          double norm2 = hogNorm[x + ym * hogWidth];
          double norm3 = hogNorm[xp + ym * hogWidth];
          double norm4 = hogNorm[xm + y * hogWidth];
          double norm5 = hogNorm[x + y * hogWidth];
          double norm6 = hogNorm[xp + y * hogWidth];
          double norm7 = hogNorm[xm + yp * hogWidth];
          double norm8 = hogNorm[x + yp *hogWidth];
          double norm9 = hogNorm[xp + yp * hogWidth];
          double factor1 = 1.0 / sqrt(norm1 + norm2 + norm4 + norm5 + 1e-4) ;
          double factor2 = 1.0 / sqrt(norm2 + norm3 + norm5 + norm6 + 1e-4) ;
          double factor3 = 1.0 / sqrt(norm4 + norm5 + norm7 + norm8 + 1e-4) ;
          double factor4 = 1.0 / sqrt(norm5 + norm6 + norm8 + norm9 + 1e-4) ;
          
          double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
          double* feature_iter = hogFeature + x + hogWidth * y;
          
          for (k = 0; k < numOrientations; k++) {
            double ha = iter[hogStride * k];
            double hb = iter[hogStride * (k + numOrientations)];
            double hc;
            *feature_iter = fmin(0.2, factor1 * (ha + hb));
            *(feature_iter + hogStride * numOrientations) = fmin(0.2, factor2 * (ha + hb));
            *(feature_iter +2 * hogStride * numOrientations) = fmin(0.2, factor3 * (ha + hb));
            *(feature_iter +3 * hogStride * numOrientations) = fmin(0.2, factor4 * (ha + hb));
            feature_iter += hogStride;
          }
          
          iter ++;
         }
      }
      hogFeature[hogWidth * hogHeight * dimension] = img_id;
      writeFeatures(partN, hogWidth * hogHeight * dimension + 1, hogFeature, index);
      delete[] hogFeature;
      delete[] hog;     
      delete[] hogNorm;
    }
  }
  double end_t = omp_get_wtime();
  cout << "feature extraction time: " << end_t - start_t << endl;
  return 0;
}

