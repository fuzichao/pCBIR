#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
using namespace cv;
using namespace std;

const char* img_path = "/home/zichao/images/";
const double PI = 3.141592653;
const int H_nbins = 18;
const int S_nbins = 3;
const int V_nbins = 3;
const int totalbins = H_nbins * S_nbins * V_nbins;
//suggested by http://scien.stanford.edu/pages/labsite/2002/psych221/projects/02/sojeong

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

int main(int argc, char** argv)
{
string path = img_path;
cout << 100 << " " << 3600 << endl;
for(int kk = 1; kk <= 100; kk++) {
path = img_path;
path.append("im");
char tmpp[10];
memset(tmpp, 0, 10);
sprintf(tmpp, "%d", kk);
path.append(tmpp);
path.append(".jpg");
  Mat src, src_HSV;
  src = imread( path.c_str() );
  cvtColor( src, src_HSV, CV_BGR2HSV );
  Size dsize = Size(500, 500);//resize image
  Mat img1;// = Mat(dsize, CV_32S);
  resize(src_HSV, img1, dsize);
//test
/*
  cout << "dims: " << img.dims;
  cout << " rows:" << img.rows;
  cout << " cols:" << img.cols;
  cout << " flags:" << img.flags;
  cout << " type:" << getImageType(img.type());
  cout << endl; 
  Vec3b pixel = img.at<Vec3b>(1,1);
  for (int i = 0; i < 6; i++)
    cout << "H:" << (unsigned int)pixel[i] <<endl;
//  imshow( "test", img );
//  waitKey(0);
*/
//color histogram part
//init an array for historgam
  double hist[totalbins];
  for(int i = 0; i < totalbins; i++){
    hist[i] =0;
  }
//serial version: traverse each pixels one by one
  for(int x = 0; x < 500; x++) {
    for(int y = 0; y < 500; y++) {
      Vec3b pixel = img1.at<Vec3b>(x, y);
      double H = pixel[0] * 2;
      double S = (double) pixel[1] / 255.0;
      double V = (double) pixel[2] / 255.0;
      int H_index = fmin(floor( H / 20 ), 17);
      int S_index = fmin(floor( S * 3  ), 2);    
      int V_index = fmin(floor( V * 3  ), 2);   
      hist[H_index * S_nbins * V_nbins + S_index * V_nbins + V_index]++;
      if(src.dims > 5) {
        cout <<x<< " " <<y <<endl;
        cout <<"a"<<endl;
        return -1;
      }
    }
  }

//show results
/*
  for(int i=0; i < totalbins; i++){
    int h = i / (S_nbins * V_nbins);
    int s = (i - h * (S_nbins * V_nbins)) / V_nbins;
    int v = i % V_nbins;
    cout << "["<< h << ", " << s << ", " << v << "]: "<<(int)hist[i]<<endl;
  }
*/
/*  Mat t, tt;
  
  t = imread( img_path , 0);
//  cvtColor( src, t, CV_BGR2GRAY );//change to gray scale  
  SiftFeatureDetector dt;
  vector<KeyPoint> keys;
  dt.detect(t, keys);
  drawKeypoints(t, keys, tt);
  imshow("result ",tt);
  waitKey(0);
*/
//HOG feature extraction
Mat img;
  resize(src, img, dsize);


  int numOrientations = 9;
  int glyphSize = 21;
  int cellSize = 50;
  int imgWidth = 500, imgHeight = 500;
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
  
  double glyphs[glyphSize * glyphSize * numOrientations];
  for(int i = 0; i < numOrientations; i++) {
    double angle = fmod((double)i * PI / (double)numOrientations + PI/2, PI);
    double x2 = glyphSize * cos(angle) / 2;
    double y2 = glyphSize * sin(angle) / 2;
    
    if( angle <= PI/4 || angle >= PI* 0.75) {
      double slope = y2 / x2;
      double offset = (1 - slope) * (glyphSize - 1) / 2;
      int skip = (1 - fabs(cos(angle))) / 2 * glyphSize;
      int j, k;
      for(j = skip; j < glyphSize - skip; j++) {
        k = round(slope * j + offset);
        glyphs[j + glyphSize * k + glyphSize * glyphSize * i] = 1;
      }
    } else {
      double slope = x2 / y2;
      double offset = (1 - slope) * (glyphSize - 1) / 2;
      int skip = (1 - fabs(sin(angle))) / 2 * glyphSize;
      int j, k;
      for(j = skip; j < glyphSize - skip; j++) {
        k = round(slope * j + offset);
        glyphs[k + glyphSize * j + glyphSize * glyphSize * i] = 1;
      }
    }
  }
  //prepare buffer
  int numChannels = 3;
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
        wx1 *= orientationWeights[0];
        wx2 *= orientationWeights[0];
        wy1 *= orientationWeights[0];
        wy2 *= orientationWeights[0];
      
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
  double* hogFeature = new double[hogWidth * hogHeight * dimension];
  //extract results
  /*compute the L2 norm*/
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
  /*block-normlization*/
  iter = hog;
  for (int y = 0; y < hogHeight; y++) {
    for (int x = 0; x < hogWidth; x++) {
      int xm = fmax ( x - 1, 0);
      int xp = fmin ( x + 1, hogWidth - 1);
      int ym = fmax ( y - 1, 0);
      int yp = fmax ( y + 1, hogHeight - 1);
    
      double norm1 = hogNorm[xm + ym * hogWidth];
      double norm2 = hogNorm[x + ym * hogWidth];
      double norm3 = hogNorm[xp + ym * hogWidth];
      double norm4 = hogNorm[xm + y * hogWidth];
      double norm5 = hogNorm[x + y * hogWidth];
      double norm6 = hogNorm[xp + yp * hogWidth];
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
  //cout<< "size is " << hogStride * dimension << endl; 
  for(int i = 0; i < hogStride * dimension; i++) {
    cout << hogFeature[i] << " ";
  }
  cout << endl;
  delete[] hogFeature;
  delete[] hog;     
  delete[] hogNorm;
  src.release();
  src_HSV.release();
  img.release(); 
}
  return 0;
}

