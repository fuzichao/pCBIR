#include "pCBIR.h"

double* readFeatures(const char* filename) {
	ifstream ifs(filename);
	int pictureN, featureN;
	double cur_feature;
	ifs >> pictureN;
	ifs >> featureN;
	int allFeaturesN = pictureN*featureN;

	double* allFeatures = new double[allFeaturesN];
	for(int i = 0; i < allFeaturesN; ++i) {
		ifs >> cur_feature;
		allFeatures[i] = cur_feature;
	}

	return allFeatures;
}
