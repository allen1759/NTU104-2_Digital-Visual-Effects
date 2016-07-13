#ifndef MySIFT_H
#define MySIFT_H
#include "Image.h"

#include <cmath>

class MySIFT
{
public:
	/*the parameters of scale space*/
	int octave_num;							//number of octaves 
	const int s;									//number of scales sampled per octave: 3
	const float baseSigma;			//base scale is 1.6 from the empirical data
	vector<float> sigmas;

	/*the image data*/
	vector<Image>* images;
	vector<Mat>	gray_sift;
	
	/*the  paramters of the detector*/
	const float contrastThreshold;			//throw out low contrast
	float curvaturesThreshold;		//throw out edge points
	
	
	MySIFT(vector<Image>* images,int,int);
	void pre_process();
	void detection();
	void description();
	void buildGaussianPyramid(int);
	void buildDoGPyramid(int);
	void findExtrema(int);
	bool adjustLocalExtrema(int, int, int, int , int, int, int);
	void assignOrientation(int);
	void createDescriptors(int);
	float*** createDescriptor(Mat, Feature, Point2f, float, int, float);
	void hist_to_descriptor(float***, FeaturePoint*, int);
	void matching();
	vector<KeyPoint> convertToCV_Kpts(int);
	Mat convertToCV_Descriptor(int);
	vector<DMatch> convertToCV_Matches(int);
};
#endif