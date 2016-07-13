#pragma warning(disable:4996)
#define _SCL_SECURE_NO_WARNINGS
#include "MySIFT.h"
#include "opencv2/flann/flann_base.hpp"
#include <iostream>
#include <algorithm>
using namespace std;
using namespace cv;
using namespace cvflann;

#define PI 3.14159
#define IMG_BORDER 5
#define SIFT_INIT_SIGMA  0.5f																					//assumed gaussian blur for input image
#define SIFT_ORI_HIST_BINS 36																				//number of bins in histogram for orientation assignment: 360 degrees => 36 bins * 10 degrees/bin
#define SIFT_ORIEN_SIGMA_FACTOR 1.5f															//determines gaussian sigma for orientation assignment
#define SIFT_ORI_RADIUS  3 * SIFT_ORIEN_SIGMA_FACTOR							//determines the radius of the region used in orientation assignment
#define SIFT_ORIEN_PEAK_RATIO 0.8f																	//orientation magnitude relative to max that results in new feature
#define interp_hist_peak( l, c, r ) ( ((r)-(l)) / (2 * ((l) -2.0*(c) + (r))) )
#define SIFT_DESCR_SCL_FCTR 3.f																			//determines the size of a single descriptor orientation histogram
#define SIFT_DESCR_HIST_ARRAY_WIDTH 4														//the width of descriptor histogram array (histogram array: 4x4)
#define SIFT_DESCR_HIST_BINS 8																			//the number of bins per histogram in descriptor array
#define SIFT_DESCR_MAG_THR 0.2f																		// threshold on magnitude of elements of descriptor vector
#define SIFT_MATCH_DISTANCE_RATIO 0.49														//threshold on squared ratio of distances between NN and 2nd NN	 (Lowe uses 0.8, RobHess uses 0.49)	

MySIFT::MySIFT(vector<Image>* imgs, int height, int width) :
	s(3), baseSigma(1.6f), contrastThreshold(0.02f), curvaturesThreshold(10.0f)
{
	images = imgs;

	/*initialize paramters for constructing scale space*/
	/*min(h/(2^(n-1)),w/(2^(n-1)))>=3 
	The width and height of gaussian images in the highest octave should at least have 3 pixels for  findExtrema.*/
	octave_num = cvRound(log(min(height,width)-log(3.0))/log(2.0)+1); 
	cout << "Execute pre_processing.\n";
	pre_process();
	cout << "Execute feature detection:\n";
	detection();
	cout << "Execute feature description.\n";
	description();
	cout << "Execute feature matching.\n";
	matching();
}
void MySIFT::pre_process()
{
	/*Pre-smooth gray image with Gaussian filter and convert pixel value from [0,255] to [0,1].
		The image is  doubled in size prior to smoothing.*/ 
	float sig_diff = sqrt(max(baseSigma*baseSigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4.0, 0.01));
	for(int i=0; i<(*images).size(); i++)
	{
		Mat gray;
		(*images)[i].gray_image.convertTo(gray, CV_32FC1, 1.0f / 255.0f, 0.0f); //[0,255]-> [0,1]
		resize(gray, gray, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);	//double the size of the input image using linear interploation in order to create more sample points
		GaussianBlur(gray, gray, Size(0,0), sig_diff, sig_diff);		//pre-smooth gray image with Gaussian filter to avoid aliasing
		gray_sift.push_back(gray);
	}
}
void MySIFT::detection()
{
	/*Scale-space extrema detection*/
	/*Construct scale spaces for each image*/
	cout << "\tBuild Gaussina Pyramid.\n";
	for (int i = 0; i < (*images).size(); i++) 
	{
		/*build Gaussian Pyramid*/
		buildGaussianPyramid(i);
		// show gaussian-blurred images
		/*for (int j = 0; j < (*images)[i].octaves.size(); j++)
		{
			for (int k = 0; k < (*images)[i].octaves[j].gaussianImgs.size(); k++){
				imshow("Keypoints", (*images)[i].octaves[j].gaussianImgs[k]);
				cvWaitKey(0);
			}
		}*/
	}
	cout << "\tBuild DoG Pyramid.\n";
	for (int i = 0; i < (*images).size(); i++) 
	{
		/*build DoG Pyramid*/
		buildDoGPyramid(i);
		// show DoG images
		/*for (int j = 0; j < (*images)[i].octaves.size(); j++)
		{
			for (int k = 0; k < (*images)[i].octaves[j].DogImgs.size(); k++){
				Mat temp = (*images)[i].octaves[j].DogImgs[k];
				double min_val = 10000.0;
				double max_val = -10000.0;
				for (int m = 0; m < temp.rows; m++)
				{
					for (int n = 0; n < temp.cols; n++)
					{
						if (temp.at<float>(m, n) < min_val)
							min_val = temp.at<float>(m, n);
						if (temp.at<float>(m, n) > max_val)
							max_val = temp.at<float>(m, n);
					}
				}
				if (min_val < 0.0)
				{
					for (int m = 0; m < temp.rows; m++)
						for (int n = 0; n < temp.cols; n++)
							temp.at<float>(m, n) += (-1.0f)*(float)min_val;
				}
				Mat temp2 = cvCreateImage(cvSize(temp.rows, temp.cols), IPL_DEPTH_8U, 1);
				convertScaleAbs(temp, temp2, 255.0 / (max_val - min_val), 0);
				imshow("Keypoints", temp2);
				cvWaitKey(0);
			}
		}*/
	}
	cout << "\tFind extrema and assign orientations.\n";
	/*Keypoint Location*/
	for (int i = 0; i < (*images).size(); i++)
	{
		/*Locate accurate maxima/minima in DoG images*/
		findExtrema(i);
		/*Orientation assignment*/
		assignOrientation(i);
	}
	
	/*adjust image size (The images were doubled before.)*/
	for (int i = 0; i < (*images).size(); i++)
	{
		for (int j = 0; j < (*images)[i].features.size(); j++){
			Feature* f = &(*images)[i].features[j];
			f->position.x /= 2;
			f->position.y /= 2;
			f->size /= 2;
		}
	}
	
}
void MySIFT::description()
{
	for (int i = 0; i < (*images).size(); i++)
	{
		/*Keypoint descriptor*/
		createDescriptors(i);
	}
}
void MySIFT::buildGaussianPyramid(int ith)
{
	/*generate  sigmas: sigma_{total}^2 = sigma_{i}^2 + sigma_{i-1}^2*/
	float K = pow(2.0, 1.0 / s);
	sigmas.push_back(baseSigma);
	for (int i = 1; i < s + 3; i++)
	{
		float sig_prev = pow(K, (float)(i - 1))*baseSigma;
		float sig_total = sig_prev*K;
		sigmas.push_back(sqrt(sig_total*sig_total - sig_prev*sig_prev));
	}

	/*We have to generate s+3 blurred images in the gaussian pyramid.
	This is becuase s+3 blurred images generates s+2 DOG images,
	and two images are needed (one at the highest and one lowest scales of the octave) for extrema detection.*/
	(*images)[ith].octaves.resize(octave_num);
	Mat ithImg = gray_sift[ith];			//the image passing through preprocessing
	for (int i = 0; i <  octave_num; i++)
	{
		Octave tmp;
		for (int j = 0; j < s + 3; j++)
		{
			Mat gaussianImg;
			if (i==0 && j==0){	//the first scale of the first octave
				gaussianImg = ithImg;
			}
			else if (j == 0){		//the first scale of the next octave is smaller twice times than the last ocatve
				Mat src = (*images)[ith].octaves[i - 1].gaussianImgs[s];
				resize(src, gaussianImg, Size(src.cols / 2, src.rows / 2), INTER_LINEAR);
			}
			else{
				Mat src = (*images)[ith].octaves[i].gaussianImgs[j-1];
				GaussianBlur(src, gaussianImg, Size(0, 0), sigmas[j], sigmas[j]);
			}
			(*images)[ith].octaves[i].gaussianImgs.push_back(gaussianImg);
		}
	}
}
void MySIFT::buildDoGPyramid(int ith)
{
	for (int i = 0; i < octave_num; i++)
	{
		for (int j = 0; j < s + 2; j++)
		{
			Mat src1 = (*images)[ith].octaves[i].gaussianImgs[j + 1];
			Mat src2 = (*images)[ith].octaves[i].gaussianImgs[j];
			Mat dog = Mat(src1.rows, src1.cols, CV_32FC1);
			subtract(src2, src1, dog);
			(*images)[ith].octaves[i].DogImgs.push_back(dog);
		}
	}
}
bool isLocalExtrema(float val, int currentPixel, float* current, float* pre, float* next, int steps)
{
	//look for local maximum
	if (val >= *(current + currentPixel - 1) && val >= *(current + currentPixel + 1) &&
		val >= *(current - steps + currentPixel - 1) && val >= *(current - steps + currentPixel) && val >= *(current - steps + currentPixel + 1) &&
		val >= *(current + steps + currentPixel - 1) && val >= *(current + steps + currentPixel) && val >= *(current + steps + currentPixel + 1) &&
		val >= *(pre + currentPixel) && val >= *(pre + currentPixel - 1) && val >= *(pre + currentPixel + 1) &&
		val >= *(pre - steps + currentPixel - 1) && val >= *(pre - steps + currentPixel) && val >= *(pre - steps + currentPixel + 1) &&
		val >= *(pre + steps + currentPixel - 1) && val >= *(pre + steps + currentPixel) && val >= *(pre + steps + currentPixel + 1) &&
		val >= *(next + currentPixel) && val >= *(next + currentPixel - 1) && val >= *(next + currentPixel + 1) &&
		val >= *(next - steps + currentPixel - 1) && val >= *(next - steps + currentPixel) && val >= *(next - steps + currentPixel + 1) &&
		val >= *(next + steps + currentPixel - 1) && val >= *(next + steps + currentPixel) && val >= *(next + steps + currentPixel + 1) 
		){
			return true;
	}
	//look for local minimum
	if (val <= *(current + currentPixel - 1) && val <= *(current + currentPixel + 1) &&
		val <= *(current - steps + currentPixel - 1) && val <= *(current - steps + currentPixel) && val <= *(current - steps + currentPixel + 1) &&
		val <= *(current + steps + currentPixel - 1) && val <= *(current + steps + currentPixel) && val <= *(current + steps + currentPixel + 1) &&
		val <= *(pre + currentPixel) && val <= *(pre + currentPixel - 1) && val <= *(pre + currentPixel + 1) &&
		val <= *(pre - steps + currentPixel - 1) && val <= *(pre - steps + currentPixel) && val <= *(pre - steps + currentPixel + 1) &&
		val <= *(pre + steps + currentPixel - 1) && val <= *(pre + steps + currentPixel) && val <= *(pre + steps + currentPixel + 1) &&
		val <= *(next + currentPixel) && val <= *(next + currentPixel - 1) && val <= *(next + currentPixel + 1) &&
		val <= *(next - steps + currentPixel - 1) && val <= *(next - steps + currentPixel) && val <= *(next - steps + currentPixel + 1) &&
		val <= *(next + steps + currentPixel - 1) && val <= *(next + steps + currentPixel) && val <= *(next + steps + currentPixel + 1)
		){
		return true;
	}
	return false;
}
/*Interpolates a scale-space extremum's location and scale to subpixel accuracy to form an image feature.*/
bool MySIFT::adjustLocalExtrema(int ith, int r, int c, int theOctave, int theLayer, int rows, int cols)
{
	const int max_rounds = 5;
	float offset_c = 0.0, offset_r = 0.0, offset_s = 0.0;
	int rounds = 0;

	Feature f;
	f.originPos = Point(c, r);
	
	/*refine location of keypoint*/ 
	while (rounds < max_rounds)
	{
		Mat currentDoG = (*images)[ith].octaves[theOctave].DogImgs[theLayer];
		Mat preDoG = (*images)[ith].octaves[theOctave].DogImgs[theLayer - 1];
		Mat nextDoG = (*images)[ith].octaves[theOctave].DogImgs[theLayer + 1];
		//construct gradient 
		float dx = (currentDoG.at<float>(r, c + 1) - currentDoG.at<float>(r, c - 1)) / 2.0f;
		float dy = (currentDoG.at<float>(r + 1, c) - currentDoG.at<float>(r - 1, c)) / 2.0f;
		float ds = (nextDoG.at<float>(r, c) - preDoG.at<float>(r, c)) / 2.0f;
		Vec3f dD = (dx, dy, ds);
		//construct Hessian matrix
		float v2 = (float)currentDoG.at<float>(r, c) * 2.0f;
		float dxx = currentDoG.at<float>(r, c + 1) + currentDoG.at<float>(r, c - 1) - v2;
		float dyy = currentDoG.at<float>(r + 1, c) + currentDoG.at<float>(r - 1, c) - v2;
		float dss = nextDoG.at<float>(r, c) - preDoG.at<float>(r, c) - v2;
		float dxy = (currentDoG.at<float>(r + 1, c + 1) - currentDoG.at<float>(r + 1, c - 1) - currentDoG.at<float>(r - 1, c + 1) + currentDoG.at<float>(r - 1, c - 1)) / 4.0;
		float dxs = (nextDoG.at<float>(r, c + 1) - nextDoG.at<float>(r, c - 1) - preDoG.at<float>(r, c + 1) + preDoG.at<float>(r, c - 1)) / 4.0;
		float dys = (nextDoG.at<float>(r + 1, c) - nextDoG.at<float>(r - 1, c) - preDoG.at<float>(r + 1, c) + preDoG.at<float>(r - 1, c)) / 4.0;
		Matx33f Hessian(dxx, dxy, dxs,
										dxy, dyy, dys,
										dxs, dys, dss);
		//refine the location of the extrema: HX = -G, slove X
		Vec3f X = Hessian.solve(dD, DECOMP_LU);
		offset_c = -X[0];
		offset_r = -X[1];
		offset_s = -X[2];
		
		if (abs(offset_c) > (float)(INT_MAX / 3) || abs(offset_r) > (float)(INT_MAX / 3) || abs(offset_s) > (float)(INT_MAX / 3))
			return false;	
		
		//update the location and scale
		c += cvRound(offset_c);
		r += cvRound(offset_r);
		theLayer += cvRound(offset_s);

		if (theLayer > s || theLayer < 1 || c<IMG_BORDER || r<IMG_BORDER || (c >= cols - IMG_BORDER) || (r >= rows - IMG_BORDER))
			return false;

		//change sample point if offset is larger than 0.5 because  
		if (abs(offset_c) < 0.5 && abs(offset_r) < 0.5 && abs(offset_s) < 0.5)
			break;

		rounds++;
	}
	
	if (rounds >= max_rounds)//ensure covergence
		return false;
	
	Mat new_currentDoG = (*images)[ith].octaves[theOctave].DogImgs[theLayer];
	Mat new_preDoG = (*images)[ith].octaves[theOctave].DogImgs[theLayer - 1];
	Mat new_nextDoG = (*images)[ith].octaves[theOctave].DogImgs[theLayer + 1];

	/*discarding low-contrast keypoint*/
	float new_dx = (new_currentDoG.at<float>(r, c + 1) - new_currentDoG.at<float>(r, c - 1)) / 2.0;
	float new_dy = (new_currentDoG.at<float>(r + 1, c) - new_currentDoG.at<float>(r - 1, c)) / 2.0;
	float new_ds = (new_nextDoG.at<float>(r, c) - new_preDoG.at<float>(r, c)) / 2.0;
	Matx31f dD = Matx31f(new_dx, new_dy, new_ds);
	Matx31f offset = Matx31f(offset_c, offset_r, offset_s);
	float accurate_local_extrema = new_currentDoG.at<float>(r, c) + (1 / 2) * dD.dot(offset);

	if (abs(accurate_local_extrema) < contrastThreshold) //Lowe uses 0.03, but Rob Hess use 0.04/s
		return false;

	/*eliminating edge esponse*/
	curvaturesThreshold = pow(curvaturesThreshold + 1.0f, 2.0f) / curvaturesThreshold;
	float new_v2 = (float)new_currentDoG.at<float>(r, c) * 2;
	float new_dxx = new_currentDoG.at<float>(r, c + 1) + new_currentDoG.at<float>(r, c - 1) - new_v2;
	float new_dyy = new_currentDoG.at<float>(r + 1, c) + new_currentDoG.at<float>(r - 1, c) - new_v2;
	float new_dxy = (new_currentDoG.at<float>(r + 1, c + 1) - new_currentDoG.at<float>(r + 1, c - 1) - new_currentDoG.at<float>(r - 1, c + 1) + new_currentDoG.at<float>(r - 1, c - 1)) / 4.0;
	float Tr = new_dxx + new_dyy;
	float Det = new_dxx*new_dyy - new_dxy*new_dxy;
	if (Det<=0.0 || (Tr*Tr) >= (curvaturesThreshold*Det))
		return false;
	
	/*label the keypoint*/
	float x = (c + offset_c) * (float)pow(2, theOctave);
	float y = (r + offset_r) * (float)pow(2, theOctave);
	f.position = Point2f(x, y);
	f.octave = theOctave;
	f.layer = theLayer + cvRound(offset_s);
	f.scale = baseSigma *  (float)pow(2.0, theOctave + (theLayer + offset_s) / s);
	f.size = baseSigma *  (float)pow(2.0, (theLayer + offset_s) / s) * pow(2, theOctave) * 2;
	f.response = (float)abs(accurate_local_extrema);
	(*images)[ith].features.push_back(f);

	return true;
}
/* Detects features at extrema in DoG scale space.  Bad features are discarded based on contrast and ratio of principal curvatures.*/
void MySIFT::findExtrema(int ith)
{
	Image tempImage = (*images)[ith];
	for (int i = 0; i < octave_num; i++)
	{
		/*Ignore the lowermost and topmost scales, because they don't have enough neighbors to do the comparison*/
		for (int j = 1; j <= s; j++)
		{
			Mat currentDoG = tempImage.octaves[i].DogImgs[j];
			Mat preDoG = tempImage.octaves[i].DogImgs[j - 1];
			Mat nextDoG = tempImage.octaves[i].DogImgs[j + 1];
			
			int rows = currentDoG.rows;
			int cols = currentDoG.cols;

			float* currentPtr = (float*)currentDoG.data;
			float* prePtr = (float*)preDoG.data;
			float* nextPtr = (float*)nextDoG.data;

			/*ignore the border of the DoG image*/
			currentPtr += (IMG_BORDER*cols);
			prePtr += (IMG_BORDER*cols);
			nextPtr += (IMG_BORDER*cols);
			for (int r = IMG_BORDER; r < rows - IMG_BORDER; r++)
			{
				for (int c = IMG_BORDER; c < cols - IMG_BORDER; c++)
				{
					float val = *(currentPtr + c);
					if (val <= 0.5 * contrastThreshold/s)
						continue;
					if (!isLocalExtrema(val, c, currentPtr, prePtr, nextPtr, (int)currentDoG.step1()))
						continue;
					/*refine the kepoint position*/
					if (!adjustLocalExtrema(ith, r, c, i, j, rows, cols))
						continue;
				}
				currentPtr += cols;
				prePtr += cols;
				nextPtr += cols;
			}
		}
	}

}
/* Computes a gradient orientation histogram at a specified pixel*/
void MySIFT::assignOrientation(int ith)
{
	int num_kpt = (*images)[ith].features.size();
	
	for (int n = 0; n < num_kpt; n++)
	{
		/*prepare parameters for the following computation of orientation*/
		Feature f = (*images)[ith].features[n];
		Mat img = (*images)[ith].octaves[f.octave].gaussianImgs[f.layer];
		Point pt = Point(f.originPos.x,f.originPos.y);
		int sigma = f.size / (pow(2, f.octave)*2);
		int radius = cvRound(SIFT_ORI_RADIUS*sigma);
		float scaled_sigma = SIFT_ORIEN_SIGMA_FACTOR*sigma;

		/*prepare a histogram for orientations*/
		vector<float> histogram;
		histogram.resize(SIFT_ORI_HIST_BINS, 0.0f);
		
		for (int i = -radius; i <= radius; i++)
		{
			int index_i = (int)pt.y + i;
			if (index_i <= IMG_BORDER)	continue;
			else if (index_i >= img.rows - 1- IMG_BORDER)	break;

			for (int j = -radius; j <= radius; j++)
			{
				int index_j = (int)pt.x + j;
				if (index_j <= IMG_BORDER)	continue;
				else if (index_j >= img.cols - 1- IMG_BORDER)	break;

				/*computie weighted gradient magnitude*/
				float dx = (img.at<float>(index_i, index_j + 1) - img.at<float>(index_i, index_j - 1)) / 2.0f;
				float dy = (img.at<float>(index_i + 1, index_j) - img.at<float>(index_i - 1, index_j)) / 2.0f;
				float magnitude = sqrt(dx*dx + dy*dy);
				float orientation = fastAtan2(dy, dx);	//0-360 degrees, the direction of 0 degree is mapped to the x-axis, and counterwise
				float gaussian_coef = exp(-(i*i + j*j) / (2 * scaled_sigma*scaled_sigma));
				float weighted_magnitude = magnitude * gaussian_coef;

				/*construct the histogram of  orientation bins*/
				int bin = cvRound(orientation/(360.f/SIFT_ORI_HIST_BINS));
				
				if (bin < 0)
					bin += SIFT_ORI_HIST_BINS;
				else if (bin >= SIFT_ORI_HIST_BINS)
					bin -= SIFT_ORI_HIST_BINS;

				histogram[bin] += weighted_magnitude;
			}
		}
		/*Next, smooth the histogram*/
		float tmp, pre = histogram[SIFT_ORI_HIST_BINS - 1], h0 = histogram[0];
		for (int i = 0; i < SIFT_ORI_HIST_BINS; i++)
		{
			tmp = histogram[i];
			//採用[0.25,0.5,0.25]的模版，作高斯模糊
			histogram[i] = 0.25 * pre + 0.5 * histogram[i] + 0.25 * ((i + 1 == SIFT_ORI_HIST_BINS) ? h0 : histogram[i + 1]);
			pre = tmp;
		}
	
		/*Finally, we'll find any other local peak that is within 80% of the hightest peak, 
			and we will create another keypoint with that orientation.
			So, multiple keypoints are created at the same location and scale but different orientation.*/

		// find the max value in the orientation histogram
		float max_bin_hist = *max_element(histogram.begin(), histogram.end());
		// find the sub max value 
		float sub_max_threshold = SIFT_ORIEN_PEAK_RATIO*max_bin_hist;
		float accurate_bin;
		for (int k = 0; k < SIFT_ORI_HIST_BINS; k++)
		{
			int left_bin = (k == 0) ? SIFT_ORI_HIST_BINS - 1 : k - 1;
			int right_bin = (k + 1) % SIFT_ORI_HIST_BINS;
			// find local peaks and their values have to more than 80% of global peak value
			if (histogram[k]>histogram[left_bin] && histogram[k] > histogram[right_bin] && histogram[k] > sub_max_threshold)
			{
				// calculate the accurate bin
				accurate_bin = k + interp_hist_peak(histogram[left_bin], histogram[k], histogram[right_bin])*(-.5);
				// map bin to [0, 35]
				accurate_bin = (accurate_bin < 0) ? (SIFT_ORI_HIST_BINS + accurate_bin) : (accurate_bin >= SIFT_ORI_HIST_BINS) ? (accurate_bin - SIFT_ORI_HIST_BINS) : accurate_bin;		
				double ori = accurate_bin * (360.0f / (float)SIFT_ORI_HIST_BINS);
				if (abs(ori - 360.0f) < FLT_EPSILON)		//if the value is less than floating-point precision of type float
					ori = 0.0f;
				(*images)[ith].features[n].orientations.push_back(ori);
			}
		}
	}
}
void release_hist(float**** histPtr, int d, int n)
{
	if (histPtr != NULL){

		for (int i = 0; i < d; i++)
		{
			for (int j = 0; j < d; j++){
				delete[](*histPtr)[i][j];
			}
			delete[](*histPtr)[i];
		}
		delete[](*histPtr);
	}
}
void MySIFT::createDescriptors(int ith)
{
	int f_num = (*images)[ith].features.size();
	for (int i = 0; i < f_num; i++)		//trace each feature points
	{
		Feature f = (*images)[ith].features[i];		// ith feature of the ith images
		int octave = f.layer / s;
		int layer = f.layer % s;
		Point2f position = Point2f(f.position.x*2.0f / pow(2.0, octave), f.position.y*2.0f / pow(2.0, octave));
		float radius = f.size * 2.0f/(pow(2.0,octave)*2);

		Mat gaussianImg = (*images)[ith].octaves[octave].gaussianImgs[layer];
		//計算descriptor所需的圖像區域半徑 = (3*sigma*sqrt(2)*(d+1)+1)/2; sigma: 關鍵點所在組(octave)的組內尺度, d=4
		float hist_region_width = SIFT_DESCR_SCL_FCTR * radius;		//3*sigma
		int radius_of_descriptor = cvRound((hist_region_width*sqrt(2)*(SIFT_DESCR_HIST_ARRAY_WIDTH + 1) + 1) / 2); // mσ(d+1)/2*sqrt(2)

		for (int j = 0; j < f.orientations.size(); j++)
		{
			float ori = f.orientations[j];
			
			//convert feature to keypoint
			FeaturePoint keypoint;
			keypoint.position = f.position;
			keypoint.orientation = ori;
			keypoint.scale = f.scale;
			float*** histogram = createDescriptor(gaussianImg, f, position, hist_region_width, radius_of_descriptor, ori);
			hist_to_descriptor(histogram, &keypoint, j);
			(*images)[ith].keypoints.push_back(keypoint);
			release_hist(&histogram,SIFT_DESCR_HIST_ARRAY_WIDTH,SIFT_DESCR_HIST_BINS); //De-allocates memory held by a descriptor histogram
		}
	}
}
/*Computes the 2D array(4x4) of orientation histograms that form the feature descriptor.  Based on Section 6.1 of Lowe's paper.*/
float***  MySIFT::createDescriptor(Mat gaussianImg, Feature f, Point2f pos, float hist_region_width, int radius_of_descriptor, float ori)
{
	// Allocate 8 orientations x 4x4 histogram array
	float ***histogram = new float**[SIFT_DESCR_HIST_ARRAY_WIDTH];
	for (int i = 0; i<SIFT_DESCR_HIST_ARRAY_WIDTH; i++)
	{
		histogram[i] = new float*[SIFT_DESCR_HIST_ARRAY_WIDTH];
		for (int j = 0; j < SIFT_DESCR_HIST_ARRAY_WIDTH; j++){
			histogram[i][j] = new float[SIFT_DESCR_HIST_BINS];
		}
	}
	// initailize the 128 dimension vector
	for (int r = 0; r < SIFT_DESCR_HIST_ARRAY_WIDTH; r++)
		for (int c = 0; c < SIFT_DESCR_HIST_ARRAY_WIDTH; c++)
			for (int o = 0; o < SIFT_DESCR_HIST_BINS; o++)
				histogram[r][c][o] = 0.0;

	/*Rotate the coordinate to match the major orientation of the keypoint and calculate the new position of the keypoint*/
	float angle_to_rotate = 360.0f - ori;
	if (abs(angle_to_rotate-360) < FLT_EPSILON)
		angle_to_rotate = 0.0f;
	float rotation_sin = sin(angle_to_rotate*(PI / 180.0f));
	float rotation_cos = cos(angle_to_rotate*(PI / 180.0f));

	for (int p = -radius_of_descriptor; p <= radius_of_descriptor; p++)
	{
		int index_i = (int)(pos.y + p);
		if (index_i<=0)	continue;
		else if (index_i>=gaussianImg.rows-1)	break;

		for (int q = -radius_of_descriptor; q <= radius_of_descriptor; q++)
		{
			int index_j = (int)(pos.x + q);
			if (index_j<=0)	continue;
			else if (index_j>=gaussianImg.cols-1)	break;

			/* Calculate sample's histogram array coords rotated relative to the keypoint's orientation.
			Subtract 0.5f so samples that fall e.g. in the center of row 1 ( i.e. rotated_i = 1.5 ) have full weight placed in row 1 after interpolation.*/
			float rotated_j = (q*rotation_cos - p*rotation_sin) / hist_region_width;						//orientation-invariant
			float rotated_i = (q*rotation_sin + p *rotation_cos) / hist_region_width;
			float i_bin = rotated_i + (float)SIFT_DESCR_HIST_ARRAY_WIDTH / 2.0f - 0.5f;			//sub-bin row coordinate of entry
			float j_bin = rotated_j + (float)SIFT_DESCR_HIST_ARRAY_WIDTH / 2.0f - 0.5f;			//sub-bin column coordinate of entry

			if (i_bin > -1 && i_bin < SIFT_DESCR_HIST_ARRAY_WIDTH && j_bin > -1 && j_bin < SIFT_DESCR_HIST_ARRAY_WIDTH)
			{
				
				float dx = (gaussianImg.at<float>(index_i, index_j + 1) - gaussianImg.at<float>(index_i, index_j - 1)) / 2.0f;
				float dy = (gaussianImg.at<float>(index_i + 1, index_j) - gaussianImg.at<float>(index_i - 1, index_j)) / 2.0f;

				// calculate magnitude  & orientation
				float w = exp(-(rotated_i * rotated_i + rotated_j * rotated_j) / (2 * (0.5*SIFT_DESCR_HIST_ARRAY_WIDTH)*(0.5*SIFT_DESCR_HIST_ARRAY_WIDTH)));
				float weighted_magnitude = w*sqrt(dx*dx + dy*dy); // Lowe 建議子區域的梯度大小按sigma=0.5*width的Gaussian加權計算
				float orientation = fastAtan2(dy, dx);
				float orientation_bin = (orientation - angle_to_rotate) / (360.0f / (float)SIFT_DESCR_HIST_BINS);		//sub-bin orientation coordinate of entry (45 degrees per bin)

				/*
				Interpolates an entry into the array of orientation histograms that form the feature descriptor.
				Tiilinear interpolation is used to distribute the value of each gradient sample into adjacent histogram bins.
				=> The entry is distributed into up to 8 bins.
				Each entry into a bin is multiplied by a weight of 1 - d for each dimension,
				where d is the distance from the center value of the bin measured in bin units.
				*/
				int r0 = cvFloor(i_bin);
				int c0 = cvFloor(j_bin);
				int o0 = cvFloor(orientation_bin);
				//Make sure ob is in 0 - 8 dimension
				if (o0 < 0){ o0 += SIFT_DESCR_HIST_BINS; }
				else if (o0 >= SIFT_DESCR_HIST_BINS){ o0 -= SIFT_DESCR_HIST_BINS; }
				//calculate the distance from the center value of the bin measured in bin units.
				float d_r = i_bin - r0;
				float d_c = j_bin - c0;
				float d_o = orientation_bin - o0;

				// histogram update using tri-linear interpolation
				for (int r = 0; r <= 1; r++)
				{
					int index_r = r0 + r;
					if (index_r >= 0 && index_r < SIFT_DESCR_HIST_ARRAY_WIDTH)
					{	
						float v_r = weighted_magnitude*( (r==0) ? (1.0-d_r) : d_r );
						for (int c = 0; c <= 1; c++)
						{
							int index_c = c0 + c;
							if (index_c >= 0 && index_c < SIFT_DESCR_HIST_ARRAY_WIDTH)
							{
								float v_c = v_r*( (c==0) ? (1.0-d_c) : d_c );
								for (int o = 0; o <= 1; o++)
								{
									float v_o = v_c*( (o==0) ? (1.0-d_o) : d_o );
									int index_o = (o0 + o) % SIFT_DESCR_HIST_BINS;
									histogram[index_r][index_c][index_o] += v_o;
									
								} //end iteration of o
							}
						} // end iteration of c
					}
				} //end iteration of r
			}
		}
	}
	return histogram;
}
int normalize_descr(FeaturePoint* keypoint)
{
	float total_len = 0.0;
	for (int i = 0; i < (*keypoint).descriptor.size(); i++)
		total_len += (*keypoint).descriptor[i] * (*keypoint).descriptor[i];
	for (int i = 0; i < (*keypoint).descriptor.size(); i++)
		(*keypoint).descriptor[i] *= 1.0 / max(sqrt(total_len), FLT_EPSILON);
	return sqrt(total_len);
}
/*Converts the 2D array of orientation histograms into a feature's descriptor vector.*/
void MySIFT::hist_to_descriptor(float*** hist, FeaturePoint* keypoint, int jthOri)
{
	// copy the histogram to a feature descrioptor
	for (int r = 0; r < SIFT_DESCR_HIST_ARRAY_WIDTH; r++)
		for (int c = 0; c < SIFT_DESCR_HIST_ARRAY_WIDTH; c++)
			for (int o = 0; o < SIFT_DESCR_HIST_BINS; o++)
				(*keypoint).descriptor.push_back(hist[r][c][o]);

	/*The changes of nonlinear light and the camera saturation in certain directions gradient value is too large, and the impact on the direction is weak. 
	Therefore, we set the threshold value (normalized vector, generally take 0.2) truncated larger gradient value. 
	Then, we do a renormalization to improve identification of features.*/

	//normalize:  achieve invariant affine change and invariant non-linear change in illumination
	int threshold = normalize_descr(keypoint)*SIFT_DESCR_MAG_THR;
	for (int i = 0; i < (*keypoint).descriptor.size(); i++)
		if ((*keypoint).descriptor[i] > threshold)
			(*keypoint).descriptor[i] = threshold;		//cut-off
	//re-nomalize
	normalize_descr(keypoint);

}
vector<KeyPoint> MySIFT::convertToCV_Kpts(int ith)
{
	vector<KeyPoint> cv_kpts;
	for (int j = 0; j < (*images)[ith].features.size(); j++)
	{
		Feature f = (*images)[ith].features[j];
		Point2f position = f.position;
		float octave = (f.octave*s + f.layer)%s;		
		float size = f.size;						
		float response = f.response;
		for (int k = 0; k < f.orientations.size(); k++)
		{
			float angle = f.orientations[k];
			cv_kpts.push_back(KeyPoint(position, size, angle, response, octave));
		}
	}
	cout << "The  number of keypoints: " << cv_kpts.size() << endl;
	return cv_kpts;
}
Matrix<float> convertDescriptorsToMatrix(vector<FeaturePoint> keypoints)
{
	float* tempData = new float[keypoints.size() * 128];
	for (int i = 0; i < keypoints.size(); i++)
	{
		for (int j = 0; j < keypoints[i].descriptor.size(); j++)
		{
			tempData[i * 128 + j] = keypoints[i].descriptor[j];
		}
	}
	Matrix<float> data(tempData, keypoints.size(), 128);
	return data;
}
void MySIFT::matching()
{
	const int nn = 2;		//the number of the nearest neighbors to search for
	for (int i = 0; i < (*images).size(); i++)
	{
		size_t theImage = i;
		size_t nextImage = i + 1;
		if (theImage == (*images).size()-1)
			nextImage = 0;
		Matrix<float> trainDescriptors = convertDescriptorsToMatrix((*images)[nextImage].keypoints);
		Matrix<float> queryDescriptors = convertDescriptorsToMatrix((*images)[theImage].keypoints);
		Matrix<int> indices(new int[queryDescriptors.rows*nn], queryDescriptors.rows, nn);
		Matrix<float> distance(new float[queryDescriptors.rows*nn], queryDescriptors.rows, nn);
		
		// construct an randomized kd-tree index using 4 kd-trees
		Index<cvflann::L2_Simple<float> > kdtree(trainDescriptors, KDTreeIndexParams(1));
		kdtree.buildIndex();
		//  find the nearest neighbor in queryD for each point in trainD using 128 checks
		kdtree.knnSearch(queryDescriptors, indices, distance, nn, SearchParams());
		for (int j = 0; j < (*images)[theImage].keypoints.size(); j++)
		{
			if (sqrt(distance[j][0]) < sqrt(distance[j][1]) * SIFT_MATCH_DISTANCE_RATIO){
				(*images)[theImage].matches.push_back(Match(j, indices[j][0], sqrt(distance[j][0])));
			}
		}
		delete[] queryDescriptors.data;
		delete[] trainDescriptors.data;
		delete[] indices.data;
		delete[] distance.data;
	
	}
	
}
Mat MySIFT::convertToCV_Descriptor(int ith)
{
	Mat cv_descr = Mat::zeros((*images)[ith].keypoints.size(),128, CV_32FC1);
	for (int i = 0; i < (*images)[ith].keypoints.size(); i++)
		for (int j = 0; j < 128; j++)
			cv_descr.at<float>(i, j) = (*images)[ith].keypoints[i].descriptor[j];
	return cv_descr;
}
vector<DMatch> MySIFT::convertToCV_Matches(int ith)
{
	size_t nextImage = ith + 1;
	if (nextImage == (*images).size())
		nextImage = 0;
	vector<DMatch> cv_matches;
	for (int i = 0; i < (*images)[ith].matches.size(); i++){
		cv_matches.push_back(DMatch(
			(*images)[ith].matches[i].queryIndex,
			(*images)[ith].matches[i].trainIndex,
			(*images)[ith].matches[i].distance));
	}
	cout << "the number of matches between  " << ith << " and " << nextImage << " = " << (*images)[ith].matches.size() << endl;
	return cv_matches;
}