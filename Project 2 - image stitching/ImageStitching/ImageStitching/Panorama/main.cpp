#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cv.h>
#include<highgui.h>
#include<nonfree/nonfree.hpp>
#include <ctime>
#include "Image.h"
#include "MySIFT.h"
#include "ImageStitch.h"

using namespace std;
using namespace cv;

//void cv_match(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &good_matches);

int main(int argc, char **argv)
{
	
	// check the number of the arguments
	if (argc != 3) // the app name, the input folder name, the output folder name
	{
		cout << "Please give the the path of the input images and the path of the output images\n";
		system("PAUSE");
		return 1;
	}
	// read file and images
	string inputPath(argv[1]);
	string filesInfo = "info.txt";
	fstream fp(inputPath + filesInfo, fstream::in);
	if (!fp){
		cout << "Fail to read file info.txt.\n";
		system("PAUSE");
		return 1;
	}
	vector<Image> images;
	string imgInfo;
	int width=0, height=0;
	while (fp >> imgInfo) 
	{
		double focalLength;
		fp >> focalLength;
		Image img(inputPath + imgInfo, focalLength);
		if (width == 0 && height == 0){
			width = img.width;
			height = img.height;
		}
		images.push_back(img);
	}
	fp.close();
	
	/*use MySIFT to do SIFT feature detection, description, matching*/
	MySIFT sift(&images,height,width);
	//save the images with kpts
	for (int i = 0; i < images.size(); i++)
	{
		vector<KeyPoint> keypoints;
		keypoints = sift.convertToCV_Kpts(i);
		images[i].cv_kpts = keypoints;
		Mat outputImg;
		drawKeypoints(images[i].image, keypoints, outputImg, Scalar(255.0, 0.0, 0.0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		char outName[100];
		sprintf(outName, "%sSIFT_kpts%d.jpg", argv[2] , i);
		imwrite(outName, outputImg);
	}
	//save the images with matches
	for (int i = 0; i < images.size(); i++)
	{
		size_t theImage = i;
		size_t nextImage = i + 1;
		if (theImage == images.size() - 1)
			nextImage = 0;
		vector<DMatch> self_matches = sift.convertToCV_Matches(theImage);
		Mat  img_matches_self;
		drawMatches(images[theImage].image, images[theImage].cv_kpts, images[nextImage].image, images[nextImage].cv_kpts, self_matches, img_matches_self);
		char outName[100];
		sprintf(outName, "%sSIFT_matches%d.jpg", argv[2], theImage);
		imwrite(outName, img_matches_self);
	}
	/*Execute image stitching*/
	srand((unsigned int)time(NULL));
	ImageStitch stitch(argv[2],images);
	stitch.StartStitching(true, false);
	
	/*use opencv to  do SIFT feature detection, description, matching*/
	//save the images with kpts
	/*for (int i = 0; i < images.size(); i++){
		SIFT cv_sift = SIFT();
		
		vector<KeyPoint> cv_kpts;
		Mat cv_decr, img_kpts_cv;
		cv_sift.detect(images[i].image, cv_kpts);
		cv_sift.compute(images[i].image, cv_kpts, cv_decr);
		drawKeypoints(images[i].image, cv_kpts, img_kpts_cv, cv::Scalar(0.0, 0.0, 255.0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		char outName[100];
		sprintf(outName, "%scvSIFT_kpts%d.jpg", argv[2], i);
		imwrite(outName,img_kpts_cv);
	}
	//save the images with matches
	for (int i = 0; i < images.size(); i++)
	{
		size_t next = i + 1;
		if (next == images.size())
			next = 0;
		Mat img_matches_cv;
		vector<DMatch> cv_matches;
		Mat cv_desc0 = sift.convertToCV_Descriptor(i);
		Mat cv_desc1 = sift.convertToCV_Descriptor(next);
		cv_match(cv_desc0, cv_desc1, cv_matches);
		drawMatches(images[i].image, images[i].cv_kpts, images[next].image, images[next].cv_kpts, cv_matches, img_matches_cv);
		char outName[100];
		sprintf(outName, "%scvSIFT_matches%d.jpg", argv[2], i);
		imwrite(outName, img_matches_cv);
	}
	*/


	system("PAUSE");
	return 0;
}

/*void cv_match(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &good_matches)
{
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	float min_dist = 100.0f;
	for (int i = 0; i < descriptors1.rows; i++)
		if (matches[i].distance < min_dist)
			min_dist = matches[i].distance;


	for (int i = 0; i < descriptors1.rows; i++)
		if (matches[i].distance <= std::max(2.0f*min_dist, 0.02f))
			good_matches.push_back(matches[i]);
}*/