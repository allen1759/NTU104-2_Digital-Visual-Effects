#ifndef Image_h
#define Image_h

#include <cv.h>
#include <highgui.h>
using namespace cv;

struct Octave{
	//double scale;
	vector<Mat> gaussianImgs;
	vector<Mat> DogImgs;
};
typedef struct Octave Oactave;

struct Feature
{
	Point originPos;						// the origin position of the feature
	Point2f position;					// the accurate position in original size image
	int octave;								// the octave where the feature was found
	int layer;									// the layer where the feature was found
	float scale;								// the absolute sigma of the local extrema
	float size;									//	 diameter of the meaningful keypoint neighborhood
	float response;						// the response by which the most strong keypoints have been selected
	vector<float> orientations;
};
typedef struct Feature Feature;

struct Match{
	int queryIndex;
	int trainIndex;
	float distance;
	Match(int q,int t, float d){
		queryIndex = q;
		trainIndex = t;
		distance = d;
	}
};
typedef struct Match Match;

struct FeaturePoint{
	Point2f position;
	float scale;
	float orientation;
	vector<float> descriptor;			//128 dimensions (4x4x8)
};
typedef struct FeaturePoint FeaturePoint;

class Image
{
public:
	/*variables*/
	string image_name;						//the name of the image
	Mat image;										//the original image
	Mat gray_image;							//the gray image
	int width;
	int height;
	double focalLen;								//the focal length of the image
	vector <Octave> octaves;
	vector<Feature> features;					//the feature list of the image
	vector<FeaturePoint> keypoints;		//the final keypoints
	vector<Match> matches;
	vector<Match> refined_matches;

	vector<KeyPoint> cv_kpts;				//the kpts with opencv format


	/*functions*/
	Image(string name, double f){
		this->image_name = name;
		this->focalLen = f;
		image = imread(image_name, CV_LOAD_IMAGE_COLOR);									//read image
		this->width = image.cols;
		this->height = image.rows;
		//get gray_scale image
		if( image.channels() == 3 || image.channels() == 4 )
			cvtColor(image, gray_image, CV_BGR2GRAY);
		else
			image.copyTo(gray_image);
	}
	
};
#endif