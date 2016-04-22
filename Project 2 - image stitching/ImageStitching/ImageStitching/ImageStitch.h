//
//  ImageStitch.h
//  ImageStitching
//
//  Created by Zhi-Wei Yang on 4/22/16.
//  Copyright © 2016 Zhi-Wei Yang. All rights reserved.
//

#ifndef ImageStitch_h
#define ImageStitch_h

#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>

class ImageStitch
{
public:
    ImageStitch(int n, const std::string & p) : imageNum(n), path(p)
    {
        features.resize(imageNum);
        images.resize(imageNum);
    }
    cv::Mat CylindricalProjection(const cv::Mat & image);
    void CalculateFeatures();
    std::pair<double, double> RANSAC(int ind1, int ind2, double thres, int k = 50, int n = 3);
    
private:
public:
    static const int MAX_IMAGE_SIZE = 10;
    int imageNum;
    std::string path;
    
    cv::Mat test1;
    cv::Mat test2;
    double focalLen = 740;
    
    std::vector<double> focalLens;
    std::vector< cv::Mat > images;
    
    std::vector< std::vector< std::pair<double, double> > > features;
    std::vector< std::pair<int, int> > matches[MAX_IMAGE_SIZE][MAX_IMAGE_SIZE];
};




#endif /* ImageStitch_h */
