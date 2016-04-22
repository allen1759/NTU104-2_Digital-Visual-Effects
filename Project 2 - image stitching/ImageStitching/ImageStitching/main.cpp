//
//  main.cpp
//  ImageStitching
//
//  Created by Zhi-Wei Yang on 4/22/16.
//  Copyright © 2016 Zhi-Wei Yang. All rights reserved.
//

#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include "ImageStitch.h"

int main(int argc, const char * argv[])
{
    // insert code here...
    srand(time(NULL));
    std::string path = "/Users/Allen/Documents/workspace/NTU104-2_Digital-Visual-Effects/Project 2 - image stitching/ImageStitching/";
    
    ImageStitch stitch(2, path);
    stitch.images[0] = cv::imread(path + "1.JPG");
    stitch.images[1] = cv::imread(path + "2.JPG");
    stitch.images[0] = stitch.CylindricalProjection(stitch.images[0]);
    stitch.images[1] = stitch.CylindricalProjection(stitch.images[1]);
    stitch.CalculateFeatures();
    stitch.RANSAC(0, 1, 1e6, 100, 1);
    
    
    return 0;
}
