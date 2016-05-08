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
	srand((unsigned int)time(NULL));
	std::string path = "/Users/Allen/Documents/workspace/NTU104-2_Digital-Visual-Effects/Project 2 - image stitching/ImageStitching/grail/";
	//std::string path = "../../";
    

	ImageStitch stitch(path);
	stitch.StartStitching();


	return 0;
}
