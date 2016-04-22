//
//  ImageStitch.cpp
//  ImageStitching
//
//  Created by Zhi-Wei Yang on 4/22/16.
//  Copyright © 2016 Zhi-Wei Yang. All rights reserved.
//

#include "ImageStitch.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <opencv2/imgproc.hpp>

cv::Mat ImageStitch::CylindricalProjection(const cv::Mat & image)
{
    cv::Mat proj = cv::Mat(image.rows,image.cols,CV_8UC3);
    proj = cv::Scalar::all(0);
    double scale = focalLen;
    
    
//    for(int b = 0; b < image.rows; b++)
//    {
//        for(int a = 0; a < image.cols; a++)
//        {
//            double theta = atan((a-image.cols/2)/focalLen);
//            double h = (b-image.rows/2)/pow(pow((a-image.cols/2),2)+pow(focalLen,2),0.5);
//            int x = focalLen * theta+image.cols/2;
//            int y = focalLen * h+image.rows/2;
//            proj.at<cv::Vec3b>(y, x)[0] = image.at<cv::Vec3b>(b,a)[0];
//            proj.at<cv::Vec3b>(y, x)[1] = image.at<cv::Vec3b>(b,a)[1];
//            proj.at<cv::Vec3b>(y, x)[2] = image.at<cv::Vec3b>(b,a)[2];
//            
//        }
//    }
    
    for(int i=0; i<proj.rows; i+=1) {
        for(int j=0; j<proj.cols; j+=1) {
            double xp = j-proj.cols/2;
            double yp = i-proj.rows/2;
            
            double x = focalLen * tan(xp / scale);
            double y = sqrt(x*x + focalLen*focalLen) * yp / scale;
            
            int newi = y + image.rows/2;
            int newj = x + image.cols/2;
            if( newi<0 || newi>=image.rows ) continue;
            if( newj<0 || newj>=image.cols ) continue;
            
            proj.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(newi, newj);
        }
    }
    
    return proj;
}



void ImageStitch::CalculateFeatures()
{
    std::string path = "/Users/Allen/Documents/workspace/NTU104-2_Digital-Visual-Effects/Project 2 - image stitching/ImageStitching/";
    std::fstream fa(path + "fa.txt", std::ios::in);
    std::fstream fb(path + "fb.txt", std::ios::in);
    std::fstream ma(path + "ma.txt", std::ios::in);
    
    
    double x, y, a, b, i1, i2;
    while( fa >> x >> y >> a >> b ) {
        features[0].push_back( std::make_pair(x, y) );
    }
    while( fb >> x >> y >> a >> b ) {
        features[1].push_back( std::make_pair(x, y) );
    }
    while( ma >> i1 >> i2 ) {
        matches[0][1].push_back( std::make_pair(i1, i2));
        matches[1][0].push_back( std::make_pair(i2, i1));
    }
    
    // feature Cylindrical warping
    
    for(int i=0; i<imageNum; i+=1) {
        for(int j=0; j<features[i].size(); j+=1) {
            double xp = features[i][j].first - images[i].cols/2;
            double yp = features[i][j].second - images[i].rows/2;
            
            double theta = atan2(xp, focalLen);
            double h = yp / sqrt( xp*xp + focalLen*focalLen );
            double x = focalLen * theta + images[i].cols/2;
            double y = focalLen * h + images[i].rows/2;
            
            features[i][j] = std::make_pair(x, y);
        }
    }

    
    // test feature matching
    cv::Mat & test1 = images[0];
    cv::Mat & test2 = images[1];
    cv::Mat merge(test1.rows, test1.cols * 2, test1.type() );
    cv::Mat mask1(merge,cv::Range(0,test1.rows),cv::Range(0,test1.cols));
    cv::Mat mask2(merge,cv::Range(0,test2.rows),cv::Range(test1.cols,test1.cols+test2.cols));
    test1.copyTo(mask1);
    test2.copyTo(mask2);
    for(int i=0; i<matches[0][1].size()/5; i+=1) {
        const auto & ind = matches[0][1][i];
        cv::line(merge,
                 cv::Point(features[0][ind.first-1].first, features[0][ind.first-1].second),
                 cv::Point(features[1][ind.second-1].first + test1.cols, features[1][ind.second-1].second),
                 cv::Scalar(255,0,0));
    }
    
    cv::imshow("2 image", merge);
    cv::waitKey();
}


std::pair<double, double> ImageStitch::RANSAC(int ind1, int ind2, double thres, int k, int n)
{
    std::pair<double, double> ret;
    int maxInlierNum = 0;
    std::pair<double, double> vect(0, 0), currvect;
    for(int i=0; i<k; i+=1) {
        for(int j=0; j<n; j+=1) {
            int select = rand()%matches[ind1][ind2].size();
            const auto & ind = matches[ind1][ind2][select];
            vect.first += features[ind1][ind.first-1].first - features[ind2][ind.second-1].first;
            vect.second += features[ind1][ind.first-1].second - features[ind2][ind.second-1].second;
            vect.first /= n;
            vect.second /= n;
        }
        
        int currInlierNum = 0;
        double currDist = 0;
        for(int j=0; j<matches[ind1][ind2].size(); j+=1) {
            const auto & ind = matches[ind1][ind2][j];
            currvect.first = features[ind1][ind.first-1].first - features[ind2][ind.second-1].first;
            currvect.second = features[ind1][ind.first-1].second - features[ind2][ind.second-1].second;
            currDist += pow(vect.first-currvect.first, 2) + pow(vect.second-currvect.second, 2);
            if(currDist < thres)
                currInlierNum += 1;
        }
        if(currInlierNum > maxInlierNum) {
            maxInlierNum = currInlierNum;
            ret = vect;
        }
    }
    std::cout << maxInlierNum << std::endl;
    
    // test RANSAC
    cv::Mat & test1 = images[0];
    cv::Mat & test2 = images[1];
    cv::Mat merge(test1.rows * 2, test1.cols * 2, test1.type() );
    cv::Mat merge1(test1.rows * 2, test1.cols * 2, test1.type() );
    cv::Mat merge2(test1.rows * 2, test1.cols * 2, test1.type() );
    cv::Mat mask1(merge1,cv::Range(100,100+test1.rows),
                  cv::Range(0,test1.cols));
    cv::Mat mask2(merge2,cv::Range(100+ret.second, 100+test2.rows+ret.second),
                  cv::Range(ret.first, test2.cols+ret.first));
    test1.copyTo(mask1);
    test2.copyTo(mask2);
    
    cv::addWeighted( merge1, 0.5, merge2, 0.5, 0.0, merge);
    cv::imshow("merge", merge);
    cv::waitKey();

    return ret;
}