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
#include <queue>
#include <opencv2/imgproc/imgproc.hpp>

ImageStitch::ImageStitch(const std::string & p, vector<Image> imgs) : path(p)
{
	for (int i = 0; i < imgs.size(); i++)
	{
		images.push_back(imgs[i].image);
		focalLens.push_back(imgs[i].focalLen);
	}
	features.resize(images.size());
	//std::cout << "text1\n";
	for (int im = 0; im < images.size(); im += 1)
	{
		for (int i = 0; i < imgs[im].keypoints.size(); i += 1)
		{
			FeaturePoint f = imgs[im].keypoints[i];
			features[im].push_back(std::make_pair(f.position.x, f.position.y));
		}
	}
	//std::cout << "text2\n";
	for (int im = 0; im + 1 < images.size(); im += 1)
	{
		for (int i = 0; i < imgs[im].matches.size(); i++)
		{
			Match m = imgs[im].matches[i];
			matches[im][im + 1].push_back(std::make_pair(m.queryIndex, m.trainIndex));
			matches[im + 1][im].push_back(std::make_pair(m.trainIndex, m.queryIndex));
		}
	}
	//std::cout << "text3\n";
}

void ImageStitch::StartStitching(bool crop, bool end2end)
{
	std::cout << "Execute Cylindrical Projection.\n";
	for (int i = 0; i < images.size(); i += 1) {
		images[i] = CylindricalProjection(i);
	}
    if (end2end) {
        CalculateFeatures_End2End();
    }
    else {
    	CalculateFeatures();
    }

    std::vector< std::pair<int, int> > shifts;
    cv::Mat merge;
    
    if (end2end) {
        for (int i = 0; i < images.size(); i += 1) {
            shifts.push_back( RANSAC(i, (i+1)%images.size(), 50.0, 500, 1) );
        }
        int cntYshift = 0;
        for (int i = 0; i < images.size(); i += 1) {
            cntYshift += shifts[i].second;
        }
        int adjust = cntYshift / (int)( images.size() );
        for (int i = 0; i < images.size(); i += 1) {
            shifts[i].second -= adjust;
        }
        
        if (crop) {
            images[0].copyTo(merge);
            for (int i = 1; i <= images.size(); i += 1) {
                //std::cout << shifts[i-1].first << " " << shifts[i-1].second << std::endl;
                MergeImage_Crop(&merge, &images[i%images.size()], shifts[i-1]).copyTo(merge);
				if (i >= shifts.size())
					continue;
                shifts[i].first += shifts[i-1].first;
                shifts[i].second += shifts[i-1].second;
            }
        }
        else {
            images[0].copyTo(merge);
            for (int i = 1; i <= images.size(); i += 1) {
                //std::cout << shifts[i-1].first << " " << shifts[i-1].second << std::endl;
                MergeImage(&merge, &images[i%images.size()], shifts[i-1]).copyTo(merge);
				if (i >= shifts.size())
					continue;
                shifts[i].first += shifts[i-1].first;
                shifts[i].second += shifts[i-1].second;
            }
        }
        
    }
    
    else {
		std::cout << "Execute image matching using RANSAC.\n";
        for (int i = 0; i+1 < images.size(); i += 1) {
            shifts.push_back( RANSAC(i, i+1, 50.0, 500, 1) );
        }
		std::cout << "Merge Images.\n";
        if (crop) {
            images[0].copyTo(merge);
            for (int i = 1; i < images.size(); i += 1) {
                //std::cout << shifts[i-1].first << " " << shifts[i-1].second << std::endl;
                MergeImage_Crop(&merge, &images[i], shifts[i-1]).copyTo(merge);
				if (i >= shifts.size())
					continue;
                shifts[i].first += shifts[i-1].first;
                shifts[i].second += shifts[i-1].second;
            }
        }
        else {
            images[0].copyTo(merge);
            for (int i = 1; i < images.size(); i += 1) {
                //std::cout << shifts[i-1].first << " " << shifts[i-1].second << std::endl;
                MergeImage(&merge, &images[i], shifts[i-1]).copyTo(merge);
				if (i >= shifts.size())
					continue;
                shifts[i].first += shifts[i-1].first;
                shifts[i].second += shifts[i-1].second;
            }
        }
    }

	//cv::imshow("Result Panorama", merge);
    cv::imwrite(path + "mypano.jpg", merge);
}

cv::Mat ImageStitch::CylindricalProjection(int ind)
{
	const cv::Mat & image = images[ind];
	cv::Mat proj = cv::Mat(image.rows, image.cols, CV_8UC3);
	proj = cv::Scalar::all(0);
//	double scale = focalLens[ind];
    
    for (int i = 0; i < image.rows; i += 1) {
        for (int j = 0; j < image.cols; j += 1) {
            double xp = j - image.cols / 2;
            double yp = i - image.rows / 2;
            
			double theta = atan2(xp, focalLens[ind]);
			double h = yp / sqrt(xp*xp + focalLens[ind]*focalLens[ind]);
			double x = focalLens[ind] * theta + image.cols / 2;
			double y = focalLens[ind] * h + image.rows / 2;
            
			proj.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(i, j);
        }
    }

//	for (int i = 0; i < proj.rows; i += 1) {
//		for (int j = 0; j < proj.cols; j += 1) {
//			double xp = j - proj.cols / 2;
//			double yp = i - proj.rows / 2;
//
//			double x = focalLens[ind] * tan(xp / scale);
//			double y = sqrt(x*x + focalLens[ind]*focalLens[ind]) * yp / scale;
//
//			int newi = y + image.rows / 2;
//			int newj = x + image.cols / 2;
//			if (newi < 0 || newi >= image.rows) continue;
//			if (newj < 0 || newj >= image.cols) continue;
//
//			proj.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(newi, newj);
//		}
//	}
    
    //cv::imwrite(path + std::to_string(ind) + ".jpg", proj);

	return proj;
}



void ImageStitch::CalculateFeatures()
{
  
	// feature Cylindrical warping

	for (int i = 0; i < images.size(); i += 1) {
		for (int j = 0; j < features[i].size(); j += 1) {
			double xp = features[i][j].first - images[i].cols / 2;
			double yp = features[i][j].second - images[i].rows / 2;

			double theta = atan2(xp, focalLens[i]);
			double h = yp / sqrt(xp*xp + focalLens[i]*focalLens[i]);
			double x = focalLens[i] * theta + images[i].cols / 2;
			double y = focalLens[i] * h + images[i].rows / 2;

			features[i][j] = std::make_pair(x, y);
		}
	}

	/*
	// test feature matching
	cv::Mat & test1 = images[0];
	cv::Mat & test2 = images[1];
	cv::Mat merge(test1.rows, test1.cols * 2, test1.type());
	cv::Mat mask1(merge, cv::Range(0, test1.rows), cv::Range(0, test1.cols));
	cv::Mat mask2(merge, cv::Range(0, test2.rows), cv::Range(test1.cols, test1.cols + test2.cols));
	test1.copyTo(mask1);
	test2.copyTo(mask2);
	for (int i = 0; i < matches[0][1].size() / 1; i += 1) {
		const auto & ind = matches[0][1][i];
		cv::line(merge,
				 cv::Point(features[0][ind.first - 1].first, features[0][ind.first - 1].second),
				 cv::Point(features[1][ind.second - 1].first + test1.cols, features[1][ind.second - 1].second),
				 cv::Scalar(255, 0, 0));
	}

	cv::imshow("2 image feature matching", merge);
	cv::waitKey();
	cv::destroyWindow("2 image feature matching");
	*/
}

void ImageStitch::CalculateFeatures_End2End()
{
    std::fstream fa, fb, ma;
    
    for(int im = 0; im < images.size(); im += 1) {
        fa.open(path + std::to_string(im) + "fa.txt");
        if( !fa.is_open() )
            std::cout << path + std::to_string(im) + "fa.txt not open" << std::endl;
        fb.open(path + std::to_string(im) + "fb.txt");
        if( !fb.is_open() )
            std::cout << path + std::to_string(im) + "fb.txt not open" << std::endl;
        ma.open(path + std::to_string(im) + "ma.txt");
        if( !ma.is_open() )
            std::cout << path + std::to_string(im) + "ma.txt not open" << std::endl;
        
    	double x, y, a, b, i1, i2;
    	while (fa >> x >> y >> a >> b) {
    		features[im].push_back(std::make_pair(x, y));
    	}
    	while (fb >> x >> y >> a >> b && im+1 < features.size() ) {
            features[im+1].push_back(std::make_pair(x, y));
    	}
    	while (ma >> i1 >> i2) {
    		matches[im][(im+1) % images.size()].push_back(std::make_pair(i1, i2));
    		matches[(im+1) % images.size()][im].push_back(std::make_pair(i2, i1));
    	}
        
        fa.close();
        fb.close();
        ma.close();
    }


	// feature Cylindrical warping

	for (int i = 0; i < images.size(); i += 1) {
		for (int j = 0; j < features[i].size(); j += 1) {
			double xp = features[i][j].first - images[i].cols / 2;
			double yp = features[i][j].second - images[i].rows / 2;

			double theta = atan2(xp, focalLens[i]);
			double h = yp / sqrt(xp*xp + focalLens[i]*focalLens[i]);
			double x = focalLens[i] * theta + images[i].cols / 2;
			double y = focalLens[i] * h + images[i].rows / 2;

			features[i][j] = std::make_pair(x, y);
		}
	}


	// test feature matching
	cv::Mat & test1 = images[6];
	cv::Mat & test2 = images[7];
	cv::Mat merge(test1.rows, test1.cols * 2, test1.type());
	cv::Mat mask1(merge, cv::Range(0, test1.rows), cv::Range(0, test1.cols));
	cv::Mat mask2(merge, cv::Range(0, test2.rows), cv::Range(test1.cols, test1.cols + test2.cols));
	test1.copyTo(mask1);
	test2.copyTo(mask2);
	for (int i = 0; i < matches[6][7].size() / 1; i += 1) {
		const auto & ind = matches[6][7][i];
		cv::line(merge,
				 cv::Point(features[6][ind.first].first, features[6][ind.first].second),
				 cv::Point(features[7][ind.second].first + test1.cols, features[7][ind.second].second),
				 cv::Scalar(255, 0, 0));
	}

	cv::imshow("2 image feature matching - end to end", merge);
	cv::waitKey();
	cv::destroyWindow("2 image feature matching - end to end");
}


std::pair<double, double> ImageStitch::RANSAC(int ind1, int ind2, double thres, int k, int n)
{
	std::pair<double, double> ret;
	int maxInlierNum = 0;
	std::pair<double, double> vect(0, 0), currvect;
    
	for (int i = 0; i < k; i += 1) {
        vect.first = vect.second = 0;
		for (int j = 0; j < n; j += 1) {
			int select = rand() % matches[ind1][ind2].size();
			const auto & ind = matches[ind1][ind2][select];
			vect.first += features[ind1][ind.first].first - features[ind2][ind.second].first;
			vect.second += features[ind1][ind.first].second - features[ind2][ind.second].second;
			vect.first /= n;
			vect.second /= n;
		}

		int currInlierNum = 0;
		double currDist = 0;
		for (int j = 0; j < matches[ind1][ind2].size(); j += 1) {
            currDist = 0;
			const auto & ind = matches[ind1][ind2][j];
			currvect.first = features[ind1][ind.first].first - features[ind2][ind.second].first;
			currvect.second = features[ind1][ind.first].second - features[ind2][ind.second].second;
			currDist += pow(vect.first - currvect.first, 2) + pow(vect.second - currvect.second, 2);
			if (currDist < thres)
				currInlierNum += 1;
		}
		if (currInlierNum > maxInlierNum) {
			maxInlierNum = currInlierNum;
			ret = vect;
		}
	}
    
	std::cout << maxInlierNum << "/" << matches[ind1][ind2].size() << std::endl;
    
    
	// test RANSAC
    //std::pair<int, int> tmp = std::make_pair((int)ret.first, (int)ret.second);
    //cv::Mat tmpImage = MergeImage(&images[ind1], &images[ind2], tmp);
    //if (ind1 == 6) {
    //    cv::imwrite(path + "tmp.jpg", tmpImage);
    //}
	
	return ret;
}

cv::Mat ImageStitch::MergeImage(cv::Mat * img1, cv::Mat * img2, std::pair<int, int> & shift)
{
	int width, height;
	if (shift.first < 0) {
		std::swap(img1, img2);
		shift = std::make_pair(-shift.first, -shift.second);
	}
	width = std::max(img1->cols, img2->cols + shift.first);
	if (shift.second < 0) {
		height = std::max(img1->rows - shift.second, img2->rows);
	}
	else {
		height = std::max(img1->rows, img2->rows + shift.second);
	}

	cv::Mat merge(height, width, img1->type());
	merge = cv::Scalar::all(0);
	if (shift.second < 0) {
        cv::Mat mask(merge, cv::Range(-shift.second, -shift.second+img1->rows),
                            cv::Range(0, img1->cols));
        img1->copyTo(mask);
	}
	else {
		cv::Mat mask(merge, cv::Range(0, img1->rows),
						    cv::Range(0, img1->cols));
		img1->copyTo(mask);
	}
    
    if (shift.second < 0) shift.second = 0;

	double overlap = img1->cols + img2->cols - width;
	for (int i = 0; i < img2->cols; i += 1) {
		double weight = 1.0;
		if (overlap > i && overlap!=0) {
			weight = i / overlap;
		}
		for (int j = 0; j < img2->rows; j += 1) {
			cv::Vec3f p1 = img2->at<cv::Vec3b>(j, i);
			cv::Vec3f p2 = merge.at<cv::Vec3b>(j + shift.second, i + shift.first);
			if (p1.val[0] + p1.val[1] + p1.val[2] <= 3)
				continue;
			else if (weight >= 1 || p2.val[0] + p2.val[1] + p2.val[2] <= 3) {
				merge.at<cv::Vec3b>(j + shift.second, i + shift.first) = p1;
				continue;
			}
			else {
				merge.at<cv::Vec3b>(j + shift.second, i + shift.first) = p1 * weight +
																		 p2 * (1 - weight);
			}
		}
	}


	//cv::imshow("merge", merge);
	//cv::waitKey();
	//cv::destroyWindow("merge");

	return merge;
}

cv::Mat ImageStitch::MergeImage_Crop(cv::Mat * img1, cv::Mat * img2, std::pair<int, int> & shift)
{
	int width, height;
	if (shift.first < 0) {
		std::swap(img1, img2);
		shift = std::make_pair(-shift.first, -shift.second);
	}
	width = std::max(img1->cols, img2->cols + shift.first);
	if (shift.second < 0) {
		height = std::max(img1->rows - shift.second, img2->rows);
	}
	else {
		height = std::max(img1->rows, img2->rows + shift.second);
	}

	cv::Mat merge(height, width, img1->type());
	merge = cv::Scalar::all(0);
	if (shift.second < 0) {
        cv::Mat mask(merge, cv::Range(-shift.second, -shift.second+img1->rows),
                            cv::Range(0, img1->cols));
        img1->copyTo(mask);
	}
	else {
		cv::Mat mask(merge, cv::Range(0, img1->rows),
						    cv::Range(0, img1->cols));
		img1->copyTo(mask);
	}
    
    if (shift.second < 0) shift.second = 0;

	double overlap = img1->cols + img2->cols - width;
	for (int i = 0; i < img2->cols; i += 1) {
		double weight = 1.0;
		if (overlap > i && overlap!=0) {
			weight = i / overlap;
		}
		for (int j = 0; j < img2->rows; j += 1) {
			cv::Vec3f p1 = img2->at<cv::Vec3b>(j, i);
			cv::Vec3f p2 = merge.at<cv::Vec3b>(j + shift.second, i + shift.first);
			if (p1.val[0] + p1.val[1] + p1.val[2] <= 3)
				continue;
			else if (weight >= 1 || p2.val[0] + p2.val[1] + p2.val[2] <= 3) {
				merge.at<cv::Vec3b>(j + shift.second, i + shift.first) = p1;
				continue;
			}
			else {
				merge.at<cv::Vec3b>(j + shift.second, i + shift.first) = p1 * weight +
																		 p2 * (1 - weight);
			}
		}
	}
    
    
    int left = 0, right = merge.cols-1, mid = merge.rows/2;
    for (int i = 0; i < merge.cols; i += 1) {
        int val = 0;
        for (int s = -1; s <= 1; s += 1) {
            for (int v = 0; v < 3; v += 1)
                val += merge.at<cv::Vec3b>(mid + s, i).val[v];
        }
        
        if (val <= 3) {
            left = std::max( left, i );
        }
        else {
            left = std::max( left, i );
            break;
        }
    }
    for (int i = merge.cols-1; i >= 0; i -= 1) {
        int val = 0;
        for (int s = -1; s <= 1; s += 1) {
            for (int v = 0; v < 3; v += 1)
                val += merge.at<cv::Vec3b>(mid + s, i).val[v];
        }
        
        if (val <= 3) {
            right = std::min( right, i );
        }
        else {
            right = std::min( right, i );
            break;
        }
    }
    
    // compute top and bot position
    int maxtop = 0, minbot = merge.rows-1;
    int searchRange = 10;
    for (int i = left; i < left + searchRange; i += 1) {
        for (int j = 0; j < merge.rows; j += 1) {
            cv::Vec3b pixel = merge.at<cv::Vec3b>(j, i);
            if (pixel.val[0] + pixel.val[1] + pixel.val[2] <= 3) {
                maxtop = std::max(maxtop, j);
            }
            else {
                break;
            }
        }
        
        for (int j = merge.rows-1; j >= 0; j -= 1) {
            cv::Vec3b pixel = merge.at<cv::Vec3b>(j, i);
            if (pixel.val[0] + pixel.val[1] + pixel.val[2] <= 3) {
                minbot = std::min(minbot, j);
            }
            else {
                break;
            }
        }
    }
    
    
    for (int i = right; i > right - searchRange; i -= 1) {
        for (int j = 0; j < merge.rows; j += 1) {
            cv::Vec3f pixel = merge.at<cv::Vec3b>(j, i);
            if (pixel.val[0] + pixel.val[1] + pixel.val[2] <= 3) {
                maxtop = std::max(maxtop, j);
            }
            else {
                break;
            }
        }
        
        for (int j = merge.rows-1; j >= 0; j -= 1) {
            cv::Vec3f pixel = merge.at<cv::Vec3b>(j, i);
            if (pixel.val[0] + pixel.val[1] + pixel.val[2] <= 3) {
                minbot = std::min(minbot, j);
            }
            else {
                break;
            }
        }
    }
    
    cv::Mat mergeCrop = merge( cv::Rect(left, maxtop, right-left+1, minbot-maxtop+1) );
    shift.first -= left;
    shift.second -= maxtop;

	//cv::imshow("merge crop", mergeCrop);
	//cv::waitKey();
	//cv::destroyWindow("merge crop");

	return mergeCrop;
}