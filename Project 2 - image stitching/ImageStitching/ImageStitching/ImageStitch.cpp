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
#include <opencv2/imgproc/imgproc.hpp>

ImageStitch::ImageStitch(const std::string & p) : path(p)
{
	std::fstream para(path + parameter, std::ios::in);

	std::string str;
	while (para >> str) {
		images.push_back( cv::imread(path + str) );
		double FL;
		para >> FL;
		focalLens.push_back(FL);
	}

	features.resize( images.size() );
}

void ImageStitch::StartStitching()
{
	for (int i = 0; i < images.size(); i += 1) {
		images[i] = CylindricalProjection(i);
	}
	CalculateFeatures();

	auto shift = RANSAC(0, 1, 1e5, 200, 1);
	cv::Mat merge = MergeImage(&images[0], &images[1], shift);
}

cv::Mat ImageStitch::CylindricalProjection(int ind)
{
	const cv::Mat & image = images[ind];
	cv::Mat proj = cv::Mat(image.rows, image.cols, CV_8UC3);
	proj = cv::Scalar::all(0);
	double scale = focalLens[ind];

	for (int i = 0; i < proj.rows; i += 1) {
		for (int j = 0; j < proj.cols; j += 1) {
			double xp = j - proj.cols / 2;
			double yp = i - proj.rows / 2;

			double x = focalLens[ind] * tan(xp / scale);
			double y = sqrt(x*x + focalLens[ind]*focalLens[ind]) * yp / scale;

			int newi = y + image.rows / 2;
			int newj = x + image.cols / 2;
			if (newi < 0 || newi >= image.rows) continue;
			if (newj < 0 || newj >= image.cols) continue;

			proj.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(newi, newj);
		}
	}

	return proj;
}



void ImageStitch::CalculateFeatures()
{
	std::fstream fa(path + "fa.txt", std::ios::in);
	std::fstream fb(path + "fb.txt", std::ios::in);
	std::fstream ma(path + "ma.txt", std::ios::in);


	double x, y, a, b, i1, i2;
	while (fa >> x >> y >> a >> b) {
		features[0].push_back(std::make_pair(x, y));
	}
	while (fb >> x >> y >> a >> b) {
		features[1].push_back(std::make_pair(x, y));
	}
	while (ma >> i1 >> i2) {
		matches[0][1].push_back(std::make_pair(i1, i2));
		matches[1][0].push_back(std::make_pair(i2, i1));
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
	cv::Mat & test1 = images[0];
	cv::Mat & test2 = images[1];
	cv::Mat merge(test1.rows, test1.cols * 2, test1.type());
	cv::Mat mask1(merge, cv::Range(0, test1.rows), cv::Range(0, test1.cols));
	cv::Mat mask2(merge, cv::Range(0, test2.rows), cv::Range(test1.cols, test1.cols + test2.cols));
	test1.copyTo(mask1);
	test2.copyTo(mask2);
	for (int i = 0; i < matches[0][1].size() / 5; i += 1) {
		const auto & ind = matches[0][1][i];
		cv::line(merge,
				 cv::Point(features[0][ind.first - 1].first, features[0][ind.first - 1].second),
				 cv::Point(features[1][ind.second - 1].first + test1.cols, features[1][ind.second - 1].second),
				 cv::Scalar(255, 0, 0));
	}

	cv::imshow("2 image feature matching", merge);
	cv::waitKey();
	cv::destroyWindow("2 image feature matching");
}


std::pair<double, double> ImageStitch::RANSAC(int ind1, int ind2, double thres, int k, int n)
{
	std::pair<double, double> ret;
	int maxInlierNum = 0;
	std::pair<double, double> vect(0, 0), currvect;
	for (int i = 0; i < k; i += 1) {
		for (int j = 0; j < n; j += 1) {
			int select = rand() % matches[ind1][ind2].size();
			const auto & ind = matches[ind1][ind2][select];
			vect.first += features[ind1][ind.first - 1].first - features[ind2][ind.second - 1].first;
			vect.second += features[ind1][ind.first - 1].second - features[ind2][ind.second - 1].second;
			vect.first /= n;
			vect.second /= n;
		}

		int currInlierNum = 0;
		double currDist = 0;
		for (int j = 0; j < matches[ind1][ind2].size(); j += 1) {
			const auto & ind = matches[ind1][ind2][j];
			currvect.first = features[ind1][ind.first - 1].first - features[ind2][ind.second - 1].first;
			currvect.second = features[ind1][ind.first - 1].second - features[ind2][ind.second - 1].second;
			currDist += pow(vect.first - currvect.first, 2) + pow(vect.second - currvect.second, 2);
			if (currDist < thres)
				currInlierNum += 1;
		}
		if (currInlierNum > maxInlierNum) {
			maxInlierNum = currInlierNum;
			ret = vect;
		}
	}
	std::cout << maxInlierNum << std::endl;

	// test RANSAC
	// MergeImage(&images[0], &images[1], ret);
	
	return ret;
}

cv::Mat ImageStitch::MergeImage(cv::Mat * img1, cv::Mat * img2, std::pair<int, int> shift)
{
	
	int width, height;
	if (shift.first < 0) {
		std::swap(img1, img2);
		shift = std::make_pair(-shift.first, -shift.second);
	}
	width = std::max(img1->cols, img2->cols + shift.first);
	if (shift.second < 0) {
		height = std::max(img1->rows, img2->rows - shift.second);
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


	cv::imshow("merge", merge);
	cv::waitKey();
	cv::destroyWindow("merge");

	return merge;
}