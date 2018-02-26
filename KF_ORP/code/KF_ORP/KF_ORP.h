#pragma once

#include <iostream>
#include "opencv.hpp"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <stdio.h>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

#define GMM_MAX_COMPONT 6
#define GMM_LEARN_ALPHA 0.05
#define GMM_THRESHOD_SUMW 0.7
#define TRAIN_FRAMES 20

const int winHeight = 600;
const int winWidth = 800;

const int xMovingSpeed = 0;     // target X-moving speed
const int yMovingSpeed = 0;     // target Y-moving speed

const int upperBound = 500;    // target upper bound
const int lowerBound = 100;    // target lower bound
const int left_bound = 100;     // target left bound
const int right_bound = 700;    // target right bound
const int area_ceil = 9000;     // maxium target area
const int area_floor = 500;     // minium target area
const int height_ceil = 600;    // maxium target height
const int height_floor = 20;    // minium target height
const int width_ceil = 100;     // maxium target width
const int width_floor = 10;     // minium target width

class Target {
public:
    
    Rect contours;
    Point center;
    bool operator== (const Target& other) {
        if (this->center == other.center) return true;
        return (this->contours & other.contours).area()? true : false;
    }
};


class KF_ORP
{
public:
	KF_ORP(void);
	~KF_ORP(void);

	void init(const Mat& _image);
	void processFirstFrame(const Mat& _image);
	void trainGMM(const Mat& _image);
	void getFitNum(const Mat& _image);
	void testGMM(const Mat& _image);
    Mat getMask(void);
    vector<Target> getTargets( Mat& _image, const Mat& m_mask );
    void KF(Mat& _image, vector<Target>& targets, const vector<Target>& m_targets);
 
private:
	Mat m_weight[GMM_MAX_COMPONT];
	Mat m_mean[GMM_MAX_COMPONT];
	Mat m_sigma[GMM_MAX_COMPONT];
	Mat m_mask;
	Mat m_fit_num;
    vector<Target> m_targets;
};
