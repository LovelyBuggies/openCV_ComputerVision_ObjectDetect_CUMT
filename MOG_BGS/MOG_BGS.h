#pragma once

#include <stdio.h>
#include <iostream>
#include "opencv.hpp"

using namespace cv;
using namespace std;


 #define GMM_MAX_COMPONT 6
 #define GMM_LEARN_ALPHA 0.05
 #define GMM_THRESHOD_SUMW 0.7
 #define TRAIN_FRAMES 20

class MOG_BGS
{
public:
	MOG_BGS(void);
	~MOG_BGS(void);

	void init(const Mat& _image);
	void processFirstFrame(const Mat& _image);
	void trainGMM(const Mat& _image);
	void getFitNum(const Mat& _image);
	void testGMM(const Mat& _image);
	Mat getMask(void){return m_mask;};
	Mat lock_target(Mat& _image,const Mat& m_mask);	
 
private:
	Mat m_weight[GMM_MAX_COMPONT];
	Mat m_mean[GMM_MAX_COMPONT];
	Mat m_sigma[GMM_MAX_COMPONT];

	Mat m_mask;
	Mat m_fit_num;

};
