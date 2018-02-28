#include "MOG_BGS.h"

using namespace cv;

MOG_BGS::MOG_BGS(void)
{

}

MOG_BGS::~MOG_BGS(void)
{

}

void MOG_BGS::init(const Mat& _image)
{
    // for each Gaussian compont
    for(int i = 0; i < GMM_MAX_COMPONT; i++)
    {
        // initial the weight of each element in matrix as 0
        m_weight[i] = Mat::zeros(_image.size(), CV_32FC1);
        // initial the mean of each element in matrix as 0
        m_mean[i] = Mat::zeros(_image.size(), CV_8UC1);
        // initial the sigma of each element in maxtrix as 0
        m_sigma[i] = Mat::zeros(_image.size(), CV_32FC1);
    }
    // initial m_mask as black backgrond
    m_mask = Mat::zeros(_image.size(),CV_8UC1);
    // initial m_fit as black backgrond
    m_fit_num = Mat::ones(_image.size(),CV_8UC1);
}

void MOG_BGS::processFirstFrame(const Mat& _image)
{
    m_weight[0].setTo(1.0);
    _image.copyTo(m_mean[0]);
    m_sigma[0].setTo(15.0);

	for(int i = 1; i < GMM_MAX_COMPONT; i++)
    {
        m_weight[i].setTo(0.0);
        m_mean[i].setTo(0);
        m_sigma[i].setTo(15.0);
    }
}

void MOG_BGS::trainGMM(const Mat& _image)
{
	for(int i = 0; i < _image.rows; i++)
	{
		for(int j = 0; j < _image.cols; j++)
		{
             int num_fit = 0;
             for(int k = 0 ; k < GMM_MAX_COMPONT; k++)
             {
				 int delm = abs(_image.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j));
				 long dist = delm * delm;
                 // EM Algorithm
                 if( dist < 3.0 * m_sigma[k].at<float>(i, j)) 
                 {
                     /****update the weight****/
                     m_weight[k].at<float>(i, j) += GMM_LEARN_ALPHA * (1 - m_weight[k].at<float>(i, j));
 
                     /****update the mean****/
                     m_mean[k].at<uchar>(i, j) += (GMM_LEARN_ALPHA / m_weight[k].at<float>(i, j)) * delm;
 
                     /****update the variance****/
                     m_sigma[k].at<float>(i, j) += (GMM_LEARN_ALPHA / m_weight[k].at<float>(i, j)) * (dist - m_sigma[k].at<float>(i, j));
                 }
                 else
				 {
                     m_weight[k].at<float>(i, j) +=
                     GMM_LEARN_ALPHA * (0 - m_weight[k].at<float>(i, j));
                     num_fit++;
                 }        
             }
 			 /**************************** Update parameters End ******************************************/


			 /*********************** Sort Gaussian component by 'weight / sigma' Start ****************************/
             for(int kk = 0; kk < GMM_MAX_COMPONT; kk++)
             {
                 for(int rr=kk; rr< GMM_MAX_COMPONT; rr++)
                 {
                     if(m_weight[rr].at<float>(i, j)/m_sigma[rr].at<float>(i, j) > m_weight[kk].at<float>(i, j)/m_sigma[kk].at<float>(i, j))
                     {
                         float temp_weight = m_weight[rr].at<float>(i, j);
                         m_weight[rr].at<float>(i, j) = m_weight[kk].at<float>(i, j);
                         m_weight[kk].at<float>(i, j) = temp_weight;
 
                         uchar temp_mean = m_mean[rr].at<uchar>(i, j);
                         m_mean[rr].at<uchar>(i, j) = m_mean[kk].at<uchar>(i, j);
                         m_mean[kk].at<uchar>(i, j) = temp_mean;
 
                         float temp_sigma = m_sigma[rr].at<float>(i, j);
                         m_sigma[rr].at<float>(i, j) = m_sigma[kk].at<float>(i, j);
                         m_sigma[kk].at<float>(i, j) = temp_sigma;
                     }
                 }
             }
			 /*********************** Sort Gaussian model by 'weight / sigma' End ****************************/
 

			 /*********************** Create new Gaussian component Start ****************************/
             if(num_fit == GMM_MAX_COMPONT && m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) == 0)
             {
                for(int k = 0 ; k < GMM_MAX_COMPONT; k++)
                 {
                     if(0 != m_weight[k].at<float>(i, j))
					 {
						 m_weight[k].at<float>(i, j) *= (1 - GMM_LEARN_ALPHA);
                      }
                     else
                     {
                         m_weight[k].at<float>(i, j) = GMM_LEARN_ALPHA;
                         m_mean[k].at<uchar>(i, j) = _image.at<uchar>(i, j);
                         m_sigma[k].at<float>(i, j) = 15.0;
                         break;
                     }
                  }
             }
             else if(num_fit == GMM_MAX_COMPONT &&
                     m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) != 0)
             {
                 m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) = GMM_LEARN_ALPHA;
                 m_mean[GMM_MAX_COMPONT - 1].at<uchar>(i, j) = _image.at<uchar>(i, j);
                 m_sigma[GMM_MAX_COMPONT - 1].at<float>(i, j) = 15.0;
             }
			 /*********************** Create new Gaussian component End ****************************/
         }
	}
}

void MOG_BGS::getFitNum(const Mat& _image)
{
	for(int i = 0; i < _image.rows; i++)
	{
		for(int j = 0; j < _image.cols; j++)
		{
			float sum_w = 0.0;
			for(uchar k = 0; k < GMM_MAX_COMPONT; k++)
			{
				sum_w += m_weight[k].at<float>(i, j);
				if(sum_w >= GMM_THRESHOD_SUMW)
                {
                     m_fit_num.at<uchar>(i, j) = k + 1;
                     break;
                }
			}
		}
	}
}

void MOG_BGS::testGMM(const Mat& _image)
{
    // for each element in the matrix
	for(int i = 0; i < _image.rows; i++)
	{
		for(int j = 0; j < _image.cols; j++)
		{
			int k = 0;
			for( ; k < m_fit_num.at<uchar>(i, j); k++)
			{
				if(abs(_image.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j))
                   < (uchar)( 2.5 * m_sigma[k].at<float>(i, j)))
				{
					m_mask.at<uchar>(i, j) = 0;
					break;
				}
			}
			if(k == m_fit_num.at<uchar>(i, j))
			{
				m_mask.at<uchar>(i, j) = 255;
			}
		}
	}
}

Mat MOG_BGS::lock_target( Mat& _image, const Mat& m_mask)
{
	vector<vector<Point>> contours;
	findContours(m_mask,
		contours,             // a vector of contours
		CV_RETR_EXTERNAL,     // retrieve the external contours
		CV_CHAIN_APPROX_NONE);// retrieve all pixels of each contours

	// Eliminate too short or too long contours.
	int cmin= 50;  // minimum contour length
	int cmax= 1000; // maximum contour length
	vector<vector<Point>>::const_iterator itc= contours.begin();
	while (itc!=contours.end()) 
	{
		if (itc->size() < cmin || itc->size() > cmax)
			itc= contours.erase(itc);
		else 
			++itc;
	}

    // Bounding the contours with rectangle.
	for (int i = 0; i < contours.size(); i++)
	{
		Rect r0= boundingRect(Mat(contours[i]));
	    rectangle(_image,r0,Scalar(0,0,255), 2);
	}
    cout << "There are " << contours.size() << " targets."<< endl;
	return _image;
}
