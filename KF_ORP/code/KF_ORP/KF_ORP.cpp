#include "KF_ORP.h"

using namespace cv;

KF_ORP::KF_ORP(void)
{

}

KF_ORP::~KF_ORP(void)
{

}

void KF_ORP::init(const Mat& _image)
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

void KF_ORP::processFirstFrame(const Mat& _image)
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

void KF_ORP::trainGMM(const Mat& _image)
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
            if(num_fit == GMM_MAX_COMPONT) {
                if (m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) == 0) {
                    for(int k = 0 ; k < GMM_MAX_COMPONT; k++) {
                        if(0 != m_weight[k].at<float>(i, j)) {
                            m_weight[k].at<float>(i, j) *= (1 - GMM_LEARN_ALPHA);
                        }
                        else {
                            m_weight[k].at<float>(i, j) = GMM_LEARN_ALPHA;
                            m_mean[k].at<uchar>(i, j) = _image.at<uchar>(i, j);
                            m_sigma[k].at<float>(i, j) = 15.0;
                         break;
                        }
                    }
                }
                else if (m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) != 0) {
                    m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) = GMM_LEARN_ALPHA;
                    m_mean[GMM_MAX_COMPONT - 1].at<uchar>(i, j) = _image.at<uchar>(i, j);
                    m_sigma[GMM_MAX_COMPONT - 1].at<float>(i, j) = 15.0;
                }
            }
            /*********************** Create new Gaussian component End ****************************/
         }
	}
}

void KF_ORP::getFitNum(const Mat& _image)
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

void KF_ORP::testGMM(const Mat& _image)
{
    // for each element in the matrix
	for(int i = 0; i < _image.rows; i++)
	{
		for(int j = 0; j < _image.cols; j++)
		{
			int k = 0;
			for( ; k < m_fit_num.at<uchar>(i, j); k++)
			{
				if(abs(_image.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j)) < (uchar)( 2.5 * m_sigma[k].at<float>(i, j)))
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

Mat KF_ORP::getMask(void) {
    morphologyEx(m_mask, m_mask, MORPH_OPEN, Mat());
    erode(m_mask, m_mask, Mat(5, 5, CV_8UC1), Point(-1, -1));
    dilate(m_mask, m_mask, Mat(8, 8, CV_8UC1), Point(-1, -1));
    //Canny(m_mask, m_mask, 50, 150, 3);
    return m_mask;
}

vector<Target> KF_ORP::getTargets( Mat& _image, const Mat& m_mask)
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
    
    m_targets = vector<Target>(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        Rect rec = boundingRect(Mat(contours[i]));
        m_targets[i].contours = rec;
        m_targets[i].center.x = rec.x + cvRound(rec.width/2);
        m_targets[i].center.y = rec.y + cvRound(rec.height/2);
    }
    

    // Unique m_targets.
    vector<Target>::iterator i;
    vector<Target>::iterator j;
    for (i = m_targets.begin(); i != m_targets.end(); i++)
        for (j = i + 1; j != m_targets.end(); j++) {
            if (*i == *j) {
                m_targets.erase(j);
                j--;
            }
        }
    
    // Trim window's turbulance.
    vector<Target>::iterator it= m_targets.begin();
    while (it != m_targets.end())
    {
        if (it->center.x < left_bound
                || it->center.x > right_bound
                || it->center.y < lowerBound
                || it->center.y > upperBound
                || it->contours.area() < area_floor
                || it->contours.area() > area_ceil
                || it->contours.height < height_floor
                || it->contours.height > height_ceil
                || it->contours.width < width_floor
                || it->contours.width > width_ceil)
            it = m_targets.erase(it);
        else
            ++it;
    }
    
    return m_targets;
}


void KF_ORP::KF(Mat& _image, vector<Target>& targets, const vector<Target>& m_targets)
{
    int targets_num = (int)m_targets.size();
    cout << "there are " << targets_num << " targets.\n"<< endl;
    
    // Special circumstances assume that the targets exist.
    if (m_targets.size() > targets.size())
    {
        targets = m_targets;
    }
    if (m_targets.size() == 0)  return;
    
    /*********************** Locate Targets' Centers using Kalman Filtering Start ****************************/
    
    const int stateNum = 4;                 // (x, y, dx, dy)
    const int measureNum = 2;               // (x, y)
    CvKalman* kalman = cvCreateKalman( stateNum, measureNum, 0 );
    CvMat* measurement = cvCreateMat( measureNum, 1, CV_32FC1 );
    
    // Transition matrix and noise_cov.
    float A[stateNum][stateNum] = {
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1
    };
    memcpy(kalman->transition_matrix->data.fl,A,sizeof(A));
    cvSetIdentity(kalman->measurement_matrix,cvRealScalar(1) );
    cvSetIdentity(kalman->process_noise_cov,cvRealScalar(1e-5));
    cvSetIdentity(kalman->measurement_noise_cov,cvRealScalar(1e-1));
    cvSetIdentity(kalman->error_cov_post,cvRealScalar(1));
    
    for (int i = 0; i < min(m_targets.size(), targets.size()); i++) {
        
        // Draw and text the initial targets.
        cout << "The " << i + 1 << "th target's initial center is ";
        cout << targets[i].center.x << ", ";
        cout << targets[i].center.y  << ".\n";
        
        // Draw and text the measure targets.
        rectangle(_image, m_targets[i].contours, Scalar(0,0,255), 2);
        circle( _image, m_targets[i].center, 1, Scalar(0,0,255), 2);
        cout << "The " << i + 1 << "th target's measure center is ";
        cout << m_targets[i].center.x << ", ";
        cout << m_targets[i].center.y  << ".\n";
        
        // State_post matrix.
        kalman->state_post->data.fl[0] = targets[i].center.x;
        kalman->state_post->data.fl[1] = targets[i].center.y;
        // assume targets are dynamic
        kalman->state_post->data.fl[2] = xMovingSpeed;
        kalman->state_post->data.fl[3] = yMovingSpeed;
        
        // Draw and text the predict targets.
        const CvMat* prediction=cvKalmanPredict(kalman,0);
        CvPoint predict_pt = CvPoint((int)prediction->data.fl[0], (int)prediction->data.fl[1]);
        int wid = targets[i].contours.width;
        int hei = targets[i].contours.height;
        Rect predict_rec = Rect(predict_pt.x - wid/2, predict_pt.y - hei/2, wid, hei);
        rectangle(_image, predict_rec, Scalar(0,255,255), 2);
        circle( _image, predict_pt, 1, Scalar(0,255,255), 2);
        cout << "The " << i + 1 << "th target's predict center is ";
        cout << (int)prediction->data.fl[0] << ", ";
        cout << (int)prediction->data.fl[1] << ".\n";
    
        //update measurement
        measurement->data.fl[0] = m_targets[i].center.x;
        measurement->data.fl[1] = m_targets[i].center.y;
    
        //update
        cvKalmanCorrect( kalman, measurement );
    
        // Draw and text the state-post targets.
        targets[i].center = cvPoint((int)kalman->state_post->data.fl[0], (int)kalman->state_post->data.fl[1]);
        targets[i].contours = m_targets[i].contours;         // approximation
        CvPoint state_post_pt = CvPoint(targets[i].center.x, targets[i].center.y);
        circle( _image, state_post_pt, 1, Scalar(0,255,0), 2);
        cout << "After correction, the center is ";
        cout << kalman->state_post->data.fl[0] << ", ";
        cout << kalman->state_post->data.fl[1] << ".\n\n";
    }
    
    /*********************** Locate Targets' Centers using Kalman Filtering End ****************************/
    
    return;
}
