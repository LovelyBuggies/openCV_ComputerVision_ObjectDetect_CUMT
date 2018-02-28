// DynamicObjectDetection.cpp

#include "opencv.hpp"
#include "MOG_BGS.h"
#include <iostream>
#include <cstdio>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat frame, gray, mask, mask1;
    VideoCapture capture;
    capture.open(argv[2]);
    
    if (!capture.isOpened())
    {
        cout<<"No camera or video input!\n"<<endl;
        return -1;
    }
    
    MOG_BGS Mog_Bgs;
    int count = 0;
    
    while (1)
    {
        count++;
        capture >> frame;
        if (frame.empty())
            break;
        
        cvtColor(frame, gray, CV_RGB2GRAY);
        
        if (count == 1)
        {
            Mog_Bgs.init(gray);
            Mog_Bgs.processFirstFrame(gray);
            cout<<" Using "<<TRAIN_FRAMES<<" frames to training GMM..."<<endl;
        }
        else if (count < TRAIN_FRAMES)
        {
            Mog_Bgs.trainGMM(gray);
        }
        else if (count == TRAIN_FRAMES)
        {
            Mog_Bgs.getFitNum(gray);
            cout<<" Training GMM complete!"<<endl;
        }
        else
        {
            Mog_Bgs.testGMM(gray);
            mask = Mog_Bgs.getMask();
            morphologyEx(mask, mask, MORPH_OPEN, Mat());
            erode(mask, mask, Mat(5, 5, CV_8UC1), Point(-1, -1));
            dilate(mask, mask, Mat(8, 8, CV_8UC1), Point(-1, -1));
            Canny(mask, mask1, 50, 150, 3);
            imshow("mask", mask);
            frame = Mog_Bgs.lock_target(frame,mask);
        }
    
        cout<<count<<endl;
        
        imshow("input", frame);
        
        if ( cvWaitKey(10) == 'q' )
            break;
    }
    
    return 0;
}

