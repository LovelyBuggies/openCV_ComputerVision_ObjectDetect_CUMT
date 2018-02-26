// DynamicObjectDetection.cpp

#include "opencv.hpp"
#include "KF_ORP.h"
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <unistd.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat frame, gray, mask, mask1;
    vector<Target> targets;
    VideoCapture capture;
    capture.open(argv[2]);
    
    if (!capture.isOpened())
    {
        cout<<"No camera or video input!\n"<<endl;
        return -1;
    }
    
    KF_ORP KF_ORP;
    int count = 0;
    while (1)
    {
        capture >> frame;
        if (frame.empty())
            break;
        count++;
        cvtColor(frame, gray, CV_RGB2GRAY);
        
        if (count == 1)
        {
            KF_ORP.init(gray);
            KF_ORP.processFirstFrame(gray);
            cout<<"Using "<<TRAIN_FRAMES<<" frames to training GMM..."<<endl;
        }
        else if (count < TRAIN_FRAMES)
        {
            cout << "Train frame loading...\n";
            KF_ORP.trainGMM(gray);
        }
        else if (count == TRAIN_FRAMES)
        {
            cout << "Training GMM complete!" << endl;
            KF_ORP.getFitNum(gray);
            targets = KF_ORP.getTargets(frame, KF_ORP.getMask());
        }
        else
        {
            cout << count << "th frame : ";
            sleep(1);
            KF_ORP.testGMM(gray);
            mask = KF_ORP.getMask();
            imshow("mask", mask);
            vector<Target> temp = KF_ORP.getTargets(frame, mask);
            KF_ORP.KF(frame, targets, temp);
        }

        cout << endl;
        
        imshow("input", frame);
        
        if ( cvWaitKey(10) == 'q' )
            break;
    }
    
    return 0;
}
