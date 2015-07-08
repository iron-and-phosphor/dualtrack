//By downloading, copying, installing or using the software you agree to this license.
//If you do not agree to this license, do not download, install,
//copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//                       (3-clause BSD License)
//
//Redistribution and use in source and binary forms, with or without modification,
//are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//This software is provided by the copyright holders and contributors "as is" and
//any express or implied warranties, including, but not limited to, the implied
//warranties of merchantability and fitness for a particular purpose are disclaimed.
//In no event shall copyright holders or contributors be liable for any direct,
//indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
//loss of use, data, or profits; or business interruption) however caused
//and on any theory of liability, whether in contract, strict liability,
//or tort (including negligence or otherwise) arising in any way out of
//the use of this software, even if advised of the possibility of such damage.


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;
Mat image2;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;

int tx1,ty1,tx2,ty2;


Mat frame2, hsv2, hue2, mask2, hist, histimg2 = Mat::zeros(200, 320, CV_8UC3), backproj2;
Rect trackWindow;
Rect trackWindow2;
int hsize = 16;
float hranges[] = {0,180};
const float* phranges = hranges;

static void onMouse( int event, int x, int y, int, void* ){
    if( selectObject ){
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event ){
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}

void second_camera(VideoCapture &cap){
    cap >> frame2;
    if( frame2.empty()) return;
    frame2.copyTo(image2);
    cvtColor(image2, hsv2, COLOR_BGR2HSV);
    if( trackObject ){
        int _vmin = vmin, _vmax = vmax;
        inRange(hsv2, Scalar(0, smin, MIN(_vmin,_vmax)),
                Scalar(180, 256, MAX(_vmin, _vmax)), mask2);
        int ch[] = {0, 0};
        hue2.create(hsv2.size(), hsv2.depth());
        mixChannels(&hsv2, 1, &hue2, 1, ch, 1);
       
        calcBackProject(&hue2, 1, 0, hist, backproj2, &phranges);
        backproj2 &= mask2;
        
        RotatedRect trackBox;
        if(trackWindow.width >= 1 || trackWindow.height >= 1){
            trackBox = CamShift(backproj2, trackWindow,
                            TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
        }

        if( trackWindow2.area() <= 1 ){
            int cols = backproj2.cols, rows = backproj2.rows, r = (MIN(cols, rows) + 5)/6;
            trackWindow2 = Rect(trackWindow2.x - r, trackWindow2.y - r,
                               trackWindow2.x + r, trackWindow2.y + r) &
                          Rect(0, 0, cols, rows);
        }
        if( backprojMode )
            cvtColor( backproj2, image2, COLOR_GRAY2BGR );
        ellipse( image2, trackBox, Scalar(0,0,255), 3, CV_AA );
        // hier !!!!!!!! 
        tx2 = trackBox.center.x;
        ty2 = trackBox.center.y;
        cout << "x2: " << trackBox.center.x << " Y2: " << trackBox.center.y << endl;
    }
    imshow( "Cam2", image2 );
}

const char* keys ={
    "{1|  | 0 | camera number}"
};

int main( int argc, const char** argv ){
    int camNum1 = 0;
    int camNum2 = 1;

    string input = "";
    cout << "cam numbers start at zero. Every camera that is conected adds to the the cam number." << endl;
    cout << "So camera one is 0. camera two is 1. Camera three is 2 and so onward" << endl;
    cout << "type the cam number of the cam you want to use for the x axis: " << endl;
    cout << "cam number: ";
    cin >> input;
    camNum1 = stoi(input);
    cout << "type the cam number of the cam you want to use for the y axis: " << endl;
    cout << "cam number: ";
    cin >> input;
    camNum2 = stoi(input);

    cout << "To start tracking select in the cam1 window the object you want to track." << endl 
         <<"The second camera wil start streaming after you have selected a object" << endl; 

    VideoCapture cap;
    VideoCapture cap2;
    
    CommandLineParser parser(argc, argv, keys);
    //int camNum = parser.get<int>("1");

    cap.open( camNum1);
    cap2.open(camNum2);

    if( !cap.isOpened() ){
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        parser.printParams();
        return -1;
    }

    namedWindow( "virtual top Cam", 0 );
    namedWindow( "Cam1", 0 );
    namedWindow( "Cam2", 0 );
    setMouseCallback( "Cam1", onMouse, 0 );
    createTrackbar( "Vmin", "Cam1", &vmin, 256, 0 );
    createTrackbar( "Vmax", "Cam1", &vmax, 256, 0 );
    createTrackbar( "Smin", "Cam1", &smin, 256, 0 );

    Mat frame, hsv, hue, mask,  histimg = Mat::zeros(600, 600, CV_8UC3), backproj;

    bool paused = false;

    for(;;){
        if( !paused ){
            cap >> frame;
            if( frame.empty() )
                break;
        }

        frame.copyTo(image);

        if( !paused ){
            cvtColor(image, hsv, COLOR_BGR2HSV);

            if( trackObject ){
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if( trackObject < 0 ){
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);

                    trackWindow = selection;
                    trackWindow2 = selection;
                    trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, CV_HSV2BGR);
                    cout << "test" << endl;
                }
                histimg = Scalar::all(0);
                rectangle( histimg, Point(10 + tx1, 10 + tx2),
                            Point(tx1,tx2),
                            Scalar(0, 0, 255 ));

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
                
                RotatedRect trackBox;
                if(trackWindow.width >= 1 || trackWindow.height >= 1){
                    trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
                }

                if( trackWindow.area() <= 1 ){
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if( backprojMode )
                    cvtColor( backproj, image, COLOR_GRAY2BGR );
                ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
                // hier !!!!!!!! 
                tx1 = trackBox.center.x;
                ty1 = trackBox.center.y;
                cout << "x: " << trackBox.center.x << " Y: " << trackBox.center.y << endl;
                second_camera(cap2);
            }
        }
        else if( trackObject < 0 )
            paused = false;

        if( selectObject && selection.width > 0 && selection.height > 0 ){
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }

        imshow( "Cam1", image );
        imshow( "virtual top Cam", histimg );

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        
    }

    return 0;
}
