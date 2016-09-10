#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat src, src_gray;
int maxCorners = 10;
int maxTrackbar = 25;

RNG rng(12345);
char* source_window = "Image";

void goodFeaturesToTrack_Demo( int, void* );

int main( int argc, char** argv )
{
    src = imread( argv[1], 1 );
    cvtColor( src, src_gray, CV_BGR2GRAY );

    namedWindow( source_window, CV_WINDOW_AUTOSIZE );

    createTrackbar( "Max corners: ", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);

    imshow( source_window, src );

    goodFeaturesToTrack_Demo( 0, 0 );

    waitKey(0);
    return 0;
}


void goodFeaturesToTrack_Demo( int, void* )
{
    if ( maxCorners < 1 ) { maxCorners = 1; }

    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    Mat copy;
    copy = src.clone();

    goodFeaturesToTrack(
        src_gray,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        Mat(),
        blockSize,
        useHarrisDetector,
        k
    );

    // draw corners detected
    cout << "** Number of corneres detected: " << corners.size() << endl;
    int r = 4;
    for ( int i = 0; i < corners.size(); i++ )
    {
        circle( copy, corners[i], r, Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
    }

    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window, copy );

    Size winSize = Size( 5, 5 );
    Size zeroZone = Size( -1, -1 );
    TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

    // calculate the refined corner locations
    cornerSubPix( src_gray, corners, winSize, zeroZone, criteria );

    for (int i = 0; i < corners.size(); i++ )
    {
        cout << "-- Refined Corner [" << i << "] (" << corners[i].x << "," << corners[i].y << ")" << endl;
    }
}
