#include <stdio.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


Mat img;
Mat templ;
Mat result;
char* image_window = "Source Image";
char* result_window = "Result Window";

int match_method;
int max_Trackbar = 5;

void MatchingMethod( int, void* );


int main ( int argc, char** argv )
{
    // load the image and template
    img = imread( argv[1], 1 );
    templ = imread( argv[2], 1 );

    // create windows
    namedWindow( image_window, CV_WINDOW_AUTOSIZE );
    namedWindow( result_window, CV_WINDOW_AUTOSIZE );

    // create trackbar
    char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );

    MatchingMethod( 0, 0 );

    waitKey(0);
    return 0;
}


void MatchingMethod( int, void* )
{
    Mat img_display;
    img.copyTo( img_display );

    // create the result matrix
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    result.create( result_rows, result_cols, CV_32FC1 );

    // matching and normalize
    matchTemplate( img, templ, result, match_method );
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

    // localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    if ( match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    {
        matchLoc = minLoc;
    }
    else
    {
        matchLoc = maxLoc;
    }
    
    rectangle( img_display, matchLoc,
                            Point( matchLoc.x + templ.cols, matchLoc.y + templ.rows ),
                            Scalar::all(0),
                            2, 8, 0);

    rectangle( result, matchLoc,
                       Point( matchLoc.x + templ.cols, matchLoc.y + templ.rows),
                       Scalar::all(0),
                       2, 8, 0);

    imshow( image_window, img_display );
    imshow( result_window, result );
}
