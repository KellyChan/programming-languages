#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

//global variables
int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
char* window_name = "Threshold Demo";
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

// function headers
void Threshold_Demo( int, void* );


int main( int argc, char** argv )
{
    // load an image
    src = imread( argv[1], 1 );

    // convert the image to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );

    // create a window to display result
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    // create trackbar to choose type of threshold
    createTrackbar( trackbar_type,
                    window_name,
                    &threshold_type,
                    max_type,
                    Threshold_Demo );

    createTrackbar( trackbar_value,
                    window_name,
                    &threshold_type,
                    max_type,
                    Threshold_Demo );

    // call the function to initialize
    Threshold_Demo( 0, 0 );

    // wait until user finishes program
    while( true)
    {
        int c;
        c = waitKey( 20 );
        if ((char) c == 27) { break; }
    }
}


void Threshold_Demo ( int, void* )
{
    /*
        0: Binary
        1: Binary Inverted
        2: Threshold Truncated
        3: Threashold to Zero
        4: Threshold to Zero Inverted
    */

    threshold( src_gray, dst, threshold_value, max_BINARY_value, threshold_type );
    imshow( window_name, dst );
}
