#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;

// global variables
Mat src, src_gray;
Mat dst, detected_edges;


int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";


void CannyThreshold(int, void*)
{
    // reduce noise with a kernel 3*3
    blur( src_gray, detected_edges, Size(3, 3));

    // canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size );

    // using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    src.copyTo( dst, detected_edges );
    imshow( window_name, dst );
}


int main ( int argc, char** argv )
{
    // load an image
    src = imread( argv[1] );

    if ( !src.data ) { return -1; }

    // create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );

    // convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );

    // create a window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    // create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold: ",
                    window_name,
                    &lowThreshold,
                    max_lowThreshold,
                    CannyThreshold);

    // show the image
    CannyThreshold(0, 0);

    // wait until user exit program by pressing a key
    waitKey(0);
    return 0;
}
