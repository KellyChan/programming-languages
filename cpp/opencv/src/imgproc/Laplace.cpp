#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;

int main( int argc, char** argv)
{
    Mat src, src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    char* window_name = "Laplace Demo";

    int c;

    // load an image
    src = imread( argv[1] );
    if ( !src.data ) { return -1; }

    // remove noise by blurring with a Gaussian filter
    GaussianBlur( src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
    
    // Covert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );

    // create window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    // apply Laplace function
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    // apply laplace function
    Mat abs_dst;
    Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( dst, abs_dst );

    // show what you got
    imshow( window_name, abs_dst );

    waitKey(0);
    return 0;
}
