#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


int main( int argc, char** argv)
{
    Mat src, dst;

    Mat kernel;
    Point anchor;
    double delta;
    int ddepth;
    int kernel_size;
    char* window_name = "filter2D Demo";

    int c;

    // load an image
    src = imread( argv[1] );
    if ( !src.data ) { return -1; }

    // create window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    // initialize arguments for the filter
    anchor = Point( -1, -1 );
    delta = 0;
    ddepth = -1;

    // loop - will filter the image withdifferent kernel sizes each 0.5 seconds
    int ind = 0;
    while ( true )
    {
        c = waitKey(500);
        if ( (char) c == 27) { break; }

        // update kernel size for a normalized box filter
        kernel_size = 3 + 2 * ( ind % 5 );
        kernel = Mat::ones( kernel_size, kernel_size, CV_32F ) / (float)(kernel_size * kernel_size);

        // apply filter
        filter2D( src, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT );
        imshow( window_name, dst );
        ind++;
    }

    return 0; 
}
