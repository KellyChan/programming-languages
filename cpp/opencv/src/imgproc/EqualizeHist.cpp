#include <stdio.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


int main ( int argc, char** argv )
{
    Mat src, dst;

    char* source_window = "Source image";
    char* equalized_window = "Equalized Image";

    // load image
    src = imread( argv[1], 1 );
    if ( !src.data )
    {
        cout << "Usage: ./Histogram_Demo <image>" << endl;
        return -1;
    }

    // convert to grayscale
    cvtColor( src, src, CV_BGR2GRAY );

    // apply histogram equalization
    equalizeHist( src, dst );

    // display results
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    namedWindow( equalized_window, CV_WINDOW_AUTOSIZE );

    imshow( source_window, src );
    imshow( equalized_window, dst );

    waitKey(0);
    return 0;
}
