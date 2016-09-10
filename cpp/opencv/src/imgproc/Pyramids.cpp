#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// global variables
Mat src, dst, tmp;

char* window_name = "Pyramids Demo";


int main( int argc, char** argv )
{
    // general instructions
    printf("\n Zoom In-Out demo \n");
    printf("------------------- \n");
    printf(" * [u] -> Zoom in   \n");
    printf(" * [d] -> Zoom out  \n");
    printf(" * [ESC] -> Close program \n \n");

    // test image
    src = imread("../images/baboon.jpg");
    if ( !src.data )
    {
        printf( " No data! -- Exiting the program \n" );
        return -1;
    }

    tmp = src;
    dst = tmp;

    // create window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    imshow( window_name, dst );

    // loop
    while( true )
    {
        int c;
        c = waitKey(10);

        if ( (char) c == 27)
        {
            break;
        }
        if ( (char) c == 'u' )
        {
            pyrUp( tmp, dst, Size( tmp.cols * 2, tmp.rows * 2 ) );
            printf( "** Zoom In: Image x2 \n" );
        }
        else if ( (char) c == 'd' )
        {
            pyrDown( tmp, dst, Size( tmp.cols / 2, tmp.rows / 2));
            printf( "** Zoom Out: Image / 2 \n" );
        }

        imshow( window_name, dst );
        tmp = dst;
    }
    return 0;
}
