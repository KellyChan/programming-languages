#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


// Global variable
Mat src, erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

/* Function Headers */
void Erosion( int, void* );
void Dilation( int, void* );

/* function main */
int main(int argc, char** argv)
{
    // load an image
    src = imread( argv[1] );

    if ( !src.data ) { return -1; }

    // create windows
    namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE );
    namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE );
    // namedWindow( "Dilation Demo", src.cols, 0 );

    // create Erosion tracker
    createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse",
                    "Erosion Demo",
                    &erosion_elem,
                    max_elem,
                    Erosion );

    createTrackbar( "Element: \n 0: Rect \n 1: Cross \n 2: Ellipse",
                    "Dilation Demo",
                    &dilation_elem,
                    max_elem,
                    Dilation);

    createTrackbar( "Kernel size: \n 2n + 1",
                    "Dilation Demo",
                    &dilation_size,
                    max_kernel_size,
                    Dilation);

    // default start
    Erosion( 0, 0 );
    Dilation( 0, 0 );

    waitKey(0);
    return 0;
}


void Erosion( int, void* )
{
    int erosion_type;
    if ( erosion_elem == 0 ) { erosion_type = MORPH_RECT; }
    else if ( erosion_elem == 1 ) { erosion_type = MORPH_CROSS; }
    else if ( erosion_elem == 2 ) { erosion_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement( erosion_type, 
                                         Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
                                         Point( erosion_size, erosion_size ));

    // apply the erosion operation
    erode( src, erosion_dst, element );
    imshow( "Erosion Demo", erosion_dst );
    
}


void Dilation( int, void* )
{
    int dilation_type;
    if ( dilation_elem == 0 ) { dilation_type = MORPH_RECT; }
    else if ( dilation_elem == 1 ) { dilation_type = MORPH_CROSS; }
    else if ( dilation_elem == 2 ) { dilation_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement( dilation_type,
                                         Size( 2 * dilation_size + 1, 2 * dilation_size + 1 ),
                                         Point( dilation_size, dilation_size ));

    // apply the dilation operation
    dilate( src, dilation_dst, element );
    imshow( "Dilation Demo", dilation_dst );
}
