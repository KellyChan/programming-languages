#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


// Global variables
Mat src, dst;

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;

char* window_name = "Morphology Transformations Demo";

void Morphology_Operations( int, void* );

int main( int argc, char** argv )
{
    src = imread( argv[1] );

    if ( !src.data ) { return -1; }

    // create window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    // create trackbar to select morphology operation
    createTrackbar("Operator: \n 0: Opening: - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat",
                   window_name,
                   &morph_operator,
                   max_operator,
                   Morphology_Operations);

    // create trackbar to select kernel type
    createTrackbar( "Element: \n 0: Rect - 1: Cross - 2: Ellipse",
                    window_name,
                    &morph_elem,
                    max_elem,
                    Morphology_Operations );

    createTrackbar( "Kernel size: \n 2n + 1",
                    window_name,
                    &morph_size,
                    max_kernel_size,
                    Morphology_Operations );

    // default start
    Morphology_Operations(0, 0);

    waitKey(0);
    return 0;
}


// Morphology_Operations

void Morphology_Operations( int, void* )
{
    // since MORPH_X: 2, 3, 4, 5, and 6
    int operation = morph_operator + 2;

    Mat element = getStructuringElement( morph_elem,
                                         Size( 2 * morph_size + 1, 2 * morph_size + 1),
                                         Point( morph_size, morph_size ) );

    // apply the specified morphology operation
    morphologyEx( src, dst, operation, element );
    imshow( window_name, dst );
}
