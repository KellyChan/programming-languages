#include <opencv2/highgui/highgui.hpp>

using namespace cv;


const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

Mat src1;
Mat src2;
Mat dst;


void on_trackbar( int, void* )
{
    alpha = (double) alpha_slider / alpha_slider_max;
    beta = (1.0 - alpha);

    addWeighted(src1, alpha, src2, beta, 0.0, dst);

    imshow( "Linear Blend", dst);
}


int main( int argc, char** argv)
{
    src1 = imread("../images/baboon.jpg");
    src2 = imread("../images/lion_king.jpg");

    if ( !src1.data ) { printf("Error loading src1 \n"); return -1; }
    if ( !src2.data ) { printf("Error loading src2 \n"); return -1; }

    alpha_slider = 0;
    namedWindow("Linear Blend", 1);

    char TrackbarName[50];
    sprintf( TrackbarName, "Alpha x %d", alpha_slider_max);
    createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );

    on_trackbar( alpha_slider, 0 );

    waitKey(0);
    return 0;
}
