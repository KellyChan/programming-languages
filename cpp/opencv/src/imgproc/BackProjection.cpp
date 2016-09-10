#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

Mat src;
Mat hsv;
Mat hue;

int bins = 25;

// function headers
void Hist_and_Backproj( int, void* );


int main ( int argc, char** argv )
{
    // read the image
    src = imread( argv[1], 1 );
    // transform it to HSV
    cvtColor( src, hsv, CV_BGR2HSV );

    // use only the Hue value
    hue.create( hsv.size(), hsv.depth() );
    int ch[] = {0, 0};
    mixChannels( &hsv, 1, &hue, 1, ch, 1);

    // create trackbar to enter the number of bins
    char* window_image = "Source image";
    namedWindow( window_image, CV_WINDOW_AUTOSIZE );
    createTrackbar("* Hue bins: ", window_image, &bins, 180, Hist_and_Backproj );
    Hist_and_Backproj(0, 0);

    // show the image
    imshow( window_image, src );

    waitKey(0);
    return 0;
}


void Hist_and_Backproj ( int, void* )
{
    MatND hist;
    int histSize = MAX( bins, 2 );
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    // get the histogram and normalize it
    calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
    normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

    // get backprojection
    MatND backproj;
    calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true);

    // draw the backproj
    imshow("BackProj", backproj);

    // draw the histogram
    int w = 400;
    int h = 400;
    int bin_w = cvRound( (double) w / histSize);
    Mat histImg = Mat::zeros( w, h, CV_8UC3 );

    for ( int i = 0; i < bins; i++ )
    {
        rectangle( histImg,
                   Point( i * bin_w, h),
                   Point( (i-1) * bin_w, h - cvRound( hist.at<float>(i) * h / 255.0 )),
                   Scalar(0, 0, 255), -1);
    }
   
    imshow("Histogram", histImg);
}

