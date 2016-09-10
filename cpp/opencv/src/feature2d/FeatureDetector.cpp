#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void readme();

int main( int argc, char** argv )
{
    if ( argc != 3 )
    {
        readme();
        return -1;
    }

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

    if ( !img_1.data || !img_2.data )
    {
        std::cout << "--(!) Error reading image " << std::endl;
        return -1;
    }

    int minHessian = 400;
    
    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    Mat img_keypoints_1;
    Mat img_keypoints_2;

    drawKeyPoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeyPoints( img_2, keypoints_2, img_keypoints_2, SCalar::all(-1), DrawMatchesFlags::DEFAULT );

    imshow( "Keypoints 1", img_keypoints_1 );
    imshow( "Keypoints 2", img_keypoints_2 );

    waitKey(0);
    return 0;
}


void readme()
{
    std::cout << "Usage: ./FeatureDetector <image1> <image2>" << std::endl;
}
