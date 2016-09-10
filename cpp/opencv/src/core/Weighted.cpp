#include <iostream>

// #include <opencv2/opencv.hpp>
// #include <opencv2/cv.h>
// #include <opencv2/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


int main(int argc, char** argv)
{
    double alpha = 0.5;
    double beta;
    double input;

    Mat src1, src2, dst;

    // ask the user enter alpha
    std::cout << "Simple Linear Blender" << std::endl;
    std::cout << "---------------------" << std::endl;
    std::cout << "* Enter alpha [0-1]: ";
    std::cin >> input;

    // use the alpha provided by the user if it is between 0 and 1
    if (input >= 0.0 && input <= 1.0)
    {
        alpha = input;
    }

    // read image (same size, same type)
    src1 = imread("../images/baboon.jpg");
    // src2 = imread("../images/lion_king.jpg");
    src2 = imread("../images/baboon.jpg");

    if (!src1.data)
    {
        printf("Error loading src1 \n");
        return -1;
    }

    if (!src2.data)
    {
        printf("Error loading src2 \n");
        return -1;
    }

    // create windows
    namedWindow("Linear Blend", 1);

    beta = (1.0 - alpha);
    addWeighted(src1, alpha, src2, beta, 0.0, dst);

    imshow("Linear Blend", dst);

    waitKey(0);
    return 0;
}
