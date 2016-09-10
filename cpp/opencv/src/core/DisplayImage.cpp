#include <stdio.h>

#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 2) 
    {
        printf("Usage: ./DisplayImage <image>\n");
        return -1;
    }

    Mat image;
    image = imread(argv[1], 1);  // read the image

    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    namedWindow("Display Image", WINDOW_AUTOSIZE);  // create a window for display
    imshow("Display Image", image);  // show the image

    waitKey(0); // wait for a keystroke in the window
    return 0;
}
