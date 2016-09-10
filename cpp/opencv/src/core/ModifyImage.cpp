#include <opencv2/opencv.hpp>

using namespace cv;


int main(int argc, char **argv)
{
    char *ImageName = argv[1];

    Mat Image;
    Image = imread(ImageName, 1);

    if (argc != 2 || !Image.data)
    {
        printf("No image data \n");
    }

    Mat GrayImage;
    cvtColor(Image, GrayImage, CV_BGR2GRAY);

    namedWindow(ImageName, CV_WINDOW_AUTOSIZE);
    namedWindow("Gray image", CV_WINDOW_AUTOSIZE);

    imshow(ImageName, Image);
    imshow("Gray image", GrayImage);

    waitKey(0);
    return 0;
}
