#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#define w 400

using namespace cv;


// Function headers
void MyEllipse(Mat img, double angle);
void MyFilledCircle(Mat img, Point center);
// void MyPolygon(Mat img);
// void MyLine(Mat img, Point start, Point end);


// main
int main(void)
{
    // windows names
    char atom_window[] = "Drawing 1: Atom";
    char rook_window[] = "Drawing 2: Rook";

    // create black empty image
    Mat atom_image = Mat::zeros(w, w, CV_8UC3);
    Mat rook_image = Mat::zeros(w, w, CV_8UC3);

    // create ellipses
    MyEllipse(atom_image, 90);
    MyEllipse(atom_image, 0);
    MyEllipse(atom_image, 45);
    MyEllipse(atom_image, -45);

    MyFilledCircle(atom_image, Point(w/2, w/2));
 
    // display
    imshow(atom_window, atom_image);
    moveWindow(atom_window, 0, 200);


    waitKey(0);
    return 0;
    
}


void MyEllipse(Mat img, double angle)
{
    int thickness = 2;
    int lineType = 8;

    ellipse(img,
        Point(w/2, w/2),
        Size(w/4, w/16),
        angle,
        0,
        360,
        Scalar(255, 0, 0),
        thickness,
        lineType
    );
}


void MyFilledCircle(Mat img, Point center)
{
    circle(img,
        center,
        w/32,
        Scalar(0, 0, 255),
        FILLED,
        LINE_8
    );
}
