#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

static void help()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "This program shows how to write video files."                                   << endl
        << "You can extract the R or G or B color channel of the input video."              << endl
        << "Usage:"                                                                         << endl
        << "./Video inputvideoName [ R | G | B] [Y | N]"                              << endl
        << "------------------------------------------------------------------------------" << endl
        << endl;
}


int main( int argc, char* argv[1] )
{
    help();

    if ( argc != 4 )
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }

    const string source = argv[1];
    const bool askOutputType = argv[3][0] == 'Y';

    VideoCapture inputVideo(source);
    if ( !inputVideo.isOpened() )
    {
        cout << "Could not open the input video: " << source << endl;
        return -1;
    }

    string::size_type pAt = source.find_last_of('.');
    const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";
    int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));

    char EXT[] = {
        (char)(ex & 0XFF),
        (char)((ex & 0XFF00) >> 8),
        (char)((ex & 0XFF0000) >> 16),
        (char)((ex & 0XFF000000) >> 24),
        0
    };

    Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
                  (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo;
    if ( askOutputType )
        outputVideo.open( NAME, ex=-1, inputVideo.get(CV_CAP_PROP_FPS), S, true);
    else
        outputVideo.open( NAME, ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);

    if ( !outputVideo.isOpened() )
    {
        cout << "Could not open the output video for write: " << source << endl;
        return -1;
    }

    cout << "Input frame resolution: Width=" << S.width << " Height=" << S.height
         << " of nr#: " << inputVideo.get(CV_CAP_PROP_FRAME_COUNT) << endl;
    cout << "Input codec type: " << EXT << endl;

    int channel = 2;
    switch( argv[2][0] )
    {
        case 'R': channel = 2; break;
        case 'G': channel = 1; break;
        case 'B': channel = 0; break;
    }

    Mat src, res;
    vector<Mat> spl;

    for (;;)
    {
        inputVideo >> src;
        if (src.empty()) break;

        split(src, spl);
        for ( int i = 0; i < 3; ++i )
        {
            if ( i != channel )
            {
                spl[i] = Mat::zeros(S, spl[0].type()); 
            }
        }
        merge(spl, res);

        outputVideo << res;
    }

    cout << "Finished writing" << endl;
    return 0;
}
