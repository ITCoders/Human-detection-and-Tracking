#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

String face_cascade1_name = "haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade1;

void detect_faces( Mat frame );

int main(int argc, const char** argv)
{
	Mat frame;
	Mat img;
	/*throwing error when any cascade file is unable to load*/
    if( !face_cascade1.load( face_cascade1_name ) )
    { 
        printf("--(!)Error loading face cascade1\n"); return -1; 
    }


	VideoCapture cam("5.mp4");  /*reading input video*/
	if ( !cam.isOpened() )  // if not read exit
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    double fps = cam.get(CV_CAP_PROP_FPS); /*getting frames per seconds*/
    cout << "frames per seconds :" << fps << endl;

    namedWindow("InputVideo",CV_WINDOW_AUTOSIZE);

    while(1)
    {
    	bool read_success= cam.read(frame);

    	if(!read_success) /*read sucessfull*/
    	{
    		cout << "Can't read frames" << endl;
    		break;
    	}

    	imshow("InputVideo", frame);
		waitKey(1000/fps); /*interval between two cosecutive frames*/
 
    }

    return 0;
}

