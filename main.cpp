#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

String face_cascade1_name = "haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade1;

void detect_faces( Mat frame, double fps);

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


    while(1)
    {
    	bool read_success= cam.read(frame);

    	if(!read_success) /*read sucessfull*/
    	{
    		cout << "Can't read frames" << endl;
    		break;
    	}

    	detect_faces(frame,fps);	 
    }

    return 0;
}

void detect_faces( Mat frame, double fps )
{
	std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY ); /*converting input image in grayscale form*/
    equalizeHist( frame_gray, frame_gray ); 
    
    /*Detecting faces*/
    face_cascade1.detectMultiScale( frame_gray, faces, 1.2, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < faces.size(); i++ )
    {   
        /*Drawing circle around faces*/ 
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 0, 250, 255 ), 4, 8, 0 );
    }

    namedWindow( "Detected faces", WINDOW_AUTOSIZE );
    imshow( "Detected faces", frame );
    waitKey(100/fps);   
}
