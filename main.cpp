#include "opencv2/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

#include <iostream>

using namespace std;
using namespace cv;

String face_cascade1_name = "haarcascade_profileface.xml";
CascadeClassifier face_cascade1;
Ptr<face::FaceRecognizer> recognizer = face::createLBPHFaceRecognizer();
vector<Rect> detect_faces( Mat frame);
Mat detect_people( Mat frame);
Mat draw_faces(Mat frame1, vector<Rect> faces);
int* recognize_face(Mat frame, vector<Rect> faces);

int main (int argc, const char * argv[])
{
    
    VideoCapture cap(argv[1]);
    recognizer->load("model.yaml");
 
    if (!cap.isOpened()) /*checking whether video file is read successfully*/
    {
        cout << "Cannot open the video file" << endl;
        return -1;
    }    
    
    /*throwing error when any cascade file is unable to load*/
    if( !face_cascade1.load( face_cascade1_name ) )
    { 
        printf("--(!)Error loading face cascade1\n"); return -1; 
    }

    Mat frame,frame1,frame2;
    vector<Rect> faces;
 
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        frame1=detect_people(frame);
        faces=detect_faces(frame);
        frame2=draw_faces(frame1, faces); /*draw circle around faces*/
	recognize_face(frame,faces);
        imshow("human_detection and face_detction", frame);
        waitKey(1);
    }
    return 0;
}

Mat detect_people( Mat frame)
{   
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	/*resize(frame, frame, Size(),0.5,0.5);*/
    vector<Rect> detected, detected_filtered;
    hog.detectMultiScale(frame, detected, 0, Size(8,8), Size(32,32), 1.05, 2);
    size_t i, j;
    /*checking for the distinctly detected human in a frame*/
    for (i=0; i<detected.size(); i++) 
    {
        Rect r = detected[i];
        for (j=0; j<detected.size(); j++) 
            if (j!=i && (r & detected[j]) == r)
                break;
        if (j== detected.size())
                detected_filtered.push_back(r);
        }
    /*for each distinctly detected human draw rectangle around it*/
    for (i=0; i<detected_filtered.size(); i++) 
    {
        Rect r = detected_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 3);       
    }
 
        return frame;
}

vector<Rect> detect_faces( Mat frame)
{
	vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY ); /*converting input image in grayscale form*/
    equalizeHist( frame_gray, frame_gray ); 
    
    /*resize(frame_gray, frame_gray, Size(),0.5,0.5);*/
    /*Detecting faces*/
    face_cascade1.detectMultiScale( frame_gray, faces, 1.2, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    return faces;  
}

Mat draw_faces(Mat frame1, vector<Rect> faces)
{   
    for ( size_t i = 0; i < faces.size(); i++ )
    {   
        /*Drawing circle around faces*/
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame1, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 0, 250, 255 ), 4, 8, 0 );
    }
    return frame1;

}

int* recognize_face(Mat frame, vector<Rect> faces)
{
	int a;
	double b;
	int predict_label[100];
	double predict_conf[100];
	Mat frame_original_grayscale;
	for ( size_t i = 0; i < faces.size(); i++ )
	{
		cv::cvtColor( frame, frame_original_grayscale, COLOR_BGR2GRAY );
		equalizeHist(frame_original_grayscale,frame_original_grayscale); 
		
		
		recognizer->predict(frame_original_grayscale, a,b);
		predict_label[i]=a;
		predict_conf[i]=b;
		cout << a << b << endl;
	}
	return predict_label;
}

