#include "opencv2/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

String face_cascade1_name = "face_cascades/haarcascade_profileface.xml";
CascadeClassifier face_cascade1;
Ptr<face::FaceRecognizer> recognizer = face::createLBPHFaceRecognizer();
vector<Rect> detect_faces( Mat frame);
Mat detect_people( Mat frame);
Mat draw_faces(Mat frame1, vector<Rect> faces);
int* recognize_face(Mat frame, vector<Rect> faces);
Mat put_label_on_face(Mat frame,vector<Rect> faces,int* label);


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
    int *label;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        int height;
        height=((frame.size().height)*800)/frame.size().width;
        resize(frame, frame, Size(800, height));
        frame1=detect_people(frame);
        faces=detect_faces(frame);
        frame2=draw_faces(frame1, faces); /*draw circle around faces*/
	    label=recognize_face(frame,faces);
	    put_label_on_face(frame,faces,label);
        imshow("human_detection and face_detection", frame);
        waitKey(1);
    }
    return 0;
}

Mat detect_people( Mat frame)
{   
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> detected, detected_filtered;
    hog.detectMultiScale(frame, detected, 0, Size(8,8), Size(16,16), 1.06, 2);
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
        rectangle(frame, r.tl(), r.br(), Scalar(0,0,255), 2);       
    }
 
        return frame;
}

vector<Rect> detect_faces( Mat frame)
{
	vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY ); /*converting input image in grayscale form*/
    //equalizeHist( frame_gray, frame_gray ); 
    /*Detecting faces*/
    face_cascade1.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(20, 20) );
    return faces;  
}

Mat draw_faces(Mat frame1, vector<Rect> faces)
{   
    for ( size_t i = 0; i < faces.size(); i++ )
    {   
        /*Drawing rectangle around faces*/
        rectangle(frame1, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2, LINE_8, 0);
    }
    return frame1;

}

int* recognize_face(Mat frame, vector<Rect> faces)
{
	int a;
	double b;
	static int predict_label[100];
	double predict_conf[100];
	Mat frame_original_grayscale;
	for ( size_t i = 0; i < faces.size(); i++ )
	{
		cvtColor( frame, frame_original_grayscale, COLOR_BGR2GRAY ); /*converting frame to grayscale*/
		//equalizeHist(frame_original_grayscale,frame_original_grayscale); 
		
		/*recognizing faces to predict label and confidence factor*/ 
		recognizer->predict(frame_original_grayscale, a,b); 
		predict_label[i]=a;
		predict_conf[i]=b;
		cout << "label="<<a <<endl<< "conf="<<b << endl;
	}
	return predict_label;
}


Mat put_label_on_face(Mat frame,vector<Rect> faces,int* label)
{
	for ( size_t j = 0; j < faces.size(); j++ )
	{
        /*converting integer to string*/
        stringstream ss;
        ss << label[j];
        string str_label = ss.str();
        /*writing label on the image frame*/
        /*putText(InputOutputArray img, const String& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=LINE_8, bool bottomLeftOrigin=false )*/
	    putText(frame, str_label, Point(faces[j].x, faces[j].y), FONT_HERSHEY_SIMPLEX,1, Scalar(255,255,255), 2);
	}
	return frame;
}


