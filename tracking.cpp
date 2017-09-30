#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main( int argc, char** argv ){
  //help
  if(argc<2){
    cout <<"Your command must contain the arguments mentioned below:" 
          << endl 
          << "./filename path/video_name"
          << endl
          << "Example: ./track video/abc.avi"
          << endl;
    return 0;
  }

  //variable declaration
  //ROI: Region Of Interest
  Rect2d roi;
  Mat frame;
  // Creating object of KCF tracker
  Ptr<Tracker> tracker = Tracker::create( "KCF" );
  // input video
  string video = argv[1];
  VideoCapture cap(video);

  //getting a frame out of video
  cap >> frame;

  // selecting a bounding box which is a rectangle
  bool showCrosshair = false; 
  bool fromCenter = false; 
  roi = selectROI("tracker", frame, fromCenter, showCrosshair);
  cout << roi << endl;
  cout << roi.x << "  " << roi.y << endl;

  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;

  // initialize the tracker
  tracker->init(frame,roi);
  // perform the tracking process
  printf("Start the tracking process, press ESC to quit.\n");

  for ( ;; ){
    // get frame from the video
    cap >> frame;
    // stop if no more frames
    if(frame.rows==0 || frame.cols==0)
      break;
    
    //continue tracking only if ROI is in the scope of video
    //stop if ROI goes out of range of image
    if (roi.x>=1 && roi.y>=1 && (roi.x+roi.width) < frame.cols && (roi.y+roi.height) <frame.rows)
    {
      // update the tracking result
      tracker->update(frame,roi);
      // draw rectangle around the tracked object
      rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
      // show image with the tracked object
      imshow("tracker",frame);
    }

    else{
      imshow("tracker",frame);
    }
    
    //quit on ESC button
    if(waitKey(1)==27)break;
  }
  return 0;
}