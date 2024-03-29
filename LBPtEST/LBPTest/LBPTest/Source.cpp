 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 //String face_cascade_name = "haarcascade_frontalface_alt.xml";
 //String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 String cascade_name = "cascade.xml";
 //CascadeClassifier face_cascade;
 //CascadeClassifier eyes_cascade;
 CascadeClassifier cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;

   //-- 1. Load the cascades
   //( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return 9; };
   //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return 8; };
   if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return 7; };

   //-- 2. Read the video stream
   //capture = cvCaptureFromCAM( -1 );
   	// Load the vidieo file
   char filename[100];
   sprintf(filename, "C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Haar Training/fullCRoom/Positives/fcrpXXX.wmv.avi");
   //sprintf(filename, "C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Recordings/M4H00707.MP4.AVI");
   capture = cvCaptureFromAVI(filename);
   if( capture )
   {
     while( true )
     {
	cvWaitKey(15);
   frame = cvQueryFrame( capture );

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { detectAndDisplay( frame ); }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }	else
   {
	   return 5;
   }
   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( int i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
  }
  //-- Show what you got
  imshow( window_name, frame );
 }