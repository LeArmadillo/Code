#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

 using namespace std;
 using namespace cv;

 void detectAndDisplay( Mat frame );

string window_name = "Capture - Face detection";
Mat frame;

 int main( int argc, const char** argv )
 {
	CvCapture* capture;

   char filename[100];
   sprintf(filename, "J:/00006.JPG.AVI");
   capture = cvCaptureFromAVI(filename);

   if( capture )
   {
     while( true )
     {
     frame = cvQueryFrame( capture );

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { 
		   detectAndDisplay( frame ); 
	   }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }	
   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	double t = (double)getTickCount();

	CvSize size;
	size.height = 360;
	size.width = 640;

	IplImage *t1 = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage *t2 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t3 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t4 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t5 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t6 = cvCreateImage( size, IPL_DEPTH_8U, 1 );

	*t1 = frame;
	cvCvtColor( t1, t2, CV_RGB2GRAY );
	cvSobel( t2, t3, 1, 0, 7 );
	cvConvertScale( t3, t4, 1.0/255.0, 0.0 );
	cvThreshold( t4, t5, 126, 255, CV_THRESH_BINARY );
	cvCanny( t5, t6, 10, 50, 3 );

	cvShowImage( "T1", t1 );
	cvShowImage( "T2", t2 );
	cvShowImage( "T3", t3 );
	cvShowImage( "T4", t4 );
	cvShowImage( "T5", t5 );
	cvShowImage( "T6", t6 );

	
	cvReleaseImage( &t1 );
	cvReleaseImage( &t2 );
	cvReleaseImage( &t3 );
	cvReleaseImage( &t4 );
	cvReleaseImage( &t5 );
	cvReleaseImage( &t6 );
	

	t = (double)getTickCount() - t;
	printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency()); 

	//imshow( window_name, frame );
}