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
string in_name = "KMeans - IN";
string out_name = "KMeans - OUT";
Mat frame;
Mat centre = cvCreateMat( 360, 640, CV_32SC1 );
Mat outFrame = cvCreateMat( 360, 640, CV_32SC1 );

 int main( int argc, const char** argv )
 {
	CvCapture* capture;

	CvSize size;
	size.height = 360;
	size.width = 640;	

	IplImage *t1 = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage *t2 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t3 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t4 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t5 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t6 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t7 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t8 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t9 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t0 = cvCreateImage( size, IPL_DEPTH_8U, 1 );

   char filename[100];
   sprintf(filename, "F:/fcrpXXX.360.AVI");
   capture = cvCaptureFromAVI(filename);

   if( capture )
   {
     while( true )
     {
     frame = cvQueryFrame( capture );

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { 
		   	double t = (double)getTickCount();

			kmeans( frame, 4, outFrame, cvTermCriteria( CV_TERMCRIT_ITER, 2, 2 ), 3, 3, centre);

			imshow( in_name, frame );
			imshow( out_name, outFrame );
			/*
			cvCvtColor( t1, t2, CV_RGB2GRAY );
			cvSobel( t2, t3, 1, 0, 3 );
			cvThreshold( t3, t4, 200, 255, CV_THRESH_BINARY );
			cvSobel( t2, t5, 1, 0, 5 );
			cvThreshold( t5, t6, 250, 255, CV_THRESH_BINARY_INV );
			cvSobel( t2, t7, 1, 0, 7 );
			cvThreshold( t7, t8, 250, 255, CV_THRESH_BINARY_INV );
			cvSobel( t2, t9, 1, 0, 9 );
			cvThreshold( t9, t0, 250, 255, CV_THRESH_BINARY_INV );

			//cvConvertScale( t3, t4, 1.0/255.0, 0.0 );
			//cvThreshold( t4, t5, 126, 255, CV_THRESH_BINARY );
			//cvCanny( t5, t6, 10, 50, 3 );

			//cvShowImage( "T1", t1 );
			//cvShowImage( "T2", t2 );
			cvShowImage( "T3", t3 );
			cvShowImage( "T4", t4 );
			cvShowImage( "T5", t5 );
			cvShowImage( "T6", t6 );
			cvShowImage( "T7", t7 );
			cvShowImage( "T8", t8 );
			cvShowImage( "T9", t9 );
			cvShowImage( "T0", t0 );

			/*
			cvReleaseImage( &t1 );
			cvReleaseImage( &t2 );
			cvReleaseImage( &t3 );
			cvReleaseImage( &t4 );
			cvReleaseImage( &t5 );
			cvReleaseImage( &t6 );
			*/
	

			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());  
	   }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }	
   return 0;
 }
