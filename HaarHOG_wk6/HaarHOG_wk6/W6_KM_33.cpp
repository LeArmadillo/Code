//#include "cxcore.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include "../../utilities.h"


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

 using namespace std;
 using namespace cv;

 int main( int argc, const char** argv )
 {
	#define MAX_CLUSTERS 5
	CvScalar color_tab[MAX_CLUSTERS];
	IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
	CvRNG rng = cvRNG(0xffffffff);
	//Mat frame;

	CvSize size;
	size.height = 360;
	size.width = 640;	

	IplImage *t1 = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage *t2 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t3 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t4 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t5 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t6 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t7 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t8 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
	IplImage *t9 = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *t0 = cvCreateImage( size, IPL_DEPTH_8U, 1 );
		IplImage *tA = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *tB = cvCreateImage( size, IPL_DEPTH_8U, 1 );
		IplImage *tC = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *tD = cvCreateImage( size, IPL_DEPTH_8U, 1 );
		IplImage *tE = cvCreateImage( size, IPL_DEPTH_16S, 1 );
	IplImage *tF = cvCreateImage( size, IPL_DEPTH_8U, 1 );

	unsigned char new_pixel[8][3] = {
		{ 0, 0, 0 },
		{ 0, 255, 255 },
		{ 255, 255, 0 },
		{ 255, 0, 255 },
		{ 255, 0, 0 },
		{ 0, 255, 0 },
		{ 0, 0, 255 },
		{ 255, 255, 255 } };


	CvCapture* capture;
	char filename[100];
	//sprintf(filename, "F:/M4H00707.MP4.AVI");
	sprintf(filename, "F:fcrpXXX.360.AVI");
	capture = cvCaptureFromAVI(filename);
	Mat frame;
	IplImage* result = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 ); 

   if( capture )
   {
     while( true )
     {
		//int cluster_count = cvRandInt(&rng)%MAX_CLUSTERS + 1;
		//int i, sample_count = cvRandInt(&rng)%1000 + 1;
		int cluster_count = 2;
		frame = cvQueryFrame( capture );
		//result = frame;
		
		Mat points = frame.reshape( 1, 640*360 );
		Mat clusters, centers;
		//Mat clusters = Mat( frame.rows*frame.cols, 1, CV_32SC1 );
		//Mat centers = Mat( frame.rows*frame.cols, 1, CV_32FC3 );
		points.convertTo(points, CV_32FC3, 1.0/255.0);

		kmeans( points, cluster_count, clusters,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		//clusters = clusters.reshape( 360 );

		int colors[8];
		for(int i=0; i<8; i++) {
			colors[i] = 255/(i+1);
		}
		//result = Mat(frame.rows, frame.cols, CV_32FC3);
		//for(int i=0; i<frame.cols*frame.rows; i++) {
		int r2 = 0;
		for(int row=0; row<frame.rows; row++) 
			for(int col=0; col<frame.cols; col++) 
			{
				int a = clusters.at<int>(r2, 0);
				PUTPIXELMACRO( result, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				r2++;
			}

		namedWindow( "INPUT" );
		namedWindow( "OUTPUT" );
		namedWindow( "TEMP" );
 

		imshow( "INPUT", frame );
		cvShowImage( "OUTPUT", result );
		imshow( "TEMP", clusters );

		*t1 = frame;
		cvCvtColor( t1, t2, CV_RGB2GRAY );
		cvCvtColor( result, t3, CV_RGB2GRAY );
		cvSobel( t2, t7, 1, 0, 9 );
		cvThreshold( t7, t8, 250, 255, CV_THRESH_BINARY );
		cvSobel( t3, t9, 1, 0, 9 );
		cvThreshold( t9, t0, 250, 255, CV_THRESH_BINARY );
		cvSobel( t2, tE, 1, 0, 7 );
		cvThreshold( tE, tF, 250, 255, CV_THRESH_BINARY );
		cvSobel( t3, tC, 1, 0, 7 );
		cvThreshold( tC, tD, 250, 255, CV_THRESH_BINARY );
		cvSobel( t2, tA, 1, 0, 5 );
		cvThreshold( tA, tB, 250, 255, CV_THRESH_BINARY );
		cvSobel( t3, t5, 1, 0, 5 );
		cvThreshold( t5, t6, 250, 255, CV_THRESH_BINARY );

		cvShowImage( "T7", t7 );
		cvShowImage( "T8", t8 );
		cvShowImage( "T9", t9 );
		cvShowImage( "T0", t0 );
		cvShowImage( "T5", t5 );
		cvShowImage( "T6", t6 );
		cvShowImage( "TA", tA );
		cvShowImage( "TB", tB );
		cvShowImage( "TC", tC );
		cvShowImage( "TD", tD );
		cvShowImage( "TE", tE );
		cvShowImage( "TF", tF );

		// RELEASE?

		int key = cvWaitKey(0);
		if( key == 27 ) // ‘ESC’
			break;
			
	 }
	}
}