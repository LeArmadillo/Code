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
	Mat frameXYZ;
	Mat frameYBR;
	Mat frameHSV;
	Mat frameHLS;
	Mat frameLAB;
	Mat frameLUV;
	IplImage* result = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );
	IplImage* resultXYZ = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );
	IplImage* resultYBR = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );
	IplImage* resultHSV = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );
	IplImage* resultHLS = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );
	IplImage* resultLAB = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );
	IplImage* resultLUV = cvCreateImage( cvSize(640, 360), IPL_DEPTH_8U, 3 );

   if( capture )
   {
     while( true )
     {
		//int cluster_count = cvRandInt(&rng)%MAX_CLUSTERS + 1;
		//int i, sample_count = cvRandInt(&rng)%1000 + 1;
		int cluster_count = 8;
		frame = cvQueryFrame( capture );
		cvtColor( frame, frameXYZ, CV_BGR2XYZ, 0 );
		cvtColor( frame, frameYBR, CV_BGR2YCrCb, 0 );
		cvtColor( frame, frameHSV, CV_BGR2HSV, 0 );
		cvtColor( frame, frameHLS, CV_BGR2HLS, 0 );
		cvtColor( frame, frameLAB, CV_BGR2Lab, 0 );
		cvtColor( frame, frameLUV, CV_BGR2Luv, 0 );

		
		//Mat points = frame.reshape( 1, 640*360 );
		Mat points = frame.reshape( 1, frame.cols*frame.rows );
		Mat clusters, centers;
		points.convertTo(points, CV_32FC3, 1.0/255.0);
		Mat pointsXYZ = frameXYZ.reshape( 1, 640*360 );
		//pointsXYZ = pointsXYZ.colRange(0, 1);
		Mat clustersXYZ, centersXYZ;
		pointsXYZ.convertTo(pointsXYZ, CV_32FC3, 1.0/255.0);
		Mat pointsYBR = frameYBR.reshape( 1, 640*360 );
		pointsYBR = pointsYBR.colRange(1, 2);
		Mat clustersYBR, centersYBR;
		pointsYBR.convertTo(pointsYBR, CV_32FC3, 1.0/255.0);
		Mat pointsHSV = frameHSV.reshape( 1, 640*360 );
		pointsHSV = pointsHSV.colRange( 0, 1 );
		Mat clustersHSV, centersHSV;
		pointsHSV.convertTo(pointsHSV, CV_32FC3, 1.0/255.0);
		Mat pointsHLS = frameHLS.reshape( 1, 640*360 );
		pointsHLS = pointsHLS.col( 0 );
		Mat clustersHLS, centersHLS;
		pointsHLS.convertTo(pointsHLS, CV_32FC3, 1.0/255.0);
		Mat pointsLAB = frameLAB.reshape( 1, 640*360 );
		pointsLAB = pointsLAB.colRange(0,1);
		Mat clustersLAB, centersLAB;
		pointsLAB.convertTo(pointsLAB, CV_32FC3, 1.0/255.0);
		Mat pointsLUV = frameLUV.reshape( 1, 640*360 );
		pointsLUV = pointsLUV.colRange(0,1);
		Mat clustersLUV, centersLUV;
		pointsLUV.convertTo(pointsLUV, CV_32FC3, 1.0/255.0);


		kmeans( points, cluster_count, clusters,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		kmeans( pointsXYZ, cluster_count, clustersXYZ,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		kmeans( pointsYBR, cluster_count, clustersYBR,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		kmeans( pointsHSV, cluster_count, clustersHSV,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		kmeans( pointsHLS, cluster_count, clustersHLS,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		kmeans( pointsLAB, cluster_count, clustersLAB,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );
		kmeans( pointsLUV, cluster_count, clustersLUV,
			TermCriteria( CV_TERMCRIT_ITER, 1, 10.0 ),
			1, KMEANS_RANDOM_CENTERS, centers );

		int r2 = 0;
		for(int row=0; row<frame.rows; row++) 
			for(int col=0; col<frame.cols; col++) 
			{
				int a = clusters.at<int>(r2, 0);
				PUTPIXELMACRO( result, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				a = clustersXYZ.at<int>(r2, 0);
				PUTPIXELMACRO( resultXYZ, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				a = clustersYBR.at<int>(r2, 0);
				PUTPIXELMACRO( resultYBR, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				a = clustersHSV.at<int>(r2, 0);
				PUTPIXELMACRO( resultHSV, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				a = clustersHLS.at<int>(r2, 0);
				PUTPIXELMACRO( resultHLS, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				a = clustersLAB.at<int>(r2, 0);
				PUTPIXELMACRO( resultLAB, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				a = clustersLUV.at<int>(r2, 0);
				PUTPIXELMACRO( resultLUV, col, row, new_pixel[a], result->widthStep, (result->widthStep/result->width), 3 );
				r2++;
			}

		namedWindow( "INPUT" );
		namedWindow( "OUTPUT" );
		namedWindow( "TEMP" );
		namedWindow( "OUTPUTXYZ" );
		namedWindow( "OUTPUTYBR" );
		namedWindow( "OUTPUTHSV" );
		namedWindow( "OUTPUTHLS" );
		namedWindow( "OUTPUTLAB" );
		namedWindow( "OUTPUTLUV" );
 

		imshow( "INPUT", frame );
		cvShowImage( "OUTPUT", result );
		imshow( "TEMP", clusters );
		cvShowImage( "OUTPUTXYZ", resultXYZ );
		cvShowImage( "OUTPUTYBR", resultYBR );
		cvShowImage( "OUTPUTHSV", resultHSV );
		cvShowImage( "OUTPUTHLS", resultHLS );
		cvShowImage( "OUTPUTLAB", resultLAB );
		cvShowImage( "OUTPUTLUV", resultLUV );

		/*
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
		*/

		// RELEASE?

		int key = cvWaitKey(0);
		if( key == 27 ) // ‘ESC’
			break;
			
	 }
	}
}