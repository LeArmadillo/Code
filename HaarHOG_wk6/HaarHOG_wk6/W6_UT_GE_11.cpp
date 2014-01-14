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

 /** Function Headers */
 Mat detectAndDisplay( Mat frame, CvSize size, CvSize &sizeROI );
 void colorSpaceAnalysis( Mat frameBGR, CvSize size ); 
 void edgeAnalysis( Mat frameE, CvSize sizeE );

 /** Global variables */
 //String face_cascade_name = "haarcascade_frontalface_alt.xml";
 //String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 String three_cascade_name = "haarcascade_frontalface_default.xml";
 String two_cascade_name = "haarcascade_mcs_upperbody.xml";
 String one_cascade_name = "haarcascade_mcs_upperbody.xml";
 String four_cascade_name = "cascadeA.xml";
 CascadeClassifier one_cascade;
 CascadeClassifier two_cascade;
 CascadeClassifier three_cascade;
 CascadeClassifier four_cascade;
 string window_name = "Capture - Face detection";
 string temp_window_name = "ROI";
 string temp_window_name1 = "ROIA";
 string temp_window_name2 = "ROIB";
 string temp_window_name3 = "ROIC";
 string temp_window_name4 = "ROID";
 string temp_window_name5 = "ROIE";
 RNG rng(12345);

 int main( int argc, const char** argv )
 {
	//#define MAX_CLUSTERS 5
	//CvScalar color_tab[MAX_CLUSTERS];
	IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
	CvRNG rng = cvRNG(0xffffffff);
	//Mat frame;

	CvSize size, sizeROI;
	size.height = 360;
	size.width = 640;	

	Mat wholeImage;
	Mat ROIframe;

	CvCapture* capture;
	char filename[100];
	//sprintf(filename, "F:/M4H00707.MP4.AVI");
	sprintf(filename, "F:/M4H00707.MP4.AVI");
	capture = cvCaptureFromAVI(filename);


   if( capture )
   {
     while( true )
     {
		wholeImage = cvQueryFrame( capture );
		ROIframe = detectAndDisplay( wholeImage, size, sizeROI );
		if( ROIframe.empty() )	{
			continue;
		}
		//colorSpaceAnalysis( ROIframe, sizeROI );
		edgeAnalysis( ROIframe, sizeROI );
		
		int key = cvWaitKey(0);
		if( key == 27 ) // �ESC�
			break;
	 }
   }
 }

 void colorSpaceAnalysis( Mat frameBGR, CvSize size )
 {	
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

	Mat frame2 = frameBGR;
	Mat frameXYZ;
	Mat frameYBR;
	Mat frameHSV;
	Mat frameHLS;
	Mat frameLAB;
	Mat frameLUV;
	IplImage* result = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage* resultXYZ = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage* resultYBR = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage* resultHSV = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage* resultHLS = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage* resultLAB = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage* resultLUV = cvCreateImage( size, IPL_DEPTH_8U, 3 );

	int cluster_count = 4;
	cvtColor( frameBGR, frameXYZ, CV_BGR2XYZ, 0 );
	cvtColor( frameBGR, frameYBR, CV_BGR2YCrCb, 0 );
	cvtColor( frameBGR, frameHSV, CV_BGR2HSV, 0 );
	cvtColor( frameBGR, frameHLS, CV_BGR2HLS, 0 );
	cvtColor( frameBGR, frameLAB, CV_BGR2Lab, 0 );
	cvtColor( frameBGR, frameLUV, CV_BGR2Luv, 0 );

		
	Mat points = frameBGR.reshape( 1, frameBGR.cols*frameBGR.rows );
	//Mat points = frame2.reshape( 1, frame2.cols*frame2.rows );
	Mat clusters, centers;
	points.convertTo(points, CV_32FC3, 1.0/255.0);
	Mat pointsXYZ = frameXYZ.reshape( 1, size.height*size.width );
	//pointsXYZ = pointsXYZ.colRange(0, 1);
	Mat clustersXYZ, centersXYZ;
	pointsXYZ.convertTo(pointsXYZ, CV_32FC3, 1.0/255.0);
	Mat pointsYBR = frameYBR.reshape( 1, size.height*size.width );
	//pointsYBR = pointsYBR.col(0);
	Mat clustersYBR, centersYBR;
	pointsYBR.convertTo(pointsYBR, CV_32FC3, 1.0/255.0);
	Mat pointsHSV = frameHSV.reshape( 1, size.height*size.width );
	//pointsHSV = pointsHSV.colRange( 0, 1 );
	Mat clustersHSV, centersHSV;
	pointsHSV.convertTo(pointsHSV, CV_32FC3, 1.0/255.0);
	Mat pointsHLS = frameHLS.reshape( 1, size.height*size.width );
	//pointsHLS = pointsHLS.colRange( 0, 1 );
	Mat clustersHLS, centersHLS;
	pointsHLS.convertTo(pointsHLS, CV_32FC3, 1.0/255.0);
	Mat pointsLAB = frameLAB.reshape( 1, size.height*size.width );
	//pointsLAB = pointsLAB.col(0);
	Mat clustersLAB, centersLAB;
	pointsLAB.convertTo(pointsLAB, CV_32FC3, 1.0/255.0);
	Mat pointsLUV = frameLUV.reshape( 1, size.height*size.width );
	//pointsLUV = pointsLUV.col(0);
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
	for(int row=0; row<frameBGR.rows; row++) 
		for(int col=0; col<frameBGR.cols; col++) 
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
 

	imshow( "INPUT", frameBGR );
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
		
}

/** @function detectAndDisplay */
Mat detectAndDisplay( Mat frame, CvSize size, CvSize &sizeROI )
{

  std::vector<Rect> one;
  std::vector<Rect> two;
  std::vector<Rect> three;
  std::vector<Rect> four;
  Mat frame_gray;
  Mat frame_gray2;
  Mat frame_gray3;
  Mat regionOfInterest;


  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  fflush(stdout);
	vector<Rect> found, found_filtered;
	double t = (double)getTickCount();
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(frame, found, 0, Size(8,8), Size(0,0), 1.05, 2);

	if( found.size() == 0 )	{
		return regionOfInterest;
	}

	size_t i, j;
	for( i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		for( j = 0; j < found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		if( j == found.size() )
			found_filtered.push_back(r);
	}
	
	for( i = 0; i < found_filtered.size(); i++ )
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		//rectangle(frame, r.tl(), r.br(), cv::Scalar(0,0,255), 3);

		
		//Mat regionOfInterest3;
		//printf(" %d %d %d %d \n", r.x, r.y, r.x+r.width, r.y+r.height);
		int x1 = r.x;
		int x2 = r.width;
		int y1 = r.y-25;
		int y2 = r.height/2;
		if (x1 < 0) x1=0;
		if ((x1+x2)-640 > 0) x2=x2-((x1+x2)-640);
		if (y1 < 0) y1=0;
		if ((y1+y2)-360 > 0) y2=y2-((y1+y2)-360);
		regionOfInterest = frame( Rect( x1, y1, x2, y2 ));
	 
		cvtColor( regionOfInterest, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );
		  //-- Detect faces
		  one_cascade.detectMultiScale( frame_gray, one, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(25, 25) );
		  
		  for( int i = 0; i < one.size(); i++ )	{
			imshow( temp_window_name, regionOfInterest );
		  }
	}

	cvtColor( frame, frame_gray2, CV_BGR2GRAY );
	equalizeHist( frame_gray2, frame_gray2 );
	
	two_cascade.detectMultiScale( frame_gray2, two, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	three_cascade.detectMultiScale( frame_gray2, three, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	//if( two.size == 0 || three.size == 0 )	{
	//	return regionOfInterest;
	//}
	for( int i = 0; i < three.size(); i++ )
	{
		for( int k = 0; k < two.size(); k++ )
		{
		  
			if( ( three[i].x < two[k].x+two[k].width*0.5 && three[i].x > two[k].x-two[k].width*0.5 &&
				three[i].y < two[k].y+two[k].height*0.5 && three[i].y > two[k].y-two[k].height*0.5 ) ||
				( two[k].x < three[i].x+three[i].width*0.5 && two[k].x > three[i].x-three[i].width*0.5 &&
				two[k].y < three[i].y+three[i].height*0.5 && two[k].y > three[i].y-three[i].height*0.5 ) )
			{
				//Point center( (two[k].x + two[k].width*0.5 + three[i].x + three[i].width*0.5)/2, (two[k].y + two[k].height*0.5 + three[i].y + three[i].height*0.5)/2 );
				//ellipse( frame, center, Size( (two[k].width*0.5+three[i].width*0.5)/2, (two[k].height*0.5+three[i].height*0.5)/2 ), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
				int x1 = two[k].x-15;
				int x2 = two[k].width+15;
				int y1 = two[k].y+25;
				int y2 = two[k].height+25;
				if (x1 < 0) x1=0;
				if (x1 > 640) x1=639;
				if ((x1+x2)-640 > 0) x2=x2-((x1+x2)-640);
				if (y1 < 0) y1=0;
				if (y1 > 360) y1=359;
				if ((y1+y2)-360 > 0) y2=y2-((y1+y2)-360);
				if (x2 == 0) x2=1;
				if (y2 == 0) y2=1;
				printf( " B %d %d %d %d %d %d \n", x1, x2, x1+x2, y1, y2, y1+y2 );
				Mat regionOfInterest2;
				regionOfInterest = frame( Rect( x1, y1, x2, y2 ));
				imshow( temp_window_name, regionOfInterest );
			}
		}
			  
	} 

	cvtColor( regionOfInterest, frame_gray3, CV_BGR2GRAY );
	equalizeHist( frame_gray3, frame_gray3 );
	four_cascade.detectMultiScale( frame_gray3, four, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10) );
	for( int i = 0; i < four.size(); i++ )
	{
		Point center( four[i].x + four[i].width*0.5, four[i].y + four[i].height*0.5 );
		ellipse( regionOfInterest, center, Size( four[i].width*0.5, four[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
		//Point center2( one[i].x, one[i].y );  std::vector<Rect> three;
		//ellipse( frame, center2, Size( 1, 1), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
		//regionOfInterest = frame( Rect( x1, y1+40, x2, y2-35 ));
		imshow( temp_window_name1, regionOfInterest );
	}
		  

	//while (false)		  ;
	t = (double)getTickCount() - t;
	printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency()); 
  //-- Show what you got
  imshow( window_name, frame );
  sizeROI.height = regionOfInterest.rows;
  sizeROI.width = regionOfInterest.cols;
  return regionOfInterest;
 }


void edgeAnalysis( Mat frameE, CvSize sizeE )
{
	IplImage *t1 = cvCreateImage( sizeE, IPL_DEPTH_8U, 3 );
	IplImage *t2 = cvCreateImage( sizeE, IPL_DEPTH_8U, 1 );
	IplImage *t3 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t4 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t5 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t6 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t7 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t8 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t9 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );
	IplImage *t0 = cvCreateImage( sizeE, IPL_DEPTH_16S, 1 );

 	*t1 = frameE;
	//cvKMeans2( t1, 4, t2, cvTermCriteria( CV_TERMCRIT_ITER, 2, 2 ));

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
	*/

	cvCvtColor( t1, t2, CV_RGB2GRAY );
	cvSobel( t2, t3, 1, 0, 3 );
	cvSobel( t2, t4, 1, 0, 5 );
	cvSobel( t2, t5, 1, 0, 7 );
	cvSobel( t2, t6, 1, 0, 9 );
	cvSobel( t2, t7, 2, 0, 7 );
	cvSobel( t2, t8, 2, 0, 9 );
	cvSobel( t2, t9, 1, 1, 7 );
	cvSobel( t2, t0, 1, 1, 9 );

	//cvConvertScale( t3, t4, 1.0/255.0, 0.0 );
	//cvThreshold( t4, t5, 126, 255, CV_THRESH_BINARY );
	//cvCanny( t5, t6, 10, 50, 3 );

	cvShowImage( "T1", t1 );
	cvShowImage( "T2", t2 );
	cvShowImage( "T3", t3 );
	cvShowImage( "T4", t4 );
	cvShowImage( "T5", t5 );
	cvShowImage( "T6", t6 );
	cvShowImage( "T7", t7 );
	cvShowImage( "T8", t8 );
	cvShowImage( "T9", t9 );
	cvShowImage( "T0", t0 );

	int key = cvWaitKey(0);
	if( key == 27 ) // �ESC�
		return;
	
	cvReleaseImage( &t1 );
	cvReleaseImage( &t2 );
	cvReleaseImage( &t3 );
	cvReleaseImage( &t4 );
	cvReleaseImage( &t5 );
	cvReleaseImage( &t6 );
	cvReleaseImage( &t7 );
	cvReleaseImage( &t8 );
	cvReleaseImage( &t9 );
	cvReleaseImage( &t0 );
	
}