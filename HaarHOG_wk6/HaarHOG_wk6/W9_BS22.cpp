//#include "cxcore.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include "../../utilities.h"
#include <opencv2/video/background_segm.hpp>


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 //Mat detectAndDisplay( Mat frame, CvSize size, CvSize &sizeROI, Point &ROI_TL );
 void IB_detectAndDisplay( Mat wholeImage, Mat bsImage, Mat bsMask, Mat &wholeROI, Mat &bsROI, Mat &bsMaskROI, Point &ROI_TL );
 void IB_colorSpaceAnalysis( Mat bsROI, Mat bsMask, Mat &bsKMeansROI ); 
 bool edgeAnalysis( Mat frameE, cv::Rect &location );
 Mat edgeThining( Mat frameET );
 int checkRightZero( Mat frameCR, int num, int col, int row );
 int checkRight255( Mat frameCR, int num, int col, int row );
 bool edgeCount( Mat &frameEC, cv::Rect &location );
 //bool edgeCheck( Mat frameEC, cv::Rect &location );
 void IB_BS( Mat BS_in, Mat &bsMask, Mat &BS_out );
 void hullPlotter( Mat src, vector<vector<Point>> contour, string name, int x, int y, Mat &mask );
 bool checkLeft( Mat drawing, int row, int col );	
 bool checkRight( Mat drawing, int row, int col );	
 bool checkUp( Mat drawing, int row, int col );	
 bool checkDown( Mat drawing, int row, int col );	
 void contourAmalgamator( vector<vector<Point>> contourIN, vector<vector<Point>> &contour );	


 /** Global variables */
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
 CvSize gSize;
 double totalFrames=0; 
 Mat average;
 Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

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
	//Mat ROIframe;
	Point ROI_TL;
	Rect location;
	Mat bsImage;
	Mat wholeROI;
	Mat bsROI;
	Mat bsKMeansROI;
	Mat bsMask;
	Mat bsMaskROI;

	CvCapture* capture;
	char filename[100];
	//sprintf(filename, "F:/M4H00707.MP4.AVI");
	sprintf(filename, "C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Recordings/M4H00717.MP4.AVI");
	capture = cvCaptureFromAVI(filename);
	pMOG2 = new BackgroundSubtractorMOG2();
	int frameCount=0;
	int modelBuildCount=0;
	int bugger=0;


   if( capture )
   {
     while( true )
     {
		if( !( frameCount == 3 ) || modelBuildCount > 0 )	{
			wholeImage = cvQueryFrame( capture );
			frameCount++;
		} else {
			modelBuildCount++;
			cout << "#" << modelBuildCount << "  Building Model!" << endl;
		}
		cout << bugger++ << endl;
		if( bugger == 52 )	{
			cout << "fail" << endl;
		}
		//Mat wholeImage2 = wholeImage.clone();
		//bsImage = wholeImage;
		IB_BS( wholeImage, bsMask, bsImage );
		IB_detectAndDisplay( wholeImage, bsImage, bsMask, wholeROI, bsROI, bsMaskROI, ROI_TL );
		if( wholeROI.empty() || frameCount == 3 )	{
			continue;
		}
		//IB_colorSpaceAnalysis( bsROI, bsMaskROI, bsKMeansROI );
		//if( edgeAnalysis( bsKMeansROI, location ) )	{
		//	rectangle( wholeImage, Point( ROI_TL+location.tl() ), Point( ROI_TL+location.br() ), Scalar( 0, 255, 0 ), 3 );;
		//}

		//colorSpaceAnalysis( ROIframe, sizeROI );
		if( edgeAnalysis( bsROI, location ) )	{
			rectangle( wholeImage, Point( ROI_TL+location.tl() ), Point( ROI_TL+location.br() ), Scalar( 0, 255, 0 ), 3 );;
		}
		//imshow( "OUTPUT2", wholeImage2 );
		cv::imshow( "OUTPUT", wholeImage );
		cv::moveWindow( "OUTPUT", 0, 0 );
		cv::imshow( "Background Subtraction", bsImage );
		cv::moveWindow( "Background Subtraction", 650, 0 );
		
		Mat wholeROI2 = wholeROI;
		Mat bsROI2 = bsROI;
		Mat bsKMeansROI2 = bsKMeansROI;
		resize( wholeROI2, wholeROI2, Size( 5*wholeROI.cols, 5*wholeROI.rows ) );
		resize( bsROI2, bsROI2, Size( 5*bsROI.cols, 5*bsROI.rows ) );
		//resize( bsKMeansROI2, bsKMeansROI2, Size( 5*bsKMeansROI.cols, 5*bsKMeansROI.rows ) );
		cv::imshow( "Region of Interest", wholeROI2 );
		cv::moveWindow( "Region of Interest", 0, 400 );
		cv::imshow( "B.S. R.O.I.", bsROI2 );
		cv::moveWindow( "B.S. R.O.I.", 410, 400 );
		//cv::imshow( "KMeans ROI", bsKMeansROI2 );
		//cv::moveWindow( "KMeans ROI", 820, 400 );

		int key = cvWaitKey(0);
		if( key == 27 ) // ‘ESC’
			break;
	 }
   }

   wholeROI.release();
   bsROI.release();
 }

 void IB_colorSpaceAnalysis( Mat bsROI, Mat bsMask, Mat &bsKMeansROI )
 {	

	Vec3b new_pixel[8] = {
		Vec3b( 255, 0, 0 ),
		Vec3b( 0, 255, 0 ),
		Vec3b( 0, 0, 255 ),
		Vec3b( 255, 255, 0 ),
		Vec3b( 255, 0, 255 ),
		Vec3b( 0, 255, 255 ),
		Vec3b( 0, 0, 0 ),
		Vec3b( 255, 255, 255 ) };

	Mat frameBGR = bsROI.clone();
	Mat frameXYZ;
	Mat frameYBR;
	Mat frameHSV;
	Mat frameHLS;
	Mat frameLAB;
	Mat frameLUV;

	int cluster_count = 3;
	/*
	cvtColor( frameBGR, frameXYZ, CV_BGR2XYZ, 0 );
	cvtColor( frameBGR, frameYBR, CV_BGR2YCrCb, 0 );
	cvtColor( frameBGR, frameHSV, CV_BGR2HSV, 0 );
	cvtColor( frameBGR, frameHLS, CV_BGR2HLS, 0 );
	cvtColor( frameBGR, frameLAB, CV_BGR2Lab, 0 );
	cvtColor( frameBGR, frameLUV, CV_BGR2Luv, 0 );
	*/
		
	Mat tempPointsBGR = frameBGR.reshape( 1, frameBGR.cols*frameBGR.rows );
	Mat tempPositionBGR = Mat( frameBGR.cols*frameBGR.rows, 2, CV_8UC1 );
	Mat pointsBGR;
	Mat positionBGR = Mat( frameBGR.cols*frameBGR.rows, 2, CV_8UC1 );
	int r1 = 0;
	int r2 = 0;
	for(int row=0; row<frameBGR.rows; row++)	{ 
		for(int col=0; col<frameBGR.cols; col++) 
		{
			if( bsMask.at<uchar>(row, col) != 0 )	{
				tempPointsBGR.at<uchar>( r2, 0 ) = frameBGR.at<Vec3b>(row, col)[0];
				tempPointsBGR.at<uchar>( r2, 1 ) = frameBGR.at<Vec3b>(row, col)[1];
				tempPointsBGR.at<uchar>( r2, 2 ) = frameBGR.at<Vec3b>(row, col)[2];
				tempPositionBGR.at<uchar>( r2, 0 ) = row;
				tempPositionBGR.at<uchar>( r2, 1 ) = col;
				r2++;
			}
			positionBGR.at<uchar>( r1, 0 ) = row;
			positionBGR.at<uchar>( r1, 1 ) = col;
			r1++;
			/*
			frameBGR.at<Vec3b>(row, col) = new_pixel[ clustersBGR.at<int>(r2, 0) ];		
			frameXYZ.at<Vec3b>(row, col) = new_pixel[ clustersXYZ.at<int>(r2, 0) ];
			frameYBR.at<Vec3b>(row, col) = new_pixel[ clustersYBR.at<int>(r2, 0) ];
			frameHSV.at<Vec3b>(row, col) = new_pixel[ clustersHSV.at<int>(r2, 0) ];
			frameHLS.at<Vec3b>(row, col) = new_pixel[ clustersHLS.at<int>(r2, 0) ];
			frameLAB.at<Vec3b>(row, col) = new_pixel[ clustersLAB.at<int>(r2, 0) ];
			frameLUV.at<Vec3b>(row, col) = new_pixel[ clustersLUV.at<int>(r2, 0) ];
			*/
		}
	}
	if( r2 == 0 )	{
		pointsBGR = frameBGR.reshape( 1, frameBGR.cols*frameBGR.rows );
		r2 = r1;
	} else {
		pointsBGR = tempPointsBGR.rowRange( 0, r2 );
		positionBGR = tempPositionBGR.rowRange( 0, r2 );
	}
	/*
	Mat pointsXYZ = frameXYZ.reshape( 1, frameBGR.cols*frameBGR.rows );
	Mat pointsYBR = frameYBR.reshape( 1, frameBGR.cols*frameBGR.rows );
	Mat pointsHSV = frameHSV.reshape( 1, frameBGR.cols*frameBGR.rows );
	Mat pointsHLS = frameHLS.reshape( 1, frameBGR.cols*frameBGR.rows );
	Mat pointsLAB = frameLAB.reshape( 1, frameBGR.cols*frameBGR.rows );
	Mat pointsLUV = frameLUV.reshape( 1, frameBGR.cols*frameBGR.rows );
	*/

	Mat clustersBGR, centersBGR;
	Mat clustersXYZ, centersXYZ;
	Mat clustersYBR, centersYBR;
	Mat clustersHSV, centersHSV;
	Mat clustersHLS, centersHLS;
	Mat clustersLAB, centersLAB;
	Mat clustersLUV, centersLUV;
	
	pointsBGR.convertTo(pointsBGR, CV_32FC3, 1.0/255.0);
	/*
	pointsXYZ.convertTo(pointsXYZ, CV_32FC3, 1.0/255.0);
	pointsYBR.convertTo(pointsYBR, CV_32FC3, 1.0/255.0);
	pointsHSV.convertTo(pointsHSV, CV_32FC3, 1.0/255.0);
	pointsHLS.convertTo(pointsHLS, CV_32FC3, 1.0/255.0);
	pointsLAB.convertTo(pointsLAB, CV_32FC3, 1.0/255.0);
	pointsLUV.convertTo(pointsLUV, CV_32FC3, 1.0/255.0);
	*/

	//pointsXYZ = pointsXYZ.colRange( 0, 1 );
	//pointsYBR = pointsYBR.colRange( 1, 2 );
	//pointsHSV = pointsHSV.col( 0 );
	//pointsHLS = pointsHLS.col( 0 );
	//pointsLAB = pointsLAB.colRange( 1, 2 );
	//pointsLUV = pointsLUV.colRange( 1, 2 );

	kmeans( pointsBGR, cluster_count, clustersBGR,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersBGR );
	/*
	kmeans( pointsXYZ, cluster_count, clustersXYZ,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersXYZ );
	kmeans( pointsYBR, cluster_count, clustersYBR,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersYBR );
	kmeans( pointsHSV, cluster_count, clustersHSV,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersHSV );
	kmeans( pointsHLS, cluster_count, clustersHLS,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersHLS );
	kmeans( pointsLAB, cluster_count, clustersLAB,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersLAB );
	kmeans( pointsLUV, cluster_count, clustersLUV,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersLUV );
		*/

	frameBGR.setTo( Scalar() );
	/*
	frameXYZ.setTo( Scalar() );
	frameYBR.setTo( Scalar() );
	frameHSV.setTo( Scalar() );
	frameHLS.setTo( Scalar() );
	frameLAB.setTo( Scalar() );
	frameLUV.setTo( Scalar() );
	*/

	int row=0, col=0;
	for(int r3=0; r3<r2; r3++) 
	{
		row = positionBGR.at<uchar>( r3, 0 );
		col = positionBGR.at<uchar>( r3, 1 );
		frameBGR.at<Vec3b>(row, col) = new_pixel[ clustersBGR.at<int>(r3, 0) ];
		
		/*	
		tempPointsBGR.at<uchar>( frameBGR.rows + (frameBGR.rows+1)*frameBGR.cols, 0 ) = frameBGR.at<Vec3b>(row, col)[0];
		tempPointsBGR.at<uchar>( frameBGR.rows + (frameBGR.rows+1)*frameBGR.cols, 1 ) = frameBGR.at<Vec3b>(row, col)[1];
		tempPointsBGR.at<uchar>( frameBGR.rows + (frameBGR.rows+1)*frameBGR.cols, 2 ) = frameBGR.at<Vec3b>(row, col)[2];
		tempPositionBGR.at<int>( frameBGR.rows + (frameBGR.rows+1)*frameBGR.cols, 0 ) = row;
		tempPositionBGR.at<int>( frameBGR.rows + (frameBGR.rows+1)*frameBGR.cols, 1 ) = col;
		rowCount++;
		r2++;
		frameBGR.at<Vec3b>(row, col) = new_pixel[ clustersBGR.at<int>(r2, 0) ];		
		frameXYZ.at<Vec3b>(row, col) = new_pixel[ clustersXYZ.at<int>(r2, 0) ];
		frameYBR.at<Vec3b>(row, col) = new_pixel[ clustersYBR.at<int>(r2, 0) ];
		frameHSV.at<Vec3b>(row, col) = new_pixel[ clustersHSV.at<int>(r2, 0) ];
		frameHLS.at<Vec3b>(row, col) = new_pixel[ clustersHLS.at<int>(r2, 0) ];
		frameLAB.at<Vec3b>(row, col) = new_pixel[ clustersLAB.at<int>(r2, 0) ];
		frameLUV.at<Vec3b>(row, col) = new_pixel[ clustersLUV.at<int>(r2, 0) ];
		*/
	}

	//resize( bsROI, bsROI, Size( 5*bsROI.cols, 5*bsROI.rows ) );
	//resize( frameBGR, frameBGR, Size( 5*frameBGR.cols, 5*frameBGR.rows ) );
	/*
	resize( frameXYZ, frameXYZ, Size( 5*frameXYZ.cols, 5*frameXYZ.rows ) );
	resize( frameYBR, frameYBR, Size( 5*frameYBR.cols, 5*frameYBR.rows ) );
	resize( frameHSV, frameHSV, Size( 5*frameHSV.cols, 5*frameHSV.rows ) );
	resize( frameHLS, frameHLS, Size( 5*frameHLS.cols, 5*frameHLS.rows ) );
	resize( frameLAB, frameLAB, Size( 5*frameLAB.cols, 5*frameLAB.rows ) );
	resize( frameLUV, frameLUV, Size( 5*frameLUV.cols, 5*frameLUV.rows ) );
	*/

	//imshow( "Upper Torso", bsROI );
	//imshow( "BGR", frameBGR );
	/*
	imshow( "XYZ", frameXYZ );
	imshow( "YBR", frameYBR );
	imshow( "HSV", frameHSV );
	imshow( "HLS", frameHLS );
	imshow( "LAB", frameLAB );
	imshow( "LUV", frameLUV );
	*/

	bsKMeansROI = frameBGR.clone();
}

/** @function detectAndDisplay */
//Mat detectAndDisplay( Mat frame, CvSize size, CvSize &sizeROIreturn, Point &ROI_TLreturn )
void IB_detectAndDisplay( Mat wholeImage, Mat bsImage, Mat bsMask, Mat &wholeROI, Mat &bsROI, Mat &bsMaskROI, Point &ROI_TL )
{
  std::vector<Rect> one;
  std::vector<Rect> two;
  std::vector<Rect> three;
  std::vector<Rect> four;
  Mat frame_gray;
  Mat frame_gray2;
  Mat frame_gray3;
  Mat regionOfInterest;
  Mat regionOfReturn;
  Mat regionOfReturn2;
  Mat wholeImageT = wholeImage.clone();
  Mat bsImageT = bsImage.clone();
  Mat bsMaskT = bsMask.clone();
 

  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  fflush(stdout);
	vector<Rect> found, found_filtered;
	double t = (double)getTickCount();
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(wholeImage, found, 0, Size(8,8), Size(0,0), 1.05, 2);

	if( found.size() == 0 )	{
		wholeROI = NULL;
		bsROI = NULL;
		return;
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
		rectangle(wholeImage, r.tl(), r.br(), cv::Scalar(0,0,255), 3);

		
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
		//frame_gray2 = wholeImage( Rect( x1, y1, x2, y2 ));
		x1 = x1 + x2/8;
		x2 = 3*x2/4;
		y1 = y1 + y2/2;
		y2 = y2/2;
		wholeROI = wholeImageT( Rect( x1, y1, x2, y2 ));
		bsROI = bsImageT( Rect( x1, y1, x2, y2 ));
		bsMaskROI = bsMaskT( Rect( x1, y1, x2, y2 ));
		ROI_TL = Point( x1, y1 );

		return;

		/*

		//sizeROIreturn.height = regionOfInterest.rows;
		//sizeROIreturn.width = regionOfInterest.cols;

		if( gSize.height == 0 )	{
			gSize.height = wholeImage.rows;
			gSize.width = wholeImage.cols;
			
			//regionOfReturn2 = regionOfInterest;
			//average = regionOfInterest.clone();
			//totalFrames++;
		}
		//regionOfReturn = regionOfReturn2;
		//resize( regionOfInterest, regionOfReturn2, gSize );
		//sizeROIreturn.height = regionOfReturn2.rows;
		//sizeROIreturn.width = regionOfReturn2.cols;
		//totalFrames++;
		//double A = 1/totalFrames;
		//double B = (totalFrames-1)/totalFrames;
		//addWeighted( regionOfReturn2, 1/2, average, 1/2, 0.0, average );
		//addWeighted( regionOfReturn2, A, average, B, 0.0, average );
		//return average;

		//return regionOfInterest;
	 
		cvtColor( frame_gray2, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );
		  //-- Detect faces
		  one_cascade.detectMultiScale( frame_gray, one, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20, 20) );
		  
		  for( int i = 0; i < one.size(); i++ )	{
			imshow( temp_window_name, regionOfInterest );
			Point center( one[i].x + one[i].width*0.5, one[i].y + one[i].height*0.5 );
			ellipse( frame, center, Size( one[i].width*0.5, one[i].height*0.5 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
			regionOfReturn = regionOfInterest;
		  }
		*/
	}

	/*
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
				Point center( (two[k].x + two[k].width*0.5 + three[i].x + three[i].width*0.5)/2, (two[k].y + two[k].height*0.5 + three[i].y + three[i].height*0.5)/2 );
				ellipse( frame, center, Size( (two[k].width*0.5+three[i].width*0.5)/2, (two[k].height*0.5+three[i].height*0.5)/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
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
				regionOfReturn = frame( Rect( x1, y1, x2, y2 ));
				ROI_TLreturn = Point( x1, y1 );
				//imshow( temp_window_name, regionOfInterest );
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
  //imshow( window_name, frame );
  //sizeROI.height = regionOfReturn.rows;
  //sizeROI.width = regionOfReturn.cols;
  if( gSize.height == 0 )	{
	  gSize.height = regionOfReturn.rows;
	  gSize.width = regionOfReturn.cols;
  }
  resize( regionOfReturn, regionOfReturn2, gSize );
  //sizeROIreturn.height = regionOfReturn2.rows;
  sizeROIreturn.width = regionOfReturn2.cols;
  //return regionOfReturn2;
  */
 }

bool edgeAnalysis( Mat frameE, cv::Rect &location )
{
	CvSize sizeE = cvSize( frameE.cols, frameE.rows );
	Mat t1;// = Mat( sizeE, CV_8UC3 );
	Mat t2;// = Mat( sizeE, CV_8UC1 );
	Mat t3 = Mat( sizeE, CV_8UC1 );
	Mat t4;// = Mat( sizeE, CV_16SC1 );
	Mat t5;// = Mat( sizeE, CV_16SC1 );
	Mat t6;// = Mat( sizeE, CV_8UC1 );
	Mat t7;// = Mat( sizeE, CV_32FC1 );
	Mat t8;// = Mat( sizeE, CV_16SC1 );
	Mat t9;// = Mat( sizeE, CV_16SC1 );
	Mat t0 = Mat( sizeE, CV_8UC1 );

 	//t1 = frameE;
	//cvKMeans2( t1, 4, t2, cvTermCriteria( CV_TERMCRIT_ITER, 2, 2 ));

	cvtColor( frameE, t2, CV_RGB2GRAY );

	Mat t2A, t2B, t2C, t7A, t7B, t7C, t7D, t7E;
	t2A = t2.clone();
	t2B = t2.clone();
	t2C = t2.clone();
	t2.convertTo( t4, CV_8UC1 );
	t2.convertTo( t2A, CV_8U );
	t2.convertTo( t2B, CV_16S, 256 );
	t2.convertTo( t2C, CV_32F, 1/256 );
	Canny( t4, t3, 100, 200, 3, true );
	Laplacian( t4, t5, CV_8U, 3 ); 

	
	Sobel( t2, t7, CV_16S, 1, 0, 7 );
	Sobel( t2A, t7A, CV_16S, 1, 0, 7 );
	Sobel( t2B, t7B, CV_32F, 1, 0, 7 );
	Sobel( t2C, t7C, CV_64F, 1, 0, 7 );
	Sobel( t2A, t7D, CV_64F, 1, 0, 7 );
	Sobel( t2A, t7E, CV_32F, 1, 0, 7 );
	//cout << kx << "£" << ky << endl;
	inRange( t7, -15000, 32767, t8 );
	inRange( t7, -32768, 15000, t9 );
	//t7.convertTo( t7, CV_8UC1 );
	for( int row=0; row<t0.rows; row++ )	{
		for( int col=0; col<t0.cols; col++ )	{
			if( t8.at<uchar>(row,col) == 0 )	{
				t0.at<uchar>(row,col) = 0;
			} else if( t9.at<uchar>(row,col) == 0 ) {
				t0.at<uchar>(row,col) = 255;
			} else {
				t0.at<uchar>(row,col) = 128;
			}
		}
	}
	//Canny( t0, t0, 200, 100 );
	t1 = t0.clone();
	t1 = edgeThining( t1 );

	/*
	imshow( "2A", t2A );
	imshow( "2B", t2B );
	imshow( "2C", t2C );
	imshow( "7A", t7A );
	imshow( "7B", t7B );
	imshow( "7C", t7C );
	imshow( "7D", t7D );
	imshow( "7E", t7E );
	*/
	
	
	//resize( frameE, frameE, Size( 5*frameE.cols, 5*frameE.rows ) );
	//imshow( "UPPER TORSO REGION", frameE );
	//resize( t2, t2, Size( 5*t2.cols, 5*t2.rows ) );
	//imshow( "GRAY SCALE", t2 );
	//t7.convertTo( t7, CV_8UC1 );
	resize( t7, t7, Size( 5*t7.cols, 5*t7.rows ) );
	imshow( "GRADIENT", t7 );
	moveWindow( "GRADIENT", 0, 700 ); 
	resize( t0, t0, Size( 5*t0.cols, 5*t0.rows ) );
	imshow( "EXTRACTED EDGES", t0 );
	moveWindow( "EXTRACTED EDGES", 400, 700 );
	//resize( t1, t1, Size( 5*t1.cols, 5*t1.rows ) );
	//imshow( "THINNED EDGES", t1 );
	resize( t3, t3, Size( 5*t3.cols, 5*t3.rows ) );
	imshow( "Canny", t3 );
	moveWindow( "Canny", 1400, 0 );
	resize( t5, t5, Size( 5*t5.cols, 5*t5.rows ) );
	imshow( "Laplacian", t5 );
	//imshow( "T6", t6 );
	//imshow( "T7", t7 );
	//imshow( "T8", t8 );
	//imshow( "T9", t9 );
	//imshow( "T0", t0 );

	if( edgeCount( t1, location ) )	{
		//rectangle( t1, Point( location.tl() ), Point( location.br() ), Scalar( 0, 255, 0 ), 3 );
		//resize( t1, t1, Size( 5*t1.cols, 5*t1.rows ) );
		//imshow( "DETECTED POINTS", t1 );
		return true;
	} else {
		return false;
	}

}
	
Mat edgeThining( Mat frameET )
{
	for( int row=0; row<frameET.rows; row++ )	{
		for( int col=0; col<frameET.cols; col++ )	{
			if( frameET.at<uchar>(row,col) == 0 )	{
				checkRightZero( frameET, 0, col, row );
			} else if( frameET.at<uchar>(row,col) == 255 ) {
				checkRight255( frameET, 0, col, row );
			} 
		}
	}
	return frameET;
}

int checkRightZero( Mat frameCR, int num, int col, int row )
{
	int val = num;
	if( col >= frameCR.cols )	{
		return num;
	}
	if( frameCR.at<uchar>(row,col) == 0 )	{
		num = checkRightZero( frameCR, num+1, col+1, row );
	} 
	if( num != 0 && val != num/2)	{
		frameCR.at<uchar>(row,col) = 128;
	}
	return num;
}

int checkRight255( Mat frameCR, int num, int col, int row )
{
	int val = num;
	if( col >= frameCR.cols )	{
		return num;
	} 
	if( frameCR.at<uchar>(row,col) == 255 )	{
		num = checkRight255( frameCR, num+1, col+1, row );
	} 
	if( num != 0 && val != num/2)	{
		frameCR.at<uchar>(row,col) = 128;
	}
	return num;
}

bool edgeCount( Mat &frameEC, cv::Rect &location )
{
	Mat frameRDD;
	//frameEC.convertTo( frameRDD, CV_8UC3 );
	cvtColor( frameEC, frameRDD, CV_GRAY2RGB );
	int strapLocation[10][2][3] = {0};
	bool strapOrientation[10][2][3] = {false};
	int rowStep = frameEC.rows/10;
	int lRow=rowStep, leftAbscent=0, rightAbscent=0;
	location.x = frameEC.cols;
	location.width = 0;
	location.y = frameEC.rows;
	location.height = 0;
	for( int i=0; i<9; i++ )	{
		int temp[2] = {0};
		bool ori[2] = {false};
		bool rep[2] = {false};
		for( int lCol=0; lCol<frameEC.cols; lCol++ )	{
			if( frameEC.at<uchar>(lRow,lCol) == 0 )	{
				if( !rep[0] )	{
					if( lCol < location.x)	location.x = lCol;
					temp[0] = lCol;
					rep[0] = true;
					frameRDD.at<Vec3b>(lRow,lCol) = Vec3b( 255, 0, 0 );
				} else if( !rep[1] )	{
					temp[1] = lCol;
					rep[1] = true;
					frameRDD.at<Vec3b>(lRow,lCol) = Vec3b( 0, 255, 0 );
				} else if( ori[0] == true || ori[1] == true )	{
					if( lCol > location.width)	location.width = lCol;
					if( lRow < location.y)	location.y = lRow;
					if( lCol > location.height)	location.height = lRow;
					if( temp[1] < frameEC.cols/2 )	{
						if( ori[1] = true )	{
							strapLocation[i][0][0] = temp[0];
							strapLocation[i][0][1] = temp[1];
							strapLocation[i][0][2] = lCol;
							strapOrientation[i][0][0] = ori[0];
							strapOrientation[i][0][1] = ori[1];
							strapOrientation[i][0][2] = false;
						}
					} else {
						if( ori[0] != ori[1] )	{
							strapLocation[i][1][0] = temp[0];
							strapLocation[i][1][1] = temp[1];
							strapLocation[i][1][2] = lCol;
							strapOrientation[i][1][0] = ori[0];
							strapOrientation[i][1][1] = ori[1];
							strapOrientation[i][1][2] = false;
						}
					}
					temp[0] = temp[1] = 0;
					rep[0] = rep[1] = false;
					ori[0] = ori[1] = false;
					frameRDD.at<Vec3b>(lRow,lCol) = Vec3b( 0, 0, 255 );
				} else {
					temp[0] = temp[1] = 0;
					rep[0] = rep[1] = false;
					ori[0] = ori[1] = false;
				}
			} else if( frameEC.at<uchar>(lRow,lCol) == 255 )	{
				if( !rep[0] )	{
					if( lCol < location.x)	location.x = lCol;
					temp[0] = lCol;
					ori[0] = true;
					rep[0] = true;
					frameRDD.at<Vec3b>(lRow,lCol) = Vec3b( 0, 255, 255 );
				} else if( !rep[1] )	{
					temp[1] = lCol;
					ori[1] = true;
					rep[1] = true;
					frameRDD.at<Vec3b>(lRow,lCol) = Vec3b( 255, 255, 0 );
				} else if( ori[0] == false || ori[1] == false )	{
					if( lCol > location.width)	location.width = lCol;
					if( lRow < location.y)	location.y = lRow;
					if( lCol > location.height)	location.height = lRow;
					if( temp[1] < frameEC.cols/2 )	{
						if( ori[1] == false )	{
							strapLocation[i][0][0] = temp[0];
							strapLocation[i][0][1] = temp[1];
							strapLocation[i][0][2] = lCol;
							strapOrientation[i][0][0] = ori[0];
							strapOrientation[i][0][1] = ori[1];
							strapOrientation[i][0][2] = true;
						}
					} else {
						if( ori[0] != ori[1] )	{
							strapLocation[i][1][0] = temp[0];
							strapLocation[i][1][1] = temp[1];
							strapLocation[i][1][2] = lCol;
							strapOrientation[i][1][0] = ori[0];
							strapOrientation[i][1][1] = ori[1];
							strapOrientation[i][1][2] = true;
						}
					}
					temp[0] = temp[1] = 0;
					rep[0] = rep[1] = false;
					ori[0] = ori[1] = false;
					frameRDD.at<Vec3b>(lRow,lCol) = Vec3b( 255, 0, 255 );
				} else {
					temp[0] = temp[1] = 0;
					rep[0] = rep[1] = false;
					ori[0] = ori[1] = false;
				}
			}
		}	
		if( strapLocation[i][0][1] == 0 )	{
			leftAbscent++;
		} 
		if( strapLocation[i][1][1] == 0 )	{
			rightAbscent++;
		} 
		lRow = lRow + rowStep;
	}
	location.width = location.width - location.x;
	location.height = location.height - location.y;
	resize( frameRDD, frameRDD, Size( 5*frameRDD.cols, 5*frameRDD.rows ) );
	imshow( "DETECTED POINTS", frameRDD );
	moveWindow( "DETECTED POINTS", 800, 700 );
	if( leftAbscent > 5 || rightAbscent > 5 )	{
		return false;
	} else {
		rectangle( frameRDD, Point( 5*location.tl() ), Point( 5*location.br() ), Scalar( 0, 255, 0 ), 1 );
		imshow( "DETECTED POINTS", frameRDD );
		return true;
	}
	return false;
}

void IB_BS( Mat BS_in, Mat &bsMask, Mat &BS_out )
{
	//BS_out = BS_out.zeros;
	BS_out.setTo( Scalar() );
	Mat BS_mask, BS_C_mask;
	pMOG2->operator()(BS_in, BS_mask, -1 );
	threshold( BS_mask, BS_mask, 250, 255, THRESH_BINARY );
	morphologyEx( BS_mask, BS_mask, MORPH_CLOSE, Mat(), Point(-1,-1), 4 );
	morphologyEx( BS_mask, BS_mask, MORPH_OPEN, Mat(), Point(-1,-1), 1 );
	vector<vector<Point>> c0, a0;
	vector<Vec4i> h0;
	findContours( BS_mask, c0, h0, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
	cvtColor( BS_mask, BS_C_mask, CV_GRAY2BGR );
	contourAmalgamator( c0, a0 );
	drawContours( BS_mask, a0, -1, uchar( 255 ), CV_FILLED );
	Mat BS_D_mask = BS_mask.clone();
	hullPlotter( BS_C_mask, c0, "HULL FOUR", 1300, 250, BS_D_mask );
	hullPlotter( BS_C_mask, a0, "HULL A FOUR", 1300, 650, BS_mask );
	//drawContours( BS_C_mask, c0, -1, Scalar( 0, 0, 255 ), CV_FILLED );
	BS_in.copyTo( BS_out, BS_mask );
	bsMask = BS_mask;
}

void hullPlotter( Mat src, vector<vector<Point>> contour, string name, int x, int y, Mat &mask )
{
	   /// Find the convex hull object for each contour
	   vector<vector<Point> >hull( contour.size() );
	   RotatedRect *rArray = new RotatedRect[ contour.size() ];
	   int *rOri = new int[ contour.size() ];
	   float *rE = new float[ contour.size() ];
	   for( int i = 0; i < contour.size(); i++ )	{  
		   if( contour[i].empty() )	{
			   continue;
		   }
		   rArray[i] = minAreaRect( Mat(contour[i]) );
		   convexHull( Mat(contour[i]), hull[i], false ); 
			Point2f tp[4]; 
			rArray[i].points( tp );
		   //cout << tp << endl;
		   rE[i] = ( tp[1].x - tp[0].x ) / ( tp[3].y - tp[0].y );
		   if( rE[i] > 1 )	{
			    rE[i] = ( tp[3].y - tp[0].y ) / ( tp[1].x - tp[0].x );
		   }
	   }

	   
	   /// Draw contours + hull results
	   Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
	   Mat rdrawing = Mat::zeros( src.size(), CV_8UC3 );
	   for( int i = 0; i< contour.size(); i++ )
		  {
			Point2f tp[4]; 
			rArray[i].points( tp );
			line( rdrawing, tp[0], tp[1], Scalar( 255, 255, 0 ) );
			line( rdrawing, tp[1], tp[2], Scalar( 255, 255, 0 ) );
			line( rdrawing, tp[2], tp[3], Scalar( 255, 255, 0 ) );
			line( rdrawing, tp[3], tp[0], Scalar( 255, 255, 0 ) );
			//Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, hull, i, Scalar( 255, 255, 255 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			//cvtColor( drawing, drawing, CV_BGR2GRAY );
			//dilate( drawing, drawing, Mat(), Point(-1, -1), 1 ); 
			//cvtColor( drawing, drawing, CV_GRAY2BGR );
			drawContours( drawing, contour, i, Scalar( 0, 0, 0 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, hull, i, Scalar( 0, 255, 0 ), 1, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, contour, i, Scalar( 0, 0, 255 ), 1, 8, vector<Vec4i>(), 0, Point() );
		  }


	   /// Draw contours + hull results
	   //Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
	   //Mat rdrawing = Mat::zeros( src.size(), CV_8UC3 );
	   for( int i = 0; i< contour.size(); i++ )
		  {
			Point2f tp[4]; 
			rArray[i].points( tp );
			line( rdrawing, tp[0], tp[1], Scalar( 0, 255, 0 ) );
			line( rdrawing, tp[1], tp[2], Scalar( 0, 255, 0 ) );
			line( rdrawing, tp[2], tp[3], Scalar( 0, 255, 0 ) );
			line( rdrawing, tp[3], tp[0], Scalar( 0, 255, 0 ) );
			//Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, hull, i, Scalar( 255, 255, 255 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			//cvtColor( drawing, drawing, CV_BGR2GRAY );
			//dilate( drawing, drawing, Mat(), Point(-1, -1), 1 ); 
			//cvtColor( drawing, drawing, CV_GRAY2BGR );
			drawContours( drawing, contour, i, Scalar( 0, 0, 0 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, hull, i, Scalar( 0, 255, 0 ), 1, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, contour, i, Scalar( 0, 0, 255 ), 1, 8, vector<Vec4i>(), 0, Point() );
		  }

	   int tt0=0;
	   for( int col=0; col<drawing.cols; col++ )	{
		   for( int row=0; row<drawing.rows; row++ )	{
			   if( drawing.at<Vec3b>(row, col) == Vec3b( 255, 255, 255 ) )	{
				   tt0=0;
				   
				   if( checkLeft( drawing, row, col ) )	{
					   tt0++;
				   }
				   if( checkRight( drawing, row, col ) )	{
					   tt0++;
				   }
				   /*
				   if( checkUp( drawing, row, col ) )	{
					   tt0++;
				   }
				   if( checkDown( drawing, row, col ) )	{
					   tt0++;
				   }
				   */
				   if( tt0 == 2 )	{
					   drawing.at<Vec3b>(row, col) = Vec3b( 255, 0, 0 );
					   mask.at<uchar>(row, col) = uchar(255);
				   } 
			   }
		   }
	   }

		/// Show in a window
		namedWindow( name, CV_WINDOW_AUTOSIZE );
		imshow( name, drawing );
		moveWindow( name, x, y );

}

bool checkLeft( Mat drawing, int row, int col )	
{
	col--;
	if( col < 0 )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 255 && drawing.at<Vec3b>(row, col)[2] == 0 )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 0 && drawing.at<Vec3b>(row, col)[2] == 255 )	{
		return true;
	}
	return checkLeft( drawing, row, col );
}

bool checkRight( Mat drawing, int row, int col )	
{
	col++;
	if( col >= drawing.cols )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 255 && drawing.at<Vec3b>(row, col)[2] == 0 )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 0 && drawing.at<Vec3b>(row, col)[2] == 255 )	{
		return true;
	}
	return checkRight( drawing, row, col );
}

bool checkUp( Mat drawing, int row, int col )	
{
	row++;
	if( row < 0 )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 255 && drawing.at<Vec3b>(row, col)[2] == 0 )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 0 && drawing.at<Vec3b>(row, col)[2] == 255 )	{
		return true;
	}
	return checkUp( drawing, row, col );
}

bool checkDown( Mat drawing, int row, int col )	
{
	row--;
	if( row >= drawing.rows )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 255 && drawing.at<Vec3b>(row, col)[2] == 0 )	{
		return false;
	}
	if( drawing.at<Vec3b>(row, col)[1] == 0 && drawing.at<Vec3b>(row, col)[2] == 255 )	{
		return true;
	}
	return checkDown( drawing, row, col );
}

void contourAmalgamator( vector<vector<Point>> contourIN, vector<vector<Point>> &contour )	
{
	contour = contourIN;
	RotatedRect *rArray = new RotatedRect[ contour.size() ];
	Rect *bArray = new Rect[ contour.size() ];
	for( int i = 0; i < contour.size(); i++ )	{  
		rArray[i] = minAreaRect( Mat(contour[i]) );
		bArray[i] = boundingRect( Mat(contour[i]) );
	}
	
	for( int i = 0; i < contour.size(); i++ )	{
		for( int j = 0; j < contour.size(); j++ )	{
			if( i <= j )	{
				continue;
			}
			//if( ( ( rArray[i].angle > rArray[j].angle-5 ) && ( rArray[i].angle < rArray[j].angle+5 ) )
			if( ( ( abs( rArray[i].center.x - rArray[j].center.x ) < 0.6*bArray[i].width ) || ( abs( rArray[i].center.x - rArray[j].center.x ) < 0.6*bArray[j].width ) )
			&& ( ( abs( rArray[i].center.y - rArray[j].center.y ) < 0.6*bArray[i].height ) || ( abs( rArray[i].center.y - rArray[j].center.y ) < 0.6*bArray[j].height ) ) )	{
			//if( true )	{
				while( !contour[j].empty() )	{
					Point temp;
					temp = contour[j].back();
					contour[j].pop_back();
					contour[i].push_back( temp );
				}
			}
		}
	}
}
