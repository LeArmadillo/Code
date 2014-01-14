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
 void IB_BS( bool active, Mat BS_in, Mat &bsMask, Mat &BS_out );
 void IB_detectAndDisplay( bool active, Mat wholeImage, Mat bsImage, Mat bsMask, Mat &wholeROI, Mat &bsROI, Mat &bsMaskROI, Point &ROI_TL );
 
 void IB_colorSpaceAnalysis( bool active, Mat bsROI, Mat bsMask, Mat &bsKMeansROI );
 bool IB_colourClassification( Rect &loc1, float &con1, Mat bsKMeansROI, Rect &location );
 bool helper_colourClassification( string name, Mat chan, bool *symetric, vector<vector<Point>> contours, bool h, Rect &location );
 
 bool IB_edgeAnalysis( Mat frameE, cv::Rect &location );
 Mat edgeThining( Mat frameET );
 int checkRightZero( Mat frameCR, int num, int col, int row );
 int checkRight255( Mat frameCR, int num, int col, int row );
 bool edgeCount( Mat &frameEC, cv::Rect &location );
 
 void hullPlotter( Mat src, vector<vector<Point>> contour, string name, int x, int y, Mat &mask );
 bool checkLeft( Mat drawing, int row, int col );	
 bool checkRight( Mat drawing, int row, int col );	
 bool checkUp( Mat drawing, int row, int col );	
 bool checkDown( Mat drawing, int row, int col );	
 void contourAmalgamator( vector<vector<Point>> contourIN, vector<vector<Point>> &contour );	
 
 void fillHorrizontalRegions( Mat &BS_mask, vector<vector<Point>> c0 );
 void IB_trackObject( bool active, Point location, Mat &frame, vector<Point> &points, int fc );


 /** Global variables */
 Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
 int startOfTracking=-1;

 int main( int argc, const char** argv )
 {
	Mat wholeImage;
	Point ROI_TL;
	Rect location;
	Mat bsImage;
	Mat wholeROI;
	Mat bsROI;
	Mat bsKMeansROI;
	Mat bsMask;
	Mat bsMaskROI;
	Mat trackFrame;

	CvCapture* capture;
	char filename[100];
	sprintf(filename, "C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Recordings/M4H00717.MP4.AVI");
	capture = cvCaptureFromAVI(filename);
	pMOG2 = new BackgroundSubtractorMOG2();
	int frameCount=0;
	int modelBuildCount=0;
	Rect loc1;
	float con1;
	vector<Point> trackingArray;

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
		cout << "Frame: " << frameCount++ << endl;


		IB_BS( true, wholeImage, bsMask, bsImage );
		IB_detectAndDisplay( true, wholeImage, bsImage, bsMask, wholeROI, bsROI, bsMaskROI, ROI_TL );

		if( wholeROI.empty() || frameCount == 3 )	{
			continue;
		}

		IB_colorSpaceAnalysis( true, bsROI, bsMaskROI, bsKMeansROI );
		if( true && IB_colourClassification( loc1, con1, bsKMeansROI, location ) )	{
			rectangle( wholeImage, Point( ROI_TL+location.tl() ), Point( ROI_TL+location.br() ), Scalar( 255, 0, 0 ), 3 );
		}
		if( true && IB_edgeAnalysis( bsROI, location ) )	{
			rectangle( wholeImage, Point( ROI_TL+location.tl() ), Point( ROI_TL+location.br() ), Scalar( 0, 255, 0 ), 3 );
		}

		Point trackingLocation = Point( ROI_TL.x + location.x + (location.width/2), ROI_TL.y + location.y + (location.height/2) );
		IB_trackObject( true, trackingLocation, wholeImage, trackingArray, frameCount );


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
		cv::imshow( "Region of Interest", wholeROI );
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

 void IB_colorSpaceAnalysis( bool active, Mat bsROI, Mat bsMask, Mat &bsKMeansROI )
 {	
	 if( !active )	{
		 return;
	 }

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
	int cluster_count = 3;

		
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
		}
	}
	if( r2 == 0 )	{
		pointsBGR = frameBGR.reshape( 1, frameBGR.cols*frameBGR.rows );
		r2 = r1;
	} else {
		pointsBGR = tempPointsBGR.rowRange( 0, r2 );
		positionBGR = tempPositionBGR.rowRange( 0, r2 );
	}

	Mat clustersBGR, centersBGR;
	pointsBGR.convertTo(pointsBGR, CV_32FC3, 1.0/255.0);

	kmeans( pointsBGR, cluster_count, clustersBGR,
		TermCriteria( CV_TERMCRIT_ITER, 10, 10.0 ),
		3, KMEANS_PP_CENTERS, centersBGR );

	frameBGR.setTo( Scalar() );

	int row=0, col=0;
	for(int r3=0; r3<r2; r3++) 
	{
		row = positionBGR.at<uchar>( r3, 0 );
		col = positionBGR.at<uchar>( r3, 1 );
		frameBGR.at<Vec3b>(row, col) = new_pixel[ clustersBGR.at<int>(r3, 0) ];
	}

	bsKMeansROI = frameBGR.clone();
}

void IB_detectAndDisplay( bool active, Mat wholeImage, Mat bsImage, Mat bsMask, Mat &wholeROI, Mat &bsROI, Mat &bsMaskROI, Point &ROI_TL )
{
	if( !active )	{
		return;
	}

  std::vector<Rect> one;
  std::vector<Rect> two;
  std::vector<Rect> three;
  std::vector<Rect> four;
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
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(wholeImage, r.tl(), r.br(), cv::Scalar(0,0,255), 3);

		int x1 = r.x;
		int x2 = r.width;
		int y1 = r.y-25;
		int y2 = r.height/2;
		if (x1 < 0) x1=0;
		if ((x1+x2)-640 > 0) x2=x2-((x1+x2)-640);
		if (y1 < 0) y1=0;
		if ((y1+y2)-360 > 0) y2=y2-((y1+y2)-360);
		x1 = x1 + x2/8;
		x2 = 3*x2/4;
		y1 = y1 + y2/2;
		y2 = y2/2;
		wholeROI = wholeImageT( Rect( x1, y1, x2, y2 ));
		bsROI = bsImageT( Rect( x1, y1, x2, y2 ));
		bsMaskROI = bsMaskT( Rect( x1, y1, x2, y2 ));
		ROI_TL = Point( x1, y1 );

		return;
	}
 }

bool IB_edgeAnalysis( Mat frameE, cv::Rect &location )
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
	Mat t11;
	Mat t12;

	cvtColor( frameE, t2, CV_RGB2GRAY );
	t2.convertTo( t4, CV_8UC1 );
	Canny( t4, t3, 100, 200, 3, true );
	Sobel( t2, t7, CV_16S, 1, 0, 7 );
	inRange( t7, -15000, 32767, t8 );
	inRange( t7, -32768, 15000, t9 );
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
	vector<vector<Point>> cA;
	vector<Vec4i> hA;
	t1 = t0.clone();
	t1 = edgeThining( t1 );
	t11 = t1.clone();
	t12 = t1.clone();
	for( int row=0; row<t0.rows; row++ )	{
		for( int col=0; col<t0.cols; col++ )	{
			if( t11.at<uchar>(row,col) == 0 )	{
				t11.at<uchar>(row,col) = 255;
			} else if( t11.at<uchar>(row,col) == 255 ) {
				t11.at<uchar>(row,col) = 255;
			} else {
				t11.at<uchar>(row,col) = 0;
			}
		}
	}
	t12.setTo( Scalar() );
	findContours( t11, cA, hA, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );
	drawContours( t12, cA, -1, Scalar( 255 ), 1, 8 );
	
	/*
	resize( t11, t11, Size( 5*t7.cols, 5*t7.rows ) );
	imshow( "CONTOURS", t11 );	
	resize( t12, t12, Size( 5*t7.cols, 5*t7.rows ) );
	imshow( "CONTOURS2", t12 );	
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
	//resize( t5, t5, Size( 5*t5.cols, 5*t5.rows ) );
	//imshow( "Laplacian", t5 );
	*/

	if( edgeCount( t1, location ) )	{
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

void IB_BS( bool active, Mat BS_in, Mat &bsMask, Mat &BS_out )
{
	if( !active )	{
		return;
	}

	BS_out.setTo( Scalar() );
	Mat BS_mask, BS_C_mask;
	pMOG2->operator()(BS_in, BS_mask, -1 );
	imshow( "A", BS_mask );
	threshold( BS_mask, BS_mask, 250, 255, THRESH_BINARY );
	imshow( "B", BS_mask );
	morphologyEx( BS_mask, BS_mask, MORPH_CLOSE, Mat(), Point(-1,-1), 4 );
	morphologyEx( BS_mask, BS_mask, MORPH_OPEN, Mat(), Point(-1,-1), 1 );
	imshow( "D", BS_mask );
	vector<vector<Point>> c0, a0;
	vector<Vec4i> h0;
	/*
	cvtColor( BS_mask, BS_C_mask, CV_GRAY2BGR );
	contourAmalgamator( c0, a0 );
	drawContours( BS_mask, a0, -1, uchar( 255 ), CV_FILLED );
	Mat BS_D_mask = BS_mask.clone();
	hullPlotter( BS_C_mask, c0, "HULL FOUR", 1300, 250, BS_D_mask );
	hullPlotter( BS_C_mask, a0, "HULL A FOUR", 1300, 650, BS_mask );
	//drawContours( BS_C_mask, c0, -1, Scalar( 0, 0, 255 ), CV_FILLED );
	*/
	BS_in.copyTo( BS_out, BS_mask );
	bsMask = BS_mask;
}

void fillHorrizontalRegions( Mat &BS_mask, vector<vector<Point>> c0 )
{
	Mat temp = BS_mask.clone();
	int fl=0, fr=0;
	for( size_t i=0; i<c0.size(); i++ )	{
		temp.setTo( Scalar() );
		drawContours( temp, c0, i, Scalar( 255 ) );
		for( int row=0; row<temp.rows; row++ )	{
			fl = fr = -1;
			for( int col=0; col<temp.cols; col++ )	{
				if( temp.at<uchar>(row, col) == 255 )	{
					if( fl < 0 )	{
						fl = col;
					} else if( col > fr )	{
						fr = col;
					}
				}
			}
			for( int col2=fl; col2<fr; col2++ )	{
				BS_mask.at<uchar>(row, col2) = 255;
			}
		}
	}
}

void hullPlotter( Mat src, vector<vector<Point>> contour, string name, int x, int y, Mat &mask )
{
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
			drawContours( drawing, hull, i, Scalar( 255, 255, 255 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, contour, i, Scalar( 0, 0, 0 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, hull, i, Scalar( 0, 255, 0 ), 1, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, contour, i, Scalar( 0, 0, 255 ), 1, 8, vector<Vec4i>(), 0, Point() );
		  }

	   for( int i = 0; i< contour.size(); i++ )
		  {
			Point2f tp[4]; 
			rArray[i].points( tp );
			line( rdrawing, tp[0], tp[1], Scalar( 0, 255, 0 ) );
			line( rdrawing, tp[1], tp[2], Scalar( 0, 255, 0 ) );
			line( rdrawing, tp[2], tp[3], Scalar( 0, 255, 0 ) );
			line( rdrawing, tp[3], tp[0], Scalar( 0, 255, 0 ) );
			drawContours( drawing, hull, i, Scalar( 255, 255, 255 ), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
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

bool IB_colourClassification( Rect &loc1, float &con1, Mat bsKMeansROI, Rect &location )	
{
	bool *symetric=NULL;
	vector<vector<Point>> contoursB, contoursG, contoursR;
	Mat B( bsKMeansROI.rows, bsKMeansROI.cols, CV_8UC1 );
	Mat G( bsKMeansROI.rows, bsKMeansROI.cols, CV_8UC1 );
	Mat R( bsKMeansROI.rows, bsKMeansROI.cols, CV_8UC1 );
	Mat out[] = { B, G, R };
	int from_to[] = { 0,0, 1,1, 2,2 };
	mixChannels( &bsKMeansROI, 1, out, 3, from_to, 3 );
	if( helper_colourClassification( "B", B, symetric, contoursB, true, location ) ||
		helper_colourClassification( "G", G, symetric, contoursG, true, location ) ||
		helper_colourClassification( "R", R, symetric, contoursR, true, location ) )	{
		return true;
	}
	return false;
}

bool helper_colourClassification( string name, Mat chan, bool *symetric, vector<vector<Point>> contours, bool h, Rect &location ) 
{
	vector<Vec4i> hierarchy;
	Mat chan2( chan.rows, chan.cols, CV_8UC1 );
	chan2 = Scalar();
	findContours( chan, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );
	if( contours.size() == 0 )	{
		return false;
	}	
	bool *Lratio = new bool[contours.size()];
	bool *Rratio = new bool[contours.size()];
	symetric = new bool[contours.size()];
	RotatedRect *cBounds = new RotatedRect[contours.size()];
	Rect *rBounds = new Rect[contours.size()];
	for( size_t i=0; i<contours.size(); i++ )	{
		Lratio[i] = false; 
		Rratio[i] = false;
		symetric[i] = false;
		cBounds[i] = minAreaRect( contours[i] );
		rBounds[i] = boundingRect( contours[i] );
		if( h && hierarchy[i][3] >= 0 )	{
			continue;
		}
		cout << cBounds[i].size.height/cBounds[i].size.width << endl;
		if( cBounds[i].size.height/cBounds[i].size.width > 0 || cBounds[i].size.height/cBounds[i].size.width < 0 )	{
			drawContours( chan2, contours, i, Scalar( 255 ) );
			if( cBounds[i].center.x > chan.cols/2 )	{
				Rratio[i] = true;
			} else {
				Lratio[i] = true;
			}
		}
	}
	resize( chan2, chan2, Size( 5*chan.cols, 5*chan.rows ) );
	imshow( name, chan2 );
	for( size_t i=0; i<contours.size(); i++ )	{
		for( size_t j=0; j<contours.size(); j++ )	{
			if( Lratio[i] == true && Rratio[j] == true )	{
				if( abs( abs( chan.cols/2 - cBounds[i].center.x ) - abs( chan.cols/2 - cBounds[j].center.x  ) ) < cBounds[i].size.width/2 &&
					abs( abs( chan.rows/2 - cBounds[i].center.y ) - abs( chan.rows/2 - cBounds[j].center.y  ) ) < cBounds[i].size.height/2 )	{
					symetric[i] = true;
					location = rBounds[i];
					cout << name << " Region #" << i << ": x; " << cBounds[i].center.x << ", " << chan.cols << ", y; " << 
						cBounds[i].center.y << ", " << chan.rows << ", r; " << cBounds[i].size.height/cBounds[i].size.width << ", " << cBounds[i].size << endl;
					cout << "S" << " Region #" << j << ": x; " << cBounds[j].center.x << ", " << chan.cols << ", y; " << 
						cBounds[j].center.y << ", " << chan.rows << ", r; " << cBounds[j].size.height/cBounds[j].size.width << ", " << cBounds[j].size << endl;
					return true;
				}
				if( abs( abs( chan.cols/2 - cBounds[i].center.x ) - abs( chan.cols/2 - cBounds[j].center.x  ) ) < cBounds[j].size.width/2 &&
					abs( abs( chan.rows/2 - cBounds[i].center.y ) - abs( chan.rows/2 - cBounds[j].center.y  ) ) < cBounds[j].size.height/2 )	{
					symetric[j] = true;
					location = rBounds[j];
					cout << name << " Region #" << j << ": x; " << cBounds[j].center.x << ", " << chan.cols << ", y; " << 
						cBounds[j].center.y << ", " << chan.rows << ", r; " << cBounds[j].size.height/cBounds[j].size.width << ", " << cBounds[j].size << endl;
					cout << "S" << " Region #" << i << ": x; " << cBounds[i].center.x << ", " << chan.cols << ", y; " << 
						cBounds[i].center.y << ", " << chan.rows << ", r; " << cBounds[i].size.height/cBounds[i].size.width << ", " << cBounds[i].size << endl;
					return true;
				}
			}
		}
	}
	return false;
}

// Needs to be re-written
void IB_trackObject( bool active, Point location, Mat &frame, vector<Point> &points, int fc )	
{
	if( !active )	{
		return;
	}

	if( ( points.empty() ) || ( abs( location.x - points.back().x ) < frame.cols*0.1 && abs( location.x - points.back().x ) < frame.cols*0.1 ) )	{
		if( points.empty() )	{
			startOfTracking = fc;
		}
		points.push_back( location );
	}
        
	for( int i=0; i<points.size()-1; i++ )	{
		// Draw a yellow line from the previous point to the current point
		line( frame, points[i], points[i+1], cvScalar(0,255,255), 1 );
	}
	if( points.size() > 1 )	{
		cout << "Confidence = " << (points.size()*100)/(fc-startOfTracking+1) << "%" << endl;
	}

}