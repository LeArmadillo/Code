//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

//global variables
Mat frame; //current frame
Mat fgMaskMOG; //fg mask generated by MOG method
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Mat fgMaskBS; //fg mask generated by MOG method
Mat fgMaskGMG; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
Ptr<BackgroundSubtractor> pBS; //MOG Background subtractor
Ptr<BackgroundSubtractor> pGMG; //MOG2 Background subtractor
int keyboard;

//function declarations
void help();
void processVideo(char* videoFilename);
void processImages(char* firstFrameFilename);
void hullPlotter( Mat src, vector<vector<Point>> contour, string name, int x, int y );
bool checkLeft( Mat drawing, int col, int row );	
bool checkRight( Mat drawing, int col, int row );	
bool checkUp( Mat drawing, int col, int row );	
bool checkDown( Mat drawing, int col, int row );
void contourAmalgamator( vector<vector<Point>> contourIN, vector<vector<Point>> &contour );

void help()
{
    cout
        << "--------------------------------------------------------------------------"  << endl
        << "This program shows how to use background subtraction methods provided by "   << endl
        << " OpenCV. You can process both videos (-vid) and images (-img)."              << endl
        << endl
        << "Usage:"                                                                      << endl
        << "./bs {-vid <video filename>|-img <image filename>}"                          << endl
        << "for example: ./bs -vid video.avi"                                            << endl
        << "or: ./bs -img /data/images/1.png"                                            << endl
        << "--------------------------------------------------------------------------"  << endl
        << endl;
}

int main(int argc, char* argv[])
{
	
	//_OutputArray kx, ky;
	//int kx[100];
	//int ky[100];
	Mat kx, ky;
	getDerivKernels( kx, ky, 1, 0, 7 );
	cout << kx << "              " << ky << endl;
	

    //print help information
    help();

    //check for the input parameter correctness
    if(argc != 3) {
        cerr <<"Incorret input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }

    //create GUI windows
    //namedWindow("Frame");
    //namedWindow("FG Mask MOG");
    //namedWindow("FG Mask MOG 2");    
	//namedWindow("FG Mask BS");
    //namedWindow("FG Mask GMG");
	//namedWindow("FG Mask FGD");

    //create Background Subtractor objects
   //NOTE HERE!!!!
    //pMOG= new BackgroundSubtractorMOG( 100, 2, 0.3, 0.2 ); //MOG approach
    //pMOG2 = new BackgroundSubtractorMOG2( 100, 3, true ); //MOG2 approach
	//pMOG= new BackgroundSubtractorMOG(); //MOG approach
    pMOG2 = new BackgroundSubtractorMOG2(); //MOG2 approach
    //pBS = new BackgroundSubtractor();
	//pGMG = new BackgroundSubtractorGMG();
	//pFGD = new BackgroundSubtractorFGD();
	//pMOG = new BackgroundSubtractorMOG2( 100, 8, true ); //MOG2 approach
	//pGMG = new BackgroundSubtractorMOG2( 100, 1, true ); //MOG2 approach
	//pBS = new BackgroundSubtractorMOG2( 100, 0, true ); //MOG2 approach

    if(strcmp(argv[1], "-vid") == 0) {
        //input data coming from a video
        processVideo(argv[2]);
    }
    else if(strcmp(argv[1], "-img") == 0) {
        //input data coming from a sequence of images
        processImages(argv[2]);
    }
    else {
        //error in reading input parameters
        cerr <<"Please, check the input parameters." << endl;
        cerr <<"Exiting..." << endl;
        return EXIT_FAILURE;
    }
    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}

void processVideo(char* videoFilename) {
    //create the capture object
    VideoCapture capture(videoFilename);
    if(!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }
	int frameCount=0;
	int modelBuildCount=0;
    //read input data. ESC or 'q' for quitting
    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
        //read the current frame
		if( !( frameCount == 3 ) || modelBuildCount > 0 )	{
			if(!capture.read(frame)) {
				cout << frameCount << endl;
				cerr << "Unable to read next frame." << endl;
				cerr << "Exiting..." << endl;
				exit(EXIT_FAILURE);
			}
			frameCount++;
		} else {
			cout << "#" << modelBuildCount << "  Building Model!" << endl;
			modelBuildCount++;
		}
        //update the background model
           //AND HERE!!!
        //pMOG->operator()(frame, fgMaskMOG, -1 );
        pMOG2->operator()(frame, fgMaskMOG2, -1 );
		//pBS->operator()(frame, fgMaskBS, -1 );
		//pGMG->operator()(frame, fgMaskGMG, -1 );
		if( frameCount == 3 )	{
			//cout << endl;
			continue;
		}
		//if( frameCount == 4 )	{
		//	cvWaitKey();
		//}
		cvWaitKey(10);
        //get the frame number and write it on the current frame
        stringstream ss;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
            cv::Scalar(255,255,255), -1);
        ss << capture.get(CV_CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //show the current frame and the fg masks
		
		threshold( fgMaskMOG2, fgMaskMOG2, 250, 255, THRESH_BINARY );
		
		//Mat three = fgMaskMOG2.clone();
		Mat four = fgMaskMOG2.clone();
		//Mat one = fgMaskMOG2.clone();
		//Mat two = fgMaskMOG2.clone();

		/*
		morphologyEx( one, one, MORPH_OPEN, Mat() );
		morphologyEx( one, one, MORPH_CLOSE, Mat(), Point(-1, -1), 3 );

		morphologyEx( two, two, MORPH_OPEN, Mat() );
		morphologyEx( two, two, MORPH_CLOSE, Mat(), Point(-1, -1), 4 );
		
		morphologyEx( three, three, MORPH_CLOSE, Mat(), Point(-1, -1), 3 );
		morphologyEx( three, three, MORPH_OPEN, Mat() );
		*/

		morphologyEx( four, four, MORPH_CLOSE, Mat(), Point(-1, -1), 4 );
		morphologyEx( four, four, MORPH_OPEN, Mat() );

		vector<vector<Point>> c0, c1, c2, c3, a0, a1, a2, a3;
		vector<Vec4i> h0, h1, h2, h3;

		//findContours( one, c0, h0, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		//findContours( two, c1, h1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		//findContours( three, c2, h2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		findContours( four, c3, h3, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

		/*
		cvtColor( one, one, CV_GRAY2BGR ); 
		cvtColor( two, two, CV_GRAY2BGR ); 
		cvtColor( three, three, CV_GRAY2BGR ); 
		*/
		cvtColor( four, four, CV_GRAY2BGR );

		/*
		contourAmalgamator( c0, a0 );
		contourAmalgamator( c1, a1 );
		contourAmalgamator( c2, a2 );
		*/
		contourAmalgamator( c3, a3 );

		//hullPlotter( one, c0, "HULL ONE", 0, -800 );
		//hullPlotter( two, c1, "HULL TWO", 650, -800 );
		//hullPlotter( three, c2, "HULL THREE", 0, -400 );
		hullPlotter( four, c3, "HULL FOUR", 0, -800 );
		//hullPlotter( one, a0, "HULL A ONE", 0, 0 );
		//hullPlotter( two, a1, "HULL A TWO", 650, 0 );
		//hullPlotter( three, a2, "HULL A THREE", 0, 400 );
		hullPlotter( four, a3, "HULL A FOUR", 650, -800 );

		//drawContours( one, c0, -1, Scalar( 0, 0, 255 ), CV_FILLED );
		//drawContours( two, c1, -1, Scalar( 0, 0, 255 ), CV_FILLED );
		//drawContours( three, c2, -1, Scalar( 0, 0, 255 ), CV_FILLED );
		drawContours( four, c3, -1, Scalar( 0, 0, 255 ), CV_FILLED );
		//drawContours( two, c1, -1, Scalar( 0, 0, 255 ), CV_FILLED );
		//drawContours( three, c2, -1, Scalar( 0, 0, 255 ), CV_FILLED, 4, h2, 1 );
		//drawContours( four, c3, -1, Scalar( 0, 0, 255 ), CV_FILLED, 1, h3, 3 );

		imshow( "MASK", fgMaskMOG2 );
		/*
		imshow( "ONE", one );
		imshow( "TWO", two );
		imshow( "THREE", three );
		*/
		imshow( "FOUR", four );
		moveWindow( "MASK", 0, -400 );
		/*
		moveWindow( "ONE", 0, 0 );
		moveWindow( "TWO", 650, 0 );
		moveWindow( "THREE", 0, 400 );
		*/
		moveWindow( "FOUR", 650, -400 );

		keyboard = waitKey( 100 );
    }
    //delete capture object
    capture.release();
}

void processImages(char* fistFrameFilename) {
    //read the first file of the sequence
    frame = imread(fistFrameFilename);
    if(!frame.data){
        //error in opening the first image
        cerr << "Unable to open first image frame: " << fistFrameFilename << endl;
        exit(EXIT_FAILURE);
    }
    //current image filename
    string fn(fistFrameFilename);
    //read input data. ESC or 'q' for quitting
    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
        //update the background model
            //ALSO HERE!!!!
        pMOG->operator()(frame, fgMaskMOG);
        pMOG2->operator()(frame, fgMaskMOG2);
        //get the frame number and write it on the current frame
        size_t index = fn.find_last_of("/");
        if(index == string::npos) {
            index = fn.find_last_of("\\");
        }
        size_t index2 = fn.find_last_of(".");
        string prefix = fn.substr(0,index+1);
        string suffix = fn.substr(index2);
        string frameNumberString = fn.substr(index+1, index2-index-1);
        istringstream iss(frameNumberString);
        int frameNumber = 0;
        iss >> frameNumber;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
            cv::Scalar(255,255,255), -1);
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("FG Mask MOG", fgMaskMOG);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        //get the input from the keyboard
        keyboard = waitKey( 30 );
        //search for the next image in the sequence
        ostringstream oss;
        oss << (frameNumber + 1);
        string nextFrameNumberString = oss.str();
        string nextFrameFilename = prefix + nextFrameNumberString + suffix;
        //read the next frame
        frame = imread(nextFrameFilename);
        if(!frame.data){
            //error in opening the next image in the sequence
            cerr << "Unable to open image frame: " << nextFrameFilename << endl;
            exit(EXIT_FAILURE);
        }
        //update the path of the current frame
        fn.assign(nextFrameFilename);
    }
}

void hullPlotter( Mat src, vector<vector<Point>> contour, string name, int x, int y )
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
		   cout << tp << endl;
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
				   } 
			   }
		   }
	   }

		/// Show in a window
		namedWindow( name, CV_WINDOW_AUTOSIZE );
		imshow( name, drawing );
		moveWindow( name, x, y );

}

// COL AND ROW ARE REVERSED !!!!!!!!!!!!1
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
	for( int i = 0; i < contour.size(); i++ )	{  
		rArray[i] = minAreaRect( Mat(contour[i]) );
	}


	for( int i = 0; i < contour.size(); i++ )	{
		for( int j = 0; j < contour.size(); j++ )	{
			if( i <= j )	{
				continue;
			}
			if( ( ( rArray[i].angle > rArray[j].angle-5 ) && ( rArray[i].angle < rArray[j].angle+5 ) )
			&& ( ( rArray[i].center.x > 0.5*rArray[j].center.x ) && ( rArray[i].center.x < 1.5*rArray[j].center.x ) )
			&& ( ( rArray[i].center.y > 0.5*rArray[j].center.y ) && ( rArray[i].center.y < 1.5*rArray[j].center.y ) ) )	{
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