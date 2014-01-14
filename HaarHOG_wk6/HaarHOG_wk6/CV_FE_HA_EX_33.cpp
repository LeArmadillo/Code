 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>
 #include <string.h>
 #include <ctype.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

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

 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;

   //-- 1. Load the cascades
   if( !one_cascade.load( one_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !two_cascade.load( two_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !three_cascade.load( three_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !four_cascade.load( four_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2A. Read the video stream from CAM
   //capture = cvCaptureFromCAM( -1 );

   //-- 2B. Read the video stream from AVI
   char filename[100];
   //sprintf(filename, "C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Haar Training/fullCRoom/Positives/fcrpXXX.wmv.avi");
   //sprintf(filename, "C:/Users/beattyoi/M4H00712.full.AVI");
   sprintf(filename, "J:/M4H00712.MP4.AVI");
   capture = cvCaptureFromAVI(filename);

   if( capture )
   {
     while( true )
     {
   frame = cvQueryFrame( capture );

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { detectAndDisplay( frame ); }
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
		//printf( "%d %d %d %d %d %d \n", x1, x2, x1+x2, y1, y2, y1+y2 );
		regionOfInterest = frame( Rect( x1, y1, x2, y2 ));
		//regionOfInterest = frame( Rect( 0, 0, 640, 360 ));
	 
		cvtColor( regionOfInterest, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );
		  //-- Detect faces
		  one_cascade.detectMultiScale( frame_gray, one, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(25, 25) );
		  //two_cascade.detectMultiScale( frame_gray, two, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
		  //three_cascade.detectMultiScale( frame_gray, three, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10) );

		  
		  for( int i = 0; i < one.size(); i++ )
		  {
			//Point center( x1 + one[i].x + one[i].width*0.5, y1 + one[i].y + one[i].height*0.5 );
			//ellipse( frame, center, Size( one[i].width*0.5, one[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
			//Point center2( one[i].x, one[i].y );
			//ellipse( frame, center2, Size( 1, 1), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
			//regionOfInterest = frame( Rect( x1, y1+40, x2, y2-35 ));
			imshow( temp_window_name, regionOfInterest );
		  }
		 /*
		  for( int i = 0; i < two.size(); i++ )
		  {
			Point center( x1 + two[i].x + two[i].width*0.5, y1 + two[i].y + two[i].height*0.5 );
			ellipse( frame, center, Size( two[i].width*0.5, two[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
			imshow( temp_window_name1, regionOfInterest );
		  }
 
		  for( int i = 0; i < three.size(); i++ )
		  {
			Point center( x1 + three[i].x + three[i].width*0.5, y1 + three[i].y + three[i].height*0.5 );
			ellipse( frame, center, Size( three[i].width*0.5, three[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
			//Point center2( three[i].x, three[i].y );
			//ellipse( frame, center2, Size( 1, 1), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
			imshow( temp_window_name5, regionOfInterest );
		  }
		  
		  

		  for( int i = 0; i < one.size(); i++ )
		  {
			  for( int j = 0; j < three.size(); j++ )
			  {
		  
				  if( ( one[i].x < three[j].x+three[j].width*0.5 && one[i].x > three[j].x-three[j].width*0.5 &&
					  one[i].y < three[j].y+three[j].height*0.5 && one[i].y > three[j].y-three[j].height*0.5 ) ||
					  ( three[j].x < one[i].x+one[i].width*0.5 && three[j].x > one[i].x-one[i].width*0.5 &&
					  three[j].y < one[i].y+one[i].height*0.5 && three[j].y > one[i].y-one[i].height*0.5 ) )	 
				  {
					  Point center( (three[j].x + three[j].width*0.5 + one[i].x + one[i].width*0.5)/2+x1, (three[j].y + three[j].height*0.5 + one[i].y + one[i].height*0.5)/2+y1 );
					  ellipse( frame, center, Size( (three[j].width*0.5+one[i].width*0.5)/2, (three[j].height*0.5+one[i].height*0.5)/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
					  imshow( temp_window_name1, regionOfInterest );
				  }
			  }
			  for( int k = 0; k < two.size(); k++ )
			  {
		  
				  if( ( one[i].x < two[k].x+two[k].width*0.5 && one[i].x > two[k].x-two[k].width*0.5 &&
					  one[i].y < two[k].y+two[k].height*0.5 && one[i].y > two[k].y-two[k].height*0.5 ) ||
					  ( two[k].x < one[i].x+one[i].width*0.5 && two[k].x > one[i].x-one[i].width*0.5 &&
					  two[k].y < one[i].y+one[i].height*0.5 && two[k].y > one[i].y-one[i].height*0.5 ) )
				  {
					  Point center( (two[k].x + two[k].width*0.5 + one[i].x + one[i].width*0.5)/2+x1, (two[k].y + two[k].height*0.5 + one[i].y + one[i].height*0.5)/2+y1 );
					  ellipse( frame, center, Size( (two[k].width*0.5+one[i].width*0.5)/2, (two[k].height*0.5+one[i].height*0.5)/2 ), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
					  imshow( temp_window_name2, regionOfInterest );
				  }
			  }
			  
		  } 

		  //imshow( temp_window_name, regionOfInterest );*/
	}

	cvtColor( frame, frame_gray2, CV_BGR2GRAY );
	equalizeHist( frame_gray2, frame_gray2 );
	
	two_cascade.detectMultiScale( frame_gray2, two, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	three_cascade.detectMultiScale( frame_gray2, three, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

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
 }