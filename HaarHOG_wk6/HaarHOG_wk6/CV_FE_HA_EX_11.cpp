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
 String two_cascade_name = "haarcascade_frontalface_alt_tree.xml";
 String one_cascade_name = "haarcascade_mcs_upperbody.xml";
 CascadeClassifier one_cascade;
 CascadeClassifier two_cascade;
 CascadeClassifier three_cascade;
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
	Mat regionOfInterest;
	Mat regionOfInterest2;
	double t = (double)getTickCount();
	
  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  fflush(stdout);
	vector<Rect> found, found_filtered;
	
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(frame, found, 0, Size(0,0), Size(0,0), 1.05, 2);
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
		rectangle(frame, r.tl(), r.br(), cv::Scalar(0,0,255), 3);
	}
	
	

  std::vector<Rect> one;
  std::vector<Rect> two;
  std::vector<Rect> three;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  
  //-- Detect faces
  one_cascade.detectMultiScale( frame_gray, one, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  two_cascade.detectMultiScale( frame_gray, two, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  three_cascade.detectMultiScale( frame_gray, three, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  
  for( int i = 0; i < one.size(); i++ )
  {
    Point center( one[i].x + one[i].width*0.5, one[i].y + one[i].height*0.5 );
    ellipse( frame, center, Size( one[i].width*0.5, one[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
	//Point center2( one[i].x, one[i].y );
	//ellipse( frame, center2, Size( 1, 1), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
  }
  
  for( int i = 0; i < two.size(); i++ )
  {
    Point center( two[i].x + two[i].width*0.5, two[i].y + two[i].height*0.5 );
    ellipse( frame, center, Size( two[i].width*0.5, two[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
  }
  
  for( int i = 0; i < three.size(); i++ )
  {
    Point center( three[i].x + three[i].width*0.5, three[i].y + three[i].height*0.5 );
    ellipse( frame, center, Size( three[i].width*0.5, three[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
	//Point center2( three[i].x, three[i].y );
	//ellipse( frame, center2, Size( 1, 1), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
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
			  Point center( (three[j].x + three[j].width*0.5 + one[i].x + one[i].width*0.5)/2, (three[j].y + three[j].height*0.5 + one[i].y + one[i].height*0.5)/2 );
			  ellipse( frame, center, Size( (three[j].width*0.5+one[i].width*0.5)/2, (three[j].height*0.5+one[i].height*0.5)/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
				int x1 = one[i].x;
				int x2 = one[i].width;
				int y1 = one[i].y;
				int y2 = one[i].height;
				if (x1 < 0) x1=0;
				if (x1 > 640) x1=639;
				if ((x1+x2)-640 > 0) x2=x2-((x1+x2)-640);
				if (y1 < 0) y1=0;
				if (y1 > 360) y1=359;
				if ((y1+y2)-360 > 0) y2=y2-((y1+y2)-360);
				if (x2 == 0) x2=1;
				if (y2 == 0) y2=1;
				printf( " A %d %d %d %d %d %d \n", x1, x2, x1+x2, y1, y2, y1+y2 );
				regionOfInterest = frame( Rect( x1, y1, x2, y2 ));
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
			  Point center( (two[k].x + two[k].width*0.5 + one[i].x + one[i].width*0.5)/2, (two[k].y + two[k].height*0.5 + one[i].y + one[i].height*0.5)/2 );
			  ellipse( frame, center, Size( (two[k].width*0.5+one[i].width*0.5)/2, (two[k].height*0.5+one[i].height*0.5)/2 ), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
				int x1 = one[i].x;
				int x2 = one[i].width;
				int y1 = one[i].y;
				int y2 = one[i].height;
				if (x1 < 0) x1=0;
				if (x1 > 640) x1=639;
				if ((x1+x2)-640 > 0) x2=x2-((x1+x2)-640);
				if (y1 < 0) y1=0;
				if (y1 > 360) y1=359;
				if ((y1+y2)-360 > 0) y2=y2-((y1+y2)-360);
				if (x2 == 0) x2=1;
				if (y2 == 0) y2=1;
				printf( " B %d %d %d %d %d %d \n", x1, x2, x1+x2, y1, y2, y1+y2 );
				regionOfInterest2 = frame( Rect( x1, y1, x2, y2 ));
			  imshow( temp_window_name2, regionOfInterest2 );
		  }
	  }
  }  
  t = (double)getTickCount() - t;
  printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
  //-- Show what you got
  imshow( window_name, frame );
 }