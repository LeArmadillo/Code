//#include "cxcore.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include "../../utilities.h"
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sstream>

 using namespace std;
 using namespace cv;
 
IplImage* image_for_on_mouse_show_values=NULL;
char* window_name_for_on_mouse_show_values=NULL;
int frameCount = 0;
int Gx=0, Gy=0;
string filename;

int main( int argc, const char** argv )
 {	 int num = 792;


	 for( int i=0; i<97; i++ )	{
		 if( num > 806 )	{
			break;
		}
		if( num == 710 || num == 718 || num == 719 || num == 720 || num == 721 || num == 732 )	{
			num++;
			continue;
		}
		if( num >= 760 && num <= 791 )	{
			num = 792;
			continue;
		}
	ofstream myfile;
	cout << num << " Name your file: " << endl;
	cin >> filename;
	myfile.open ( filename );

	IplImage* wholeImage;

	CvCapture* capture;
	char filename[100];
	sprintf(filename, "C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Recordings/M4H00%d.MP4.AVI", num );
	capture = cvCaptureFromAVI(filename);
	Point TL = NULL;
	Point BR = NULL;
	bool tl = false;
	bool quit = false;
	int num_b_l_o=0;
	stringstream fileContent;
	
	cvNamedWindow( "Original", 1 );
	int firstDetection = 0;
	frameCount = 0;

   if( capture )
   {
     while( true )
     {
		wholeImage = cvQueryFrame( capture );
		if( wholeImage == NULL )	{
			cout << endl << frameCount << endl;
			break;
		}
		frameCount++;

		cvSetMouseCallback( "Original", on_mouse_show_values, 0 );
		window_name_for_on_mouse_show_values="Original";
		image_for_on_mouse_show_values=wholeImage;
		cvShowImage( "Original", wholeImage );
		int key = NULL;
		bool real = false;
		bool human = false;
		bool left = false;
		bool right = false;
		bool lo = false;
		bool ro = false;

		num_b_l_o = 1;
		key = cvWaitKey(0);
		firstDetection++;
		if( key == 'q')	{
			fileContent << "@";
			firstDetection--;
			num_b_l_o = 0;
		} else if( key == 'w' )	{
			real = human = left = right = true;
		} else if( key == 's' )	{
			human = left = right = true;
			real = false;
		} else if( key == 'x' )	{
			real = left = right = true;
			human = false;
		} else if( key == 'e' )	{
			human = left = true;
			right = false;
		} else if( key == 'd' )	{
			human = real = left = true;
			real = right = false;
		} else if( key == 'c' )	{
			real = left = true;
			human = right = false;
		} else if( key == 'r' )	{
			human = right = true;
			left = false;
		} else if( key == 'f' )	{
			human = real = right = true;
			real = left = false;
		} else if( key == 'v' )	{
			real = right = true;
			human = left = false;
		}
		lo = ro = false;
		if( left )	{
			cout << endl << "Left Occlusion?";
			key = cvWaitKey(0);
			if( key == 'l' )	{
				lo = true;
				cout << "YES";
			}
		}
		if( right )	{
			cout << endl << "Right Occlusion?";
			key = cvWaitKey(0);
			if( key == 'l' )	{
				ro = true;
				cout << "YES";
			}
		}
		if( firstDetection != 0 && firstDetection%25 == 1 )	{
			do {
				cout << endl << "Draw Double Box" << endl;
				do{	
					key = cvWaitKey(0);
				} while( key != 'a' );
				TL.x = Gx;
				TL.y = Gy;
				tl = true;
				cout << "Top Left: " << Gx << ", " << Gy << endl;
				do{	
					key = cvWaitKey(0);
				} while( key != 's' );
				if( tl )	{
					BR.x = Gx;
					BR.y = Gy;
					tl = false;
					cout << "Bottom Right: " << Gx << ", " << Gy << endl << "Happy?" << endl;
					cvDrawRect( wholeImage, TL, BR, Scalar( 255, 255, 0 ) );
				} else {
					cout << endl << "NO TL" << endl;
				}
				key = cvWaitKey(0);
			} while( key != 'd' );
		}
		fileContent << "#" << frameCount << "~" << num_b_l_o;
		cout << "#" << frameCount << "~" << num_b_l_o << endl;
		fileContent  << "!"<< "0" << "£" << real << "$" << human << "%" << left << "¦" << lo << "^" << right << "|" << ro;
		if( firstDetection != 0 && firstDetection%25 == 1 )	{
			fileContent << "&" << TL << "*" << BR << "¬";
			//fileConent = fileConent + "&%d*%d*%d*%d", TL.x, TL.y, BR.x, BR.y;
		} else {
			fileContent << "?";
			//fileConent = fileConent + "?";
		}
	 }
   }
   cout << "Closing File" << endl << endl;
   //fileContent << ":";
   myfile << fileContent.str();
   myfile << ":" << endl;	// endl necessary to stop stringstream being limited to 4096 characters
	cvWaitKey(10);
	myfile.close();  

	cout << "Finished!" << endl;
	cvWaitKey(0);
	num++;
}
}
	
void write_text_on_image(IplImage* image, int top_row, int top_column, char* text)
{
	CvFont font;
	double hScale=0.5;
	double vScale=0.5;
	int    lineWidth=1;
	cvInitFont(&font,CV_FONT_VECTOR0, hScale,vScale,0,lineWidth);
	unsigned char colour[4] = { 0,0,255, 0 };
	cvPutText(image,text,cvPoint(top_column,top_row+12), &font, cvScalar(colour[0],colour[1],colour[2]));
}

void on_mouse_show_values( int event, int x, int y, int flags, void* )
{
	Gx = x;
	Gy = y;
	static IplImage* local_image = NULL;
	static int pixel_step = 0;
	static int width_step = 0;

    if (( !image_for_on_mouse_show_values ) || ( !window_name_for_on_mouse_show_values ))
        return;

	if ((local_image) && ((image_for_on_mouse_show_values->width != local_image->width) || 
						  (image_for_on_mouse_show_values->height != local_image->height)))
	{
		cvReleaseImage( &local_image );
		local_image = NULL;
	}
	if (local_image == NULL)
	{
		local_image = cvCloneImage( image_for_on_mouse_show_values );
		width_step = local_image->widthStep;
		pixel_step = local_image->widthStep/local_image->width;
	}

	if ((x < local_image->width) && (y < local_image->height))
	{
		cvCopyImage( image_for_on_mouse_show_values, local_image );
		char curr_point_text[100];
		unsigned char* curr_point = GETPIXELPTRMACRO( local_image, x, y, width_step, pixel_step );
		if (strncmp(local_image->colorModel,"RGB",3) == 0)
			sprintf(curr_point_text,"%4s = %d %d %d",local_image->colorModel,curr_point[2],curr_point[1],curr_point[0]);
		else sprintf(curr_point_text,"%4s = %d %d %d",local_image->colorModel,curr_point[0],curr_point[1],curr_point[2]);
		write_text_on_image(local_image,1,1,curr_point_text);
		sprintf(curr_point_text,"Position = %d %d",x,y);
		write_text_on_image(local_image,20,1,curr_point_text);
		sprintf(curr_point_text,"Frame: %d",frameCount);
		write_text_on_image(local_image,40,1,curr_point_text);
		cvShowImage( window_name_for_on_mouse_show_values, local_image );
	}
}
