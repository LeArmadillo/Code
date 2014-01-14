#ifdef _CH_
#pragma package <opencv>
#endif

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "../utilities.h"
#include <string>

int main( int argc, char** argv )
	// Unchanged from provided version
{
	int img_num = 1;
	IplImage* selected_image1 = NULL;
	IplImage* selected_image2 = NULL;
	IplImage* selected_image3 = NULL;
	IplImage* selected_image4 = NULL;

	// Create display windows for images
    cvNamedWindow( "Vidieo1", 1 );
	cvNamedWindow( "Vidieo2", 1 );
	cvNamedWindow( "Vidieo3", 1 );
	cvNamedWindow( "Vidieo4", 1 );

	// Play all the images.
	int user_clicked_key = 0;
	std::string img_str;
	while ( user_clicked_key != ESC )
	{
		char filename1[300];
		if ( img_num > 9999 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_004/Time_13-06/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_004/Time_13-06/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_004/Time_13-06/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_004/Time_13-06/000000%d.jpg",img_num);
		}	else {
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_004/Time_13-06/0000000%d.jpg",img_num);
		}
		if( (selected_image1 = cvLoadImage(filename1,-1)) == 0 )
			return 6;
		cvShowImage( "Vidieo1", selected_image1 );
				char filename2[300];
		if ( img_num > 9999 )	{
		sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/000000%d.jpg",img_num);
		}	else {
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/0000000%d.jpg",img_num);
		}
		if( (selected_image2 = cvLoadImage(filename2,-1)) == 0 )
			return 7;
		cvShowImage( "Vidieo2", selected_image2 );
				char filename3[300];
		if ( img_num > 999 )	{
			sprintf(filename3,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename3,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename3,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename3,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/000000%d.jpg",img_num);
		}	else {
			sprintf(filename3,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/0000000%d.jpg",img_num);
		}
		if( (selected_image3 = cvLoadImage(filename3,-1)) == 0 )
			return 8;
		cvShowImage( "Vidieo3", selected_image3 );
				char filename4[300];
		if ( img_num > 999 )	{
		sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-19/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-19/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-19/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-19/000000%d.jpg",img_num);
		}	else {
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-19/0000000%d.jpg",img_num);
		}
		if( (selected_image4 = cvLoadImage(filename4,-1)) == 0 )
			return 9;
		cvShowImage( "Vidieo4", selected_image4 );
		user_clicked_key = cvWaitKey(15);
		img_num++;
		cvReleaseImage(&selected_image1);
		cvReleaseImage(&selected_image2);
		cvReleaseImage(&selected_image3);
		cvReleaseImage(&selected_image4);
	}

    return 1;
}

/*
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_003/Time_13-06/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_003/Time_13-06/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_003/Time_13-06/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_003/Time_13-06/000000%d.jpg",img_num);
		}	else {
			sprintf(filename4,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_003/Time_13-06/0000000%d.jpg",img_num);




						sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/000000%d.jpg",img_num);
		}	else {
			sprintf(filename1,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_001/Time_13-19/0000000%d.jpg",img_num);





						sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/000%d.jpg",img_num);
		}	else if ( img_num > 999 )	{
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/0000%d.jpg",img_num);
		}	else if ( img_num > 99 )	{
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/00000%d.jpg",img_num);
		}	else if ( img_num > 9 )	{
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/000000%d.jpg",img_num);
		}	else {
			sprintf(filename2,"C:/Users/Ian Beatty-Orr/Documents/Y5 Eng/5E1/Source Images/PETS2009/Crowd_PETS09/S0/Background/View_002/Time_13-32/0000000%d.jpg",img_num);
			*/