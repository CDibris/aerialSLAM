/**
Aerial SLAM
Author: Carmine RECCHIUTO, DIBRIS; UNIGE
*/
#include <ros/ros.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/flann/flann.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include </usr/include/opencv2/stitching/stitcher.hpp>
#include <cvaux.h>
#include <math.h>
#include <cxcore.h>

//#include </opt/ros/hydro/include/opencv/cv.hpp>
//#include </opt/ros/hydro/include/opencv2/highgui/highgui.hpp>
//#include "/opt/ros/hydro/include/opencv2/features2d/features2d.hpp"
//#include "/opt/ros/hydro/include/opencv2/nonfree/features2d.hpp"
//#include "/opt/ros/hydro/include/opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

static IplImage *frame = NULL, *frame1_1C = NULL, *frame2_1C = NULL, *frame_save = NULL;

int count1 = 0;
	Mat img_1, img_2, img_tot;

Point Pt_old, Pt_new;
cv::Mat imageToDraw;

Mat detected_edges_2, detected_edges_1;

int edgeThresh = 1;
int lowThreshold=75;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold1(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( img_1, detected_edges_1, Size(3,3) );

  /// Canny detector
  Canny( detected_edges_1, detected_edges_1, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  //dst = Scalar::all(0);

  //src.copyTo( dst, detected_edges);
  imshow( window_name, detected_edges_1 );
 }

void CannyThreshold2(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( img_2, detected_edges_2, Size(3,3) );

  /// Canny detector
  Canny( detected_edges_2, detected_edges_2, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  //dst = Scalar::all(0);

  //src.copyTo( dst, detected_edges);
  //imshow( window_name, detected_edges_2 );
 }


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
	{	

	
	if (count1 ==0)
		{

		cv_bridge::CvImagePtr cv_ptr;
		try
		    {
		      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		    }
		    catch (cv_bridge::Exception& e)
		    {
		      ROS_ERROR("cv_bridge exception: %s", e.what());
		      return;
		    }
			img_1 = cv_ptr->image;	
			imageToDraw = img_1;
                        count1++;
			img_tot=img_1; 
		}


	else
		{
			
		cv_bridge::CvImagePtr cv_ptr;
		try
		    {
		      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		    }
		    catch (cv_bridge::Exception& e)
		    {
		      ROS_ERROR("cv_bridge exception: %s", e.what());
		      return;
		    }
   
			img_2 = cv_ptr->image;

   
			  if( !img_1.data || !img_2.data )
			  { std::cout<< " --(!) Error reading images " << std::endl;}

			// canny detection

			//dst.create(img_1.size(), img_1.type() );
			
    			namedWindow( window_name, CV_WINDOW_AUTOSIZE );

 			 /// Create a Trackbar for user to enter threshold
  			createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold1 );

 			/// Show the image
 			CannyThreshold1(0, 0);
			CannyThreshold2(0, 0); 
			
			//img_1 = detected_edges_1;
			//img_2 = detected_edges_2;


			  //-- Step 1: Detect the keypoints using SURF Detector
			  int minHessian = 400;

			  SurfFeatureDetector detector( minHessian );

			  std::vector<KeyPoint> keypoints_1, keypoints_2;

			  detector.detect( img_1, keypoints_1 );
			  detector.detect( img_2, keypoints_2 );

			  //-- Step 2: Calculate descriptors (feature vectors)
			  SurfDescriptorExtractor extractor;

			  Mat descriptors_1, descriptors_2;

			  extractor.compute( img_1, keypoints_1, descriptors_1 );
			  extractor.compute( img_2, keypoints_2, descriptors_2 );

			  //-- Step 3: Matching descriptor vectors using FLANN matcher
			  FlannBasedMatcher matcher;
			  std::vector< DMatch > matches;
			  matcher.match( descriptors_1, descriptors_2, matches );

			  double max_dist = 0; double min_dist = 100;

			  //-- Quick calculation of max and min distances between keypoints
			  for( int i = 0; i < descriptors_1.rows; i++ )
			  { double dist = matches[i].distance;
			    if( dist < min_dist ) min_dist = dist;
			    if( dist > max_dist ) max_dist = dist;
			  }

			  printf("-- Max dist : %f \n", max_dist );
			  printf("-- Min dist : %f \n", min_dist );

			  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
			  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
			  //-- small)
			  //-- PS.- radiusMatch can also be used here.
			  std::vector< DMatch > good_matches;
			  std::vector<KeyPoint> good_keypoints_1, good_keypoints_2;

			  for( int i = 0; i < descriptors_1.rows; i++ )
			  { if( matches[i].distance <= max(3*min_dist, 0.0) )
			    { good_matches.push_back( matches[i]); 
			      good_keypoints_1.push_back(keypoints_1[i]);			     
			      good_keypoints_2.push_back(keypoints_2[i]);
			     }
			  }

			  //-- Draw only "good" matches
			  Mat img_matches;
			  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
				       good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				       vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

			  //-- Show detected matches
			  imshow( "Good Matches", img_matches );
			
			//float deltax, deltay;
                         float deltax[(int)good_matches.size()], deltay[(int)good_matches.size()];
			 float sum_deltax=0, sum_deltay=0;

			  for( int i = 0; i < (int)good_matches.size(); i++ )
			  { 
			   
			   // printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
			   // cout << "-- Good Match Keypoint 1:" << keypoints_1[good_matches[i].queryIdx].pt.x << endl;
   			   // cout << "-- Good Match Keypoint 2:" << keypoints_2[good_matches[i].trainIdx].pt.x << endl;
				
				deltax[i] = keypoints_2[good_matches[i].trainIdx].pt.x - keypoints_1[good_matches[i].queryIdx].pt.x;
				deltay[i] = keypoints_2[good_matches[i].trainIdx].pt.y - keypoints_1[good_matches[i].queryIdx].pt.y;

			  sum_deltax=+deltax[i];
			  sum_deltay=+deltay[i];
		

			  }

			float av_deltax = sum_deltax/(int)good_matches.size();
			float av_deltay = sum_deltay/(int)good_matches.size();

			float av_deltax2 = 0, av_deltay2 =0;

			cout << "before: av_deltax " << av_deltax << " av_deltay " << av_deltay << endl;
			
			int count2=0;

			for( int i = 0; i < (int)good_matches.size(); i++ )
			  {
			  if ((abs(deltax[i]-av_deltax) < 50 ) & (abs(deltay[i]-av_deltay)<50))
			  	{
				av_deltax2 =+deltax[i];
				av_deltay2 =+deltay[i];
				count2++;	      			
				}
			  }
			
			if (count2>0)
			{av_deltax2 = -av_deltax2/count2;
			 av_deltay2 = -av_deltay2/count2;
			cout << "after: av_deltax " << av_deltax << " av_deltay " << av_deltay << endl;
			}
			
			else
			{cout << "ATTENZIONE:keeping the old value " << endl;
			av_deltax2=0;
			av_deltay2=0;}
			
			 Pt_new.x=Pt_old.x+av_deltax2;
			 Pt_new.y=Pt_old.y+av_deltay2;

			cout << "Pt_new.x " << Pt_new.x << " Pt_old.x " << Pt_old.x << endl;
			
			line(imageToDraw, Pt_new, Pt_old, Scalar(255, 255, 255), 2);
			
			imshow( "Trajectory", imageToDraw );

			//Stitcher stitcher = Stitcher::createDefault(); 
	
			//Mat rImg;

			//vector< Mat > vImg; 

			//vImg.push_back(img_1);  
 			//vImg.push_back(img_2);  

			//Stitcher::Status status = stitcher.stitch(vImg, rImg);
			//if (Stitcher::OK == status)   
 			//imshow("Stitching Result",rImg);  
  			//else  
  			//printf("Stitching fail.");  

			 std::vector< Point2f > obj;
 			std::vector< Point2f > scene;

			for( int i = 0; i < good_matches.size(); i++ )
 			{
			 //-- Get the keypoints from the good matches
			 obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
			 scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
			 }
			 
			// Find the Homography Matrix
			 Mat H = findHomography( obj, scene, CV_RANSAC );
			 // Use the Homography Matrix to warp the images
			 cv::Mat result;
			 warpPerspective(img_1,result,H,cv::Size(img_1.cols+img_tot.cols,img_1.rows));
			 cv::Mat half(result,cv::Rect(0,0,img_tot.cols,img_tot.rows));
			 img_tot.copyTo(half);
			 imshow( "Result", result );


			


			Pt_old = Pt_new;
			img_1 = img_2;

		        waitKey(11);
		}
	}	


/**
 * @function main
 * @brief Main function
 */

int main( int argc, char** argv )
{
 // if( argc != 3 )
 // { 
 //   std::cout << "beh?" << std::endl;
 //   readme(); return -1; }
   
   initModule_nonfree();
   Pt_old.x = 320;
   Pt_old.y = 240;
   ros::init(argc, argv, "aerial_slam");
   ros::NodeHandle nh_;
   ros::Subscriber image_sub;

   image_sub = nh_.subscribe("/camera/image_raw",1,imageCallback);
   ros::spin();
   return 0;
}


