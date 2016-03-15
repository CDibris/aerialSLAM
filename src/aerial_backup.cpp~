
#include <ros/ros.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/flann/flann.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include </usr/include/opencv2/stitching/stitcher.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <cvaux.h>
#include <math.h>
#include <cxcore.h>

using namespace cv;
using namespace std;
using namespace cv::detail;

Point Pt_old, Pt_new;
int count1=0;
Mat img_1, img_2, img_tot, imageToDraw;

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
   			cout << "qui??" << endl;
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

		 Ptr<FeaturesFinder> finder;
		
		finder = new SurfFeaturesFinder();
    		

		Mat img_1_resize, img_2_resize;
		vector<ImageFeatures> features(2);
		vector<Mat> images(2);
		vector<Size> full_img_sizes(2);

		(*finder)(img_1, features[0]);
        	features[0].img_idx = 0;

		(*finder)(img_2, features[1]);
        	features[1].img_idx = 1;

		images[0]=img_1;
		images[1]=img_2;
        	//cout << "Features in image #0 : " << features[0].keypoints.size();
		finder->collectGarbage();

		vector<MatchesInfo> pairwise_matches;
    		BestOf2NearestMatcher matcher(false, 0.3);
    		matcher(features, pairwise_matches);
    		matcher.collectGarbage();

		 HomographyBasedEstimator estimator;
   		 vector<CameraParams> cameras;
    		estimator(features, pairwise_matches, cameras);

		for (size_t i = 0; i < cameras.size(); ++i)
    			{
       		 	Mat R;
        		cameras[i].R.convertTo(R, CV_32F);
       			 cameras[i].R = R;
   			 }
		
		Ptr<detail::BundleAdjusterBase> adjuster;
		adjuster = new detail::BundleAdjusterRay();
		 adjuster->setConfThresh(1.0);
                 Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);

		refine_mask(0,0) = 1;
		refine_mask(0,1) = 1;
		refine_mask(0,2) = 1;
		refine_mask(1,1) = 1;
		refine_mask(1,2) = 1;

		adjuster->setRefinementMask(refine_mask);
    		(*adjuster)(features, pairwise_matches, cameras);

		vector<double> focals;
   		 for (size_t i = 0; i < cameras.size(); ++i)
    		{
        	focals.push_back(cameras[i].focal);
   		 }

    	
		 sort(focals.begin(), focals.end());	
		float warped_image_scale;
    		if (focals.size() % 2 == 1)
        	warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
       		 warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;


		vector<Mat> rmats;
        	for (size_t i = 0; i < cameras.size(); ++i)
            	rmats.push_back(cameras[i].R);
        	waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
        	for (size_t i = 0; i < cameras.size(); ++i)
            	cameras[i].R = rmats[i];

		vector<Point> corners(2);
    		vector<Mat> masks_warped(2);
    		vector<Mat> images_warped(2);
    		vector<Size> sizes(2);
    		vector<Mat> masks(2);

      		 masks[0].create(img_1.size(), CV_8U);
       		 masks[0].setTo(Scalar::all(255));
    		 masks[1].create(img_2.size(), CV_8U);
       		 masks[1].setTo(Scalar::all(255));

		Ptr<WarperCreator> warper_creator;

		warper_creator = new cv::SphericalWarper();

		Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale));

		for (int i = 0; i < 2; ++i)
   		 {
        	Mat_<float> K;
        	cameras[i].K().convertTo(K, CV_32F);
        	float swa = (float)1.0;
        	K(0,0) *= swa; K(0,2) *= swa;
        	K(1,1) *= swa; K(1,2) *= swa;

        	corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        	sizes[i] = images_warped[i].size();

        	warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

		vector<Mat> images_warped_f(2);
    		for (int i = 0; i < 2; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    compensator->feed(corners, images_warped, masks_warped);

                imshow( "img_1", img_1 );
 		imshow( "img_2", img_2 );

	

		img_1=img_2;
		cvWaitKey(10);
		}

		
	}


int main( int argc, char** argv )
{
   initModule_nonfree();

   Pt_old.x = 408;
   Pt_old.y = 256;
   ros::init(argc, argv, "aerial_slam");
   ros::NodeHandle nh_;
   ros::Subscriber image_sub;

   image_sub = nh_.subscribe("/camera/image_raw",1,imageCallback);
   ros::spin();
   return 0;
}



