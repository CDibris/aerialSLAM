/**
Aerial SLAM (Image stitching + tracking + canny edge)
Author: Carmine RECCHIUTO, DIBRIS; UNIGE
*/
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

static IplImage *frame = NULL, *frame1_1C = NULL, *frame2_1C = NULL, *frame_save = NULL;
int count1 = 0;
Mat img_1, img_2, img_tot;
Point Pt_old, Pt_new;
Mat imageToDraw;
Mat detected_edges_2, detected_edges_1;

vector<string> img_names;
bool preview = false;
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

int edgeThresh = 1;
int lowThreshold=75;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

float find_min (float a, float b)
{
	if (a<b)
	return a;
	else
	return b;
}

float find_max (float a, float b)
{
	if (a<b)
	return b;
	else
	return a;
}

void printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching_detailed img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb)\n"
        "      Type of features used for images matching. The default is surf.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (reproj|ray)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}

int parseCmdArgs(int argc, char** argv)
{
  //  if (argc == 1)
  //  {
  //      printUsage();
  //      return -1;
  //  }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_gpu = true;
            else
            {
                cout << "Bad --try_gpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features = argv[i + 1];
            if (features == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}


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
			img_2=img_1;
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
			cout << "qui 2??" << endl;
   
			  if( !img_1.data || !img_2.data )
			  { std::cout<< " --(!) Error reading images " << std::endl;}
			int num_images = 2;

			double work_scale = 1, seam_scale = 1, compose_scale = 1;
    			bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

			LOGLN("Finding features...");
    int64 t = getTickCount();

    Ptr<FeaturesFinder> finder;
    if (features == "surf")
	 {
#ifdef HAVE_OPENCV_GPU
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            finder = new SurfFeaturesFinderGpu();
        else
#endif
            finder = new SurfFeaturesFinder();
    }
	 else if (features == "orb")
	 {
        finder = new OrbFeaturesFinder();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features << "'.\n";
        return;
    }

Mat img_1_resize, img_2_resize;
vector<ImageFeatures> features(num_images);
vector<Mat> images(num_images);
vector<Size> full_img_sizes(num_images);
double seam_work_aspect = 1;

    full_img_sizes[0] = img_1.size();
    full_img_sizes[1] = img_2.size();

	cout << "qui 3??" << endl;

    if (work_megapix < 0)
        {
            img_1_resize = img_1;
            work_scale = 1;
            is_work_scale_set = true;
	    img_2_resize = img_2;	
        }
	else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / img_1.size().area()));
                is_work_scale_set = true;
            }
            resize(img_1, img_1_resize, Size(), work_scale, work_scale);

	   if (!is_work_scale_set)
            {
              work_scale = min(1.0, sqrt(work_megapix * 1e6 / img_2.size().area()));
                is_work_scale_set = true;
            }
            resize(img_2, img_2_resize, Size(), work_scale, work_scale);


	    if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / img_1.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

	(*finder)(img_1_resize, features[0]);
        features[0].img_idx = 0;
        LOGLN("Features in image #0 : " << features[0].keypoints.size());

        resize(img_1, img_1_resize, Size(), seam_scale, seam_scale);
        images[0] = img_1_resize.clone();

	   if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / img_2.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

	(*finder)(img_2_resize, features[1]);
        features[1].img_idx = 1;
        LOGLN("Features in image #0 : " << features[1].keypoints.size());

        resize(img_2, img_2_resize, Size(), seam_scale, seam_scale);
        images[1] = img_2_resize.clone();
	    
        }
	
    cout << "qui 4??" << endl;
    finder->collectGarbage();
    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

 LOG("Pairwise matching");
    t = getTickCount();
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    LOGLN("pairwise_matches.size() = " << pairwise_matches.size() << "\n");
	for (size_t i=0; i!=pairwise_matches.size(); ++i) 
    {
        LOGLN("src_img_idx = " << pairwise_matches[i].src_img_idx  << "\t");
        LOGLN("dst_img_idx = " << pairwise_matches[i].dst_img_idx  << "\t");
        LOGLN("matches.size() = " << pairwise_matches[i].matches.size()  << "\t");
        LOGLN("num_inliers = " << pairwise_matches[i].num_inliers  << "\t");
        LOGLN("confidence = " << pairwise_matches[i].confidence  << "\n");
    }
    
	if (save_graph)
    {
        LOGLN("Saving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
//        for (int i=0; i!=pairwise_matches.size(); ++i)
//            f << pairwise_matches[i].src_img_idx << " " << pairwise_matches[i].dst_img_idx << "\n";
    }

vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<string> img_names_subset;
    vector<Size> full_img_sizes_subset;
  //  for (size_t i = 0; i < indices.size(); ++i)
  //  {
  //      img_names_subset.push_back(img_names[indices[i]]);
  //  cout << "qui 5??" << endl;
  //      img_subset.push_back(images[indices[i]]);
  //      full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
  //  }

  //  images = img_subset;
  //  img_names = img_names_subset;
  //  full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
 //   num_images = static_cast<int>(img_names.size());
  num_images=2;
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return;
    }

    LOG("Homography-based init\n");
    t = getTickCount();
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);
    LOGLN("Homography-based init, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
cout << "qui 5??" << endl;
 for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K());
    }

	LOG("Bundle Adjustment\n");
    t = getTickCount();
    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
	else 
    { 
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n"; 
        return; 
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);
    LOGLN("Bundle Adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

 vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i]+1 << ":\n" << cameras[i].K());
        focals.push_back(cameras[i].focal);
    }

 sort(focals.begin(), focals.end());
	float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

	LOGLN("Warping images (auxiliary)... ");
    t = getTickCount();

 
    vector<Point> corners_br(num_images);
    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

	 // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_GPU
    if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarperGpu();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarperGpu();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarperGpu();
    }

else
#endif
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
		else if (warp_type == "fisheye") warper_creator = new cv::FisheyeWarper();
		else if (warp_type == "stereographic") warper_creator = new cv::StereographicWarper();
		else if (warp_type == "compressedPlaneA2B1") warper_creator = new cv::CompressedRectilinearWarper(2, 1);
		else if (warp_type == "compressedPlaneA1.5B1") warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
		else if (warp_type == "compressedPlanePortraitA2B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
		else if (warp_type == "compressedPlanePortraitA1.5B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
		else if (warp_type == "paniniA2B1") warper_creator = new cv::PaniniWarper(2, 1);
		else if (warp_type == "paniniA1.5B1") warper_creator = new cv::PaniniWarper(1.5, 1);
		else if (warp_type == "paniniPortraitA2B1") warper_creator = new cv::PaniniPortraitWarper(2, 1);
		else if (warp_type == "paniniPortraitA1.5B1") warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
		else if (warp_type == "mercator") warper_creator = new cv::MercatorWarper();
		else if (warp_type == "transverseMercator") warper_creator = new cv::TransverseMercatorWarper();
    }

 if (warper_creator.empty())
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return;
    }

	 Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        cout << "corners x : " << corners[i].x << "   y : " << corners[i].y << endl;
        sizes[i] = images_warped[i].size();
        cout << "sizes height : " << sizes[i].height << "   width : " << sizes[i].width << endl;

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "voronoi")
        seam_finder = new detail::VoronoiSeamFinder();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_GPU
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_GPU
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    if (seam_finder.empty())
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
    t = getTickCount();

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

   for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / img_1.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                cout << "corner x : " << corners[i].x << "   y : " << corners[i].y << endl;

		corners_br[i] = roi.br();
                cout << "corner_br x : " << corners_br[i].x << "   y : " << corners_br[i].y << endl;
	
                sizes[i] = roi.size();
                cout << "size height : " << sizes[i].height << "   width : " << sizes[i].width << endl;
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(img_1, img_1_resize, Size(), compose_scale, compose_scale);
        else
            img_1 = img_1_resize;
        Size img_size = img_1.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img_1, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        
        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img_1.release();
        mask.release();
	img_2.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;



        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_gpu);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_gpu);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
                fb->setSharpness(1.f/blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	std::cout << corners[0] << " " << corners [1] << std::endl;
    }


    Mat result, result_mask;
    blender->blend(result, result_mask);

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    imwrite(result_name, result);

    Point Pt_old, Pt_new;
    
    float x_offset = find_min (corners[0].x, corners[1].x);
    float x_max = find_max (corners_br[0].x, corners_br[1].x); 
    float y_offset = find_min (corners[0].y, corners[1].y);
    float y_max = find_max (corners_br[0].y, corners_br[1].y);

    corners[0].x = corners[0].x - x_offset;
    corners_br[0].x = corners_br[0].x - x_offset;
    corners[1].x = corners[1].x - x_offset;
    corners_br[1].x = corners_br[1].x - x_offset;
    corners[0].y = corners[0].y - y_offset; 
    corners_br[0].y = corners_br[0].y - y_offset;
    corners[1].y = corners[1].y - y_offset;
    corners_br[1].y = corners_br[1].y - y_offset;

  //  Pt_old.x = (corners[0].x + corners_br[0].x)/2;
  //  Pt_old.y = (corners[0].y + corners_br[0].y)/2;
    Pt_new.x = (corners[1].x + corners_br[1].x)/2;
    Pt_new.y = (corners[1].y + corners_br[1].y)/2;

    //Pt_old.x= (corners[0].x+sizes[0].width)/2;
    //Pt_old.y= (corners[0].y+sizes[0].height)/2;
    // Pt_new.x= (corners[1].x+sizes[1].width)/2;
  
    cout << result.size() << endl;

    line(result, Pt_new, Pt_old, Scalar(255, 255, 255), 2);
    imwrite( "/home/carmine/Desktop/test.jpg", result );
	
    imwrite( "/home/carmine/Desktop/test1.jpg", img_1 );
    imwrite( "/home/carmine/Desktop/test2.jpg", img_2 );


    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

	Pt_old = Pt_new;
	img_1 = result;

	// waitKey(5);
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
   
   int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

   Pt_old.x = 320;
   Pt_old.y = 240;
   ros::init(argc, argv, "aerial_slam");
   ros::NodeHandle nh_;
   ros::Subscriber image_sub;

   image_sub = nh_.subscribe("/camera/image_raw",1,imageCallback);
   ros::spin();
   return 0;
}


