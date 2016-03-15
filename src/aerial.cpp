///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//
//M*/

#include <ros/ros.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <unistd.h>
//#include <unistd.h>
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
#include <std_msgs/Int32.h>
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


using namespace std;
using namespace cv;
using namespace cv::detail;


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



static void printUsage()
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


// Default command line args
vector<String> img_names;
bool preview = true;     // era false
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "orb"; // era "surf
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = false;   // era true
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "plane"; // era spherical
int expos_comp_type = ExposureCompensator::NO; // era GAIN_BLOCKS
float match_conf = 0.3f; // era 0.3
string seam_find_type = "gc_color";
int blend_type = Blender::NO; // era multi_band
float blend_strength = 5;
string result_name;

Point Pt_new[10];


void initializeDataSaving(){
  
    ofstream fs2("localization.txt", std::ofstream::out | std::ofstream::trunc);
    
    if (fs2.is_open()){
        fs2 << "image n \t";
        fs2 << "x \t";
	fs2 << "y \t";
       
        fs2 << "\n";
        
        cout << "File with controls initialized" << endl;
    }
    else {
        cerr << "File not opened!" << endl;
    }
}






static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
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
            features_type = argv[i + 1];
            if (features_type == "orb")
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
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
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


void imageCallback(const std_msgs::Int32ConstPtr& msg)
	{
#if ENABLE_LOG
    int64 app_start_time = getTickCount();
#endif

    cv::setBreakOnError(true);
   
	if (msg->data == 0)
	{
	char str[80];
	cout << "First image ok.. waiting for the second one" << endl;
        sprintf (str, "/home/carmine/Desktop/photo_folder/img_%d.png",msg->data);
	img_names.push_back(str);
	}
	else
	{
	char str[80];
	sprintf (str, "/home/carmine/Desktop/photo_folder/img_%d.png", msg->data);
	cout << "second image arrived" << endl;
	img_names.push_back(str);
	

   // int retval = parseCmdArgs(argc, argv);
   // if (retval)
   //     return retval;

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
	cout << "not enough images!!" << endl;
        LOGLN("Need more images");
        return;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<FeaturesFinder> finder;
    if (features_type == "surf")
    {
#ifdef HAVE_OPENCV_NONFREE
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            finder = new SurfFeaturesFinderGpu();
        else
#endif
            finder = new SurfFeaturesFinder();
    }
    else if (features_type == "orb")
    {
        finder = new OrbFeaturesFinder();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return;
    }
   cout << "going on ... 0.1" << endl;
    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);

        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));  
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        (*finder)(img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    finder->collectGarbage();
    full_img.release();
    img.release();

	cout << "going on ... 1" << endl;

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

  cout << "going on ... 1.1" << endl;

    // Check if we should save matches graph
    if (save_graph)
    {
        LOGLN("Saving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // Leave only images we are sure are from the same panorama --- questa cosa qui la puoi eliminare
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;

    cout << "going on ... 1.2" << endl; 
    cout << "num images " << num_images << endl;

   for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
	 cout << "num images after! " << num_images << endl;
    if (num_images < 2)
    {
	//img_names.push_back(str);
        LOGLN("Need more images");
	
        return;
    }

	cout << "going on ... 1.3" << endl;

    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);         ///queste 3 operazioni le puoi fare solo la prima volta?

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K());
    }

	cout << "going on ... 1.4" << endl;

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
    cout << "going on ... 1.5" << endl;     // puoi levare il bundle adjustment per accelerare i tempi
    // Find median focal length

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

     cout << "going on ... 1.6" << endl;

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
#if ENABLE_LOG
    t = getTickCount();
#endif


    vector<Point> corners_br(num_images);
    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

	cout << "going on ... 2" << endl;

    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_GPUWARPING
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
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
   
    cout << "going on ... 3" << endl;
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
   compensator->feed(corners, images_warped, masks_warped);
   cout << "going on ... 3.1" << endl;
    Ptr<SeamFinder> seam_finder;
	cout << "going on ... 3.2" << endl;
    if (seam_find_type == "no")
        {seam_finder = new detail::NoSeamFinder();
	cout << "going on ... 3.25" << endl;}
    else if (seam_find_type == "voronoi")
        {seam_finder = new detail::VoronoiSeamFinder();
	cout << "going on ... 3.25" << endl;}
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_GPU
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            {seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
		cout << "going on ... 3.3" << endl;}
        else
#endif
           { seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
		cout << "going on ... 3.3" << endl;		}
    }
	
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_GPU
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            {seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		cout << "going on ... 3.4" << endl;}
        else
#endif
            {seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		cout << "going on ... 3.4" << endl;}
    }
	
    else if (seam_find_type == "dp_color")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    if (seam_finder.empty())
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return;
    }
    cout << "going on ... 3.5" << endl;
    seam_finder->find(images_warped_f, corners, masks_warped);
    
    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

	 cout << "going on ... 4" << endl;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);
	
		// -----------------------------------//
	//while (!img_names.empty())
  	//{
    	//sum+=myvector.back();
    	//img_names.pop_back();
  	//}
		/// ------------------------ /// test test test
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
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
		corners_br[i] = roi.br();
		cout << "i: corner -> " << " x: " << corners[i].x << " y: " << corners[i].y << endl;
		cout << "i: corner_br -> " << " x: " << corners_br[i].x << " y: " << corners_br[i].y << endl;
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

	cout << "going on ... 5" << endl;

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
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);
   
    cout << "saving first elab image " << endl;
    
    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
 //   char resname[80];
// sprintf (resname, "/home/carmine/Desktop/photo_folder/elab%d.png", msg->data);
 
  // imwrite(resname, result);	
	

   // img_names.push_back(str);  //// TTTTTEEEEST!!!!




	// tracking
 
    //Point Pt_new;
    
   // float x_offset = find_min (corners[0].x, corners[1].x);
   // float x_max = find_max (corners_br[0].x, corners_br[1].x); 
   // float y_offset = find_min (corners[0].y, corners[1].y);
   // float y_max = find_max (corners_br[0].y, corners_br[1].y);

   // corners[0].x = corners[0].x - x_offset;
   // corners_br[0].x = corners_br[0].x - x_offset;
   // corners[1].x = corners[1].x - x_offset;
   // corners_br[1].x = corners_br[1].x - x_offset;
   // corners[0].y = corners[0].y - y_offset; 
   // corners_br[0].y = corners_br[0].y - y_offset;
   // corners[1].y = corners[1].y - y_offset;
   // corners_br[1].y = corners_br[1].y - y_offset;

    //Pt_old.x = (corners[0].x + corners_br[0].x)/2;
    //Pt_old.y = (corners[0].y + corners_br[0].y)/2;

for (int i = 0; i < num_images; i++)
            {
    Pt_new[i].x = (corners[i].x + corners_br[i].x)/2;
    Pt_new[i].y= (corners[i].y + corners_br[i].y)/2;	
	     }


     int min_x=Pt_new[0].x;
     int min_y=Pt_new[0].y;
///trova il minimo per l'offset!

for (int i = 1; i < num_images; i++)
            {
	           if (Pt_new[i].x < min_x)
			{
			min_x =Pt_new[i].x;
			}	
		   if (Pt_new[i].y < min_y)
			{
			min_y =Pt_new[i].y;
			}
	     }

for (int i = 0; i < num_images; i++)
            {
    Pt_new[i].x = Pt_new[i].x - min_x+200;
    Pt_new[i].y= Pt_new[i].y - min_y+225;	
	     }

     //cout << msg->data << ": PTNEW: " << Pt_new[msg->data].x << " " << Pt_new[msg->data].y << endl;
	// tentativo 	//

	//if (Pt_new[msg->data -1].x!=0)
				//{
    //Pt_new[msg->data].x = Pt_new[msg->data -1].x + (corners_br[1].x - corners_br[0].x);
    //Pt_new[msg->data].y = Pt_new[msg->data -1].y + (corners_br[1].y - corners_br[0].y);
				
 //  cout << "PTNEW: " << Pt_new.x << " " << Pt_new.y << endl;
 //  cout << "PTOLD: " << Pt_old.x << " " << Pt_old.y << endl;
     //cout << msg->data << ": corner_x_old: " << corners_br[1].x << " corner_x_new " << corners_br[0].x << endl; 
	//cout << msg->data << ": corner_y_old: " << corners_br[1].y << " corner_y_new " << corners_br[0].y << endl; 
     //cout << msg->data << ": PTNEW: " << Pt_new[msg->data].x << " " << Pt_new[msg->data].y << endl;
	//	}
 //   Pt_old.x= (corners[0].x+sizes[0].width)/2;
 //   Pt_old.y= (corners[0].y+sizes[0].height)/2;
 //   Pt_new[msg->data].x= (corners[1].x+sizes[1].width)/2;
 //   Pt_new[msg->data].y= (corners[1].y+sizes[1].height)/2;
   
  //  cout << msg->data << ": PTNEW: " << Pt_new[msg->data].x << " " << Pt_new[msg->data].y << endl;

// cout << result.size() << endl;

    cout << ": PTNEW_0: " << Pt_new[0].x << " " << Pt_new[0].y << endl;

    for (int i=1; i<num_images; i++)
	{
        //int bu=0; 
	//if ((Pt_new[i].x!=0))
		{
	//	for (int j=0; j<=i-1; j++)
	//		{
	//		if (Pt_new[j].x!=0)
	//			{
	//			bu=j;
				//cout << "but: " << bu << endl;
			//	}
		//	}
	//if (Pt_new[i].x < 230)
	//	{
		

	//	}
        cout << i << ": PTNEW: " << Pt_new[i].x << " " << Pt_new[i].y << endl;
	line(result, Pt_new[i], Pt_new[i-1], Scalar(255, 255, 255), 2);
     
		}
	}
	
// stampare spostamento
     ofstream fs2("localization.txt", std::ofstream::out | std::ofstream::app);

    
    if(fs2.is_open()){
         fs2 << msg->data << "\t";
       for (int i=1; i<num_images; i++)
	{
	fs2 << Pt_new[i].x-Pt_new[i-1].x << "\t";
	fs2 << Pt_new[i].y-Pt_new[i-1].y << "\n";
	}
	}
else {
        std::cerr << "ERRORE nella scrittura del file!" << std::endl;
    }



    char resname2[80];
   sprintf (resname2, "/home/carmine/Desktop/photo_folder/elab_track%d.png", msg->data);

   imwrite(resname2, result);
//    Mat boh;
 
 //  boh = imread(resname2);
	//usleep (500);
	//namedWindow("Image");
 
       // imshow("Image",boh); 
	// waitKey (1);
	       
	//added for testing canny edge detection//
 //       Mat gray, edge, result2;
 //    	cvtColor(boh, gray, CV_BGR2GRAY);
 //   	Canny( gray, edge, 200, 150, 3);
 //   	edge.convertTo(result2, CV_8U);

//char resname3[80];
//   sprintf (resname3, "/home/carmine/Desktop/photo_folder/elab_track_canny%d.png", msg->data);
    //cout << "size: " <<  result.size() << endl;
// imwrite(resname3, result2);
 
  
	
   // imwrite( "/home/carmine/Desktop/test1.jpg", img_1 );
   // imwrite( "/home/carmine/Desktop/test2.jpg", img_2 );


    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return;
}
}




int main( int argc, char** argv )
{
   Pt_new[0].x = 240; //200
   Pt_new[0].y = 376; //225
   initModule_nonfree();
   ros::init(argc, argv, "aerial");
   ros::NodeHandle nh_;
   ros::Subscriber image_sub;

   image_sub = nh_.subscribe("/sync",100,imageCallback);
   ros::spin();
   return 0;
}


