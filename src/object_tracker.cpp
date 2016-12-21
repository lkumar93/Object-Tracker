//
// THIS IS AN IMPLEMENTATION OF OBJECT TRACKING USING OPENCV'S CASCADE
// CLASSIFIER AND PCL'S KLD ADAPTIVE PARTICLE FILTER IMPLEMENTATION
// WITH DRIFT/OUT OF VIEW DETECTION AND RELOCALIZATION CAPABILITIES
//
// COPYRIGHT BELONGS TO THE AUTHOR OF THIS CODE
//
// AUTHOR : LAKSHMAN KUMAR
// AFFILIATION : UNIVERSITY OF MARYLAND, MARYLAND ROBOTICS CENTER
// EMAIL : LKUMAR93@UMD.EDU
// LINKEDIN : WWW.LINKEDIN.COM/IN/LAKSHMANKUMAR1993
//
// THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THE MIT LICENSE
// THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF
// THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
// 
// BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
// BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
// CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
// CONDITIONS.
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <string>

#include <ros/ros.h>

#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
#include <pcl/search/pcl_search.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/Polygon.h>

#include<object_tracker/object_tracker.h>

using namespace std;
using namespace sensor_msgs;
using namespace pcl;
using namespace pcl::tracking;

typedef cv::CascadeClassifier ClassifierT;
typedef cv::Mat ImageT;
typedef cv_bridge::CvImageConstPtr ImagePtr;
typedef cv::Rect_<int> RectangleT;

typedef pcl::ParticleXYZRPY ParticleT;
typedef pcl::PointXYZRGB PointT;
typedef std::vector<PointT, Eigen::aligned_allocator<PointT>> PointsT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef PointCloudT::Ptr PointCloudPtr;
typedef PointCloudT::ConstPtr PointCloudConstPtr;

typedef ParticleFilterTracker<PointT, ParticleT> ParticleFilterT;
typedef KLDAdaptiveParticleFilterOMPTracker<PointT, ParticleT> KLDParticleFilterT;



struct PointCloudDataT
{
	PointCloudPtr cloud_ptr;
	int r_avg = 0;
	int g_avg = 0;
	int b_avg = 0;
	int number_of_points = 0;
	double x_center = 0.0;
	double y_center = 0.0;
	double z_center = 0.0;
	double length = 0.0;
	double breadth = 0.0;
	double height = 0.0;
	bool is_valid = false;

};

class object_tracker
{
   classifierT object_detector;

   ros::Subscriber image_subscriber ; 
   ros::Subscriber pointcloud_subscriber ; 

   bool image_initialized = false;
   bool pointcloud_initialized = false;

   bool object_detected = false;

   PointCloudPtr input_pointcloud;
   PointCloudPtr object_pointcloud;

   ImageT input_image;
   ImageT input_image_grayscale;
   ImagePtr input_image_ptr;

   RectangleT ref_object_rect;

   int input_image_height = 0;
   int input_image_width = 0;

   double depth_offset;
   double length_offset;
   double height_offset;

   struct PointCloudDataT target_cloud_data;

   boost::shared_ptr<ParticleFilterT> particle_tracker;
ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr coherence; 
      
   public :
	void object_tracker(ClassifierT detector, string image_topic, string pointcloud_topic,Eigen::Vector4f offset)
	{
		ros::NodeHandle nh;
	
		object_detector = detector;

		length_offset = offset[0];
		height_offset = offset[1]; 
		depth_offset = offset[2];

		image_subscriber =  nh.subscribe(image_topic, 1, &object_tracker::image_callback, this);

		pointcloud_subscriber =  nh.subscribe(pointcloud_topic, 1, &object_tracker::pointcloud_callback, this);	
	}

	 void image_callback(const sensor_msgs::ImageConstPtr& image )
  	{
		try
	    	{
	      		input_image_ptr = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
			
	    	}
	   	catch (cv_bridge::Exception& e)
	    	{
	      		ROS_ERROR("cv_bridge exception: %s", e.what());
			image_initialized = false;
	      		return;
	    	}	

		image_initialized = true;
		
		input_image = input_image_ptr->image;

		input_image_height = input_image.rows;
		input_image_width = input_image.cols;
		
	}

	 void pointcloud_callback( const sensor_msgs::PointCloud2ConstPtr& pointcloud )
  	{
	
		try
		{
			pcl::fromROSMsg (*pointcloud, *input_pointcloud);
		}

		catch(const std::exception& e)
		{
			ROS_ERROR("pointcloud_callback exception: %s", e.what());
			pointcloud_initialized = false;
	      		return;
		}

		catch(...)
		{
			ROS_ERROR("Unexpected expection in pointcloud_callback");
			pointcloud_initialized = false;
	      		return;
		}

		pointcloud_initialized = true;
	}


	bool object_detection()
	{
		if(image_initialized && pointcloud_initialized)
		{
			std::vector<RectangleT> objects;
			input_image_grayscale = ImageT(image_height,image_width, CV8UC3,cv::Scalar(0, 0, 0));
			object_detector.detectMultiScale(input_image_grayscale, objects);

			if(!object_detected)
			{
			    int largest_area = 0 ;

			    for (auto object : objects)
    			    {
				int area = object.width * object.height;
	
				struct PointCloudDataT extracted_cloud_data;

				extracted_cloud_data = extract_cloud(object)

				if(extracted_cloud_data.is_valid)
				{
					if(area > largest_area)
					{
						largest_area = area;
						target_cloud_data = extracted_cloud_data;
						object_detected = true;	
					}
				}								
			     }
			}
			else
			{
			}		
		}

		return object_detected;			
	}

	PointCloudDataT extract_cloud(RectangleT ref_rect)
	{

		int r_value, g_value ,b_value;

		float object_depth[5];
	
		struct PointCloudDataT ref_cloud_data;
		
		ref_cloud_data.cloud_ptr(new PointCloudT ());

		int cloud_width = input_pointcloud->width;
		int cloud_height = input_pointcloud->height;

		int center_x = ref_rect.x + ref_rect.width/2 ;
		int center_y = ref_rect.y + ref_rect.height/2 ;

		double center_object_depth = 0.0;

		PointsT input_points = input_pointcloud->points;

		object_depth[0]	= input_points.at(center_x + center_y*cloud_width).z;
		object_depth[1]	= input_points.at(center_x + 1 + center_y*cloud_width).z;
		object_depth[2]	= input_points.at(center_x - 1 + center_y*cloud_width).z;
		object_depth[3]	= input_points.at(center_x + (center_y + 1)*cloud_width).z;
		object_depth[4]	= input_points.at(center_x + (center_y - 1)*cloud_width).z;

		for (int i = 0; i< 5;i++)
		{
			if(object_depth[i] > center_object_depth)
			{
				center_object_depth = object_depth[i];
			}

		}

		double min_z_value = center_object_depth - depth_offset;
		double max_z_value = center_object_depth + depth_offset;

		PointT input_point;
	
		for ( int i = 0 ; i< cloud_width ; i++)
		{
				
			for ( int j = 0 ; j< cloud_height ; j++)
			{

		 		if( ref_rect.contains(cv::Point(i,j)) )
			        {
					int index = i+j*cloud_width ;

					input_point = input_points.at(index) ;

					r_value = input_point.r ;
					g_value = input_point.g ;
					b_value = input_point.b ;
				
					if(z_value >= min_z_value && z_value <= max_z_value )
					{	
						ref_cloud_data.r_avg += r_value;
						ref_cloud_data.g_avg += g_value;
						ref_cloud_data.b_avg += b_value;
						ref_cloud_data.cloud_ptr->points.push_back(input_point);
						ref_cloud_data.number_of_points++;
					}
				  }
			 }
	         }

		if ( ref_cloud_data.number_of_points > 5 )
		{

			ref_cloud_data.r_avg /= ref_cloud_data.number_of_points;
			ref_cloud_data.g_avg /= ref_cloud_data.number_of_points;
			ref_cloud_data.b_avg /= ref_cloud_data.number_of_points;

			Eigen::Vector4f ref_cloud_centroid;

			pcl::compute3DCentroid<PointT> (*ref_cloud_data.cloud_ptr, ref_cloud_centroid);

			ref_cloud_data.x_center = ref_cloud_centroid[0];
			ref_cloud_data.y_center = ref_cloud_centroid[1];
			ref_cloud_data.z_center = ref_cloud_centroid[2];

			PointT point_min;
			PointT point_max;

			pcl::getMinMax3D(*ref_cloud_data.cloud_ptr,point_min,point_max);

			ref_cloud_data.length = abs( point_max.x - point_min.x );
			ref_cloud_data.breadth = abs( point_max.z - point_min.z );
			ref_cloud_data.height = abs( point_max.y - point_min.y );
		
			ref_cloud_data.is_valid = true;			
		
		}

		else
		{
			ref_cloud_data.is_valid = false;	
		}
		
		return ref_cloud_data ; 

	} 

	void downsample(const PointCloudConstPtr &cloud, PointCloudT &result, double leaf_size)
	{

		//Downsample
		pcl::ApproximateVoxelGrid<PointT> grid;
		grid.setLeafSize (static_cast<float> (leaf_size), static_cast<float> (leaf_size), static_cast<float> (leaf_size));
		grid.setInputCloud (cloud);
		grid.filter (result);
	}

	void init_particle_filter()
	{

	 // Particle Filter Initialization

	  downsampling_grid_size_ =  0.01;

	  std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
	  default_step_covariance[3] *= 40.0;
	  default_step_covariance[4] *= 40.0;
	  default_step_covariance[5] *= 40.0;

	  std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.00002);
	  std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);

	  boost::shared_ptr<KLDParticleFilterT> kld_tracker (new KLDParticleFilterT(8));

	  ParticleT bin_size;

	  bin_size.x = 0.1f;
	  bin_size.y = 0.1f;
	  bin_size.z = 0.1f;
	  bin_size.roll = 0.1f;
	  bin_size.pitch = 0.1f;
	  bin_size.yaw = 0.1f;

	 //Set all parameters for  KLDAdaptiveParticleFilterOMPTracker

	  kld_tracker->setMaximumParticleNum (500);
	  kld_tracker->setDelta (0.99);
	  kld_tracker->setEpsilon (0.2);
	  kld_tracker->setBinSize (bin_size);

	 //Set all parameters for  ParticleFilter

	  particle_tracker = kld_tracker;
	  particle_tracker->setTrans (Eigen::Affine3f::Identity ());
	  particle_tracker->setStepNoiseCovariance (default_step_covariance);
	  particle_tracker->setInitialNoiseCovariance (initial_noise_covariance);
	  particle_tracker->setInitialNoiseMean (default_initial_mean);
	  particle_tracker->setIterationNum (1);
	  particle_tracker->setParticleNum (500);
	  particle_tracker->setResampleLikelihoodThr(0.00);
	  particle_tracker->setUseNormal (false);

	 //Setup coherence object for tracking

	  coherence = ApproxNearestPairPointCloudCoherence<PointT>::Ptr
	  (new ApproxNearestPairPointCloudCoherence<PointT> ());
				    
	  boost::shared_ptr<DistanceCoherence<PointT> > distance_coherence
				    = boost::shared_ptr<DistanceCoherence<PointT> > (new DistanceCoherence<PointT> ());
	  distance_coherence->setWeight(0.5);
	  coherence->addPointCoherence (distance_coherence);

	  boost::shared_ptr<HSVColorCoherence<PointT> > color_coherence
	  = boost::shared_ptr<HSVColorCoherence<PointT> > (new HSVColorCoherence<PointT> ());
	  color_coherence->setWeight (0.5);
	  coherence->addPointCoherence (color_coherence);

	  boost::shared_ptr<pcl::search::Octree<PointT> > search (new pcl::search::Octree<PointT> (0.01));
	  coherence->setSearchMethod (search);
	  coherence->setMaximumDistance (0.03);

	  particle_tracker->setCloudCoherence (coherence);

	  //prepare the model of tracker's target
	  Eigen::Vector4f c;
	  Eigen::Affine3f transform = Eigen::Affine3f::Identity ();

	  PointCloudPtr transformed_cloud (new PointCloudT());
	  PointCloudPtr transformed_cloud_downsampled (new PointCloudT());

	  pcl::compute3DCentroid<PointT> (*target_cloud, c);
	  transform.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
	  pcl::transformPointCloud<PointT> (*target_cloud, *transformed_cloud, transform.inverse());

	  downsample(transformed_cloud, *transformed_cloud_downsampled, downsampling_grid_size_);

	  //set reference model and trans
	  particle_tracker->setReferenceCloud (transformed_cloud_downsampled);
	  particle_tracker->setTrans (trans);

	}


	PointCloudDataT crop_cloud(const PointCloudConstPtr &cloud, double x_prev, double y_prev, double z_prev)
	{

		int r_value, g_value ,b_value;
		double x_value,y_value,z_value;

		float object_depth[5];
	
		struct PointCloudDataT cropped_cloud_data;
		
		cropped_cloud_data.cloud_ptr(new PointCloudT ());

		int cloud_width = cloud->width;
		int cloud_height = cloud->height;

		double min_x_value = x_prev - length_offset ;
		double max_x_value = x_prev + length_offset ;

		double min_y_value = y_prev - height_offset ;
		double max_y_value = y_prev + height_offset ;

		double center_object_depth = 0.0;

		PointsT input_points = cloud->points;
	
		for ( int i = 0 ; i< (cloud->width) ; i++)
	    	{				
			for ( int j = 0 ; j< (cloud->height) ; j++)
			{

				int index = i+j*cloud_width ;

				input_point = input_points.at(index) ;

				r_value = input_point.r ;
				g_value = input_point.g ;
				b_value = input_point.b ;

				x_value = input_point.x ;
				y_value = input_point.y ;
				z_value = input_point.z ;

				// Y value might be inverted , check
		
			    	if(x_value > min_x_value && x_value < max_x_value && y_value > min_y_value && y_value < max_y_value)
				{
					cropped_cloud_data.r_avg += r_value;
					cropped_cloud_data.g_avg += g_value;
					cropped_cloud_data.b_avg += b_value;
					cropped_cloud_data.cloud_ptr->points.push_back(input_point);
					cropped_cloud_data.number_of_points++;
				}
			}
	    	}

		if ( cropped_cloud_data.number_of_points > 5 )
		{

			cropped_cloud_data.r_avg /= cropped_cloud_data.number_of_points;
			cropped_cloud_data.g_avg /= cropped_cloud_data.number_of_points;
			cropped_cloud_data.b_avg /= cropped_cloud_data.number_of_points;

			Eigen::Vector4f cropped_cloud_centroid;

			pcl::compute3DCentroid<PointT> (*cropped_cloud_data.cloud_ptr, cropped_cloud_centroid);

			cropped_cloud_data.x_center = ref_cloud_centroid[0];
			cropped_cloud_data.y_center = ref_cloud_centroid[1];
			cropped_cloud_data.z_center = ref_cloud_centroid[2];

			PointT point_min;
			PointT point_max;

			pcl::getMinMax3D(*ref_cloud_data.cloud_ptr,point_min,point_max);

			cropped_cloud_data.length = abs( point_max.x - point_min.x );
			cropped_cloud_data.breadth = abs( point_max.z - point_min.z );
			cropped_cloud_data.height = abs( point_max.y - point_min.y );
		
			cropped_cloud_data.is_valid = true;			
		
		}

		else
		{
			cropped_cloud_data.is_valid = false;	
		}
		
		return cropped_cloud_data ; 

	
	}

	void track()
	{
		//Declare the minimum point to crop the input cloud

		Eigen::Vector4f minPoint; 
		minPoint[0]=PrevX+MinXOffset;  // define minimum point x
		minPoint[1]=PrevY+MinYOffset;  // define minimum point y
		minPoint[2]=PrevZ+MinZOffset;  // define minimum point z 

		//Declare the maximum point to crop the input cloud
		Eigen::Vector4f maxPoint;
		maxPoint[0]=PrevX+MaxXOffset;  // define max point x
		maxPoint[1]=PrevY+MaxYOffset;  // define max point y
		maxPoint[2]=PrevZ+MaxZOffset;  // define max point z


	}


};

int main( int argc, char** argv )
{    
    //Initialize the object tracker node
    ros::init(argc, argv, "object_tracker_node");

   // ros::NodeHandle nh;
    object_tracker hand_tracker("bottom", NumberOfFeaturesToUse, FocalLength_X, FocalLength_Y, PrincipalPoint_X, PrincipalPoint_Y, Distortion);
    ros::spin();
    return 0;
}
