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
 #include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/transforms.h>

//#include<object_tracker/object_tracker.h>

using namespace std;
using namespace sensor_msgs;
using namespace pcl;
using namespace pcl::tracking;

typedef cv::CascadeClassifier ClassifierT;
typedef cv::Mat ImageT;
typedef cv_bridge::CvImageConstPtr ImagePtrT;
typedef cv::Rect_<int> RectangleT;

typedef ParticleXYZRPY ParticleT;
typedef pcl::PointXYZRGB PointT;
typedef std::vector<PointT, Eigen::aligned_allocator<PointT>> PointsT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef PointCloudT::Ptr PointCloudPtrT;
typedef PointCloudT::ConstPtr PointCloudConstPtrT;

typedef ParticleFilterTracker<PointT, ParticleT> ParticleFilterT;
typedef ParticleFilterT::PointCloudStatePtr ParticleCloudPtrT;
typedef KLDAdaptiveParticleFilterOMPTracker<PointT, ParticleT> KLDParticleFilterT;



struct PointCloudDataT
{
	PointCloudPtrT cloud_ptr;
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
   ClassifierT object_detector;

   ros::Subscriber image_subscriber ; 
   ros::Subscriber point_cloud_subscriber ; 

   ros::Publisher object_cloud_pub,object_centroid_pub,tracked_cloud_pub;

   bool image_initialized = false;
   bool point_cloud_initialized = false;

   bool object_detected = false;

   PointCloudPtrT input_point_cloud ;
   PointCloudPtrT object_point_cloud;
   PointCloudPtrT tracked_point_cloud;
   PointCloudPtrT target_point_cloud;
   PointCloudPtrT object_particle_cloud;

   ImageT input_image;
   ImageT input_image_grayscale;
   ImagePtrT input_image_ptr;

   RectangleT ref_object_rect;

   int input_image_height = 0;
   int input_image_width = 0;

   double depth_offset,length_offset,height_offset,object_offset;
   double prev_x, prev_y, prev_z;


   PointCloudDataT target_cloud_data;

   boost::shared_ptr<ParticleFilterT> particle_tracker;
   ApproxNearestPairPointCloudCoherence<PointT>::Ptr coherence; 

   geometry_msgs::PointStamped object_centroid_msg;
      
   public :
	object_tracker(ClassifierT detector, string image_topic, string point_cloud_topic,Eigen::Vector4f offset)
	{
		ros::NodeHandle nh;
	
		object_detector = detector;

		length_offset = offset[0];
		height_offset = offset[1]; 
		depth_offset = offset[2];
		object_offset = offset[3];

//		*input_point_cloud = new PointCloudT();
//   		*object_point_cloud = new PointCloudT();
//   		*tracked_point_cloud = new PointCloudT();
//  		*target_point_cloud = new PointCloudT();
//   		*object_particle_cloud = new PointCloudT ();

		image_subscriber =  nh.subscribe(image_topic, 1, &object_tracker::image_callback, this);

		point_cloud_subscriber =  nh.subscribe(point_cloud_topic, 1, &object_tracker::point_cloud_callback, this);

		object_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> (point_cloud_topic+"/object_cloud", 1); 

		tracked_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> (point_cloud_topic+"/tracked_cloud", 1); 	

	        object_centroid_pub = nh.advertise<geometry_msgs::PointStamped> (point_cloud_topic+"/object_centroid", 1);

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

	 void point_cloud_callback( const sensor_msgs::PointCloud2ConstPtr& point_cloud )
  	{
	
		try
		{
			pcl::fromROSMsg (*point_cloud, *input_point_cloud);
		}

		catch(const std::exception& e)
		{
			ROS_ERROR("point_cloud_callback exception: %s", e.what());
			point_cloud_initialized = false;
	      		return;
		}

		catch(...)
		{
			ROS_ERROR("Unexpected expection in point_cloud_callback");
			point_cloud_initialized = false;
	      		return;
		}

		point_cloud_initialized = true;

		run();
	}


	bool object_detection(bool redetect = false)
	{
		if(image_initialized && point_cloud_initialized)
		{
			std::vector<RectangleT> objects;
			input_image_grayscale = ImageT(input_point_cloud->height,input_point_cloud->width, CV_8UC1,cv::Scalar(0, 0, 0));
			cvtColor(input_image, input_image_grayscale, CV_BGR2GRAY);
			object_detector.detectMultiScale(input_image_grayscale, objects);

			if(!object_detected)
			{
			    int largest_area = 0 ;

			    for (auto object : objects)
    			    {
				int area = object.width * object.height;
	
				PointCloudDataT extracted_cloud_data;

				extracted_cloud_data = extract_cloud(object);

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
		
		}

		else
		{

			object_detected = false;

		}

		return object_detected;			
	}

	PointCloudDataT extract_cloud(RectangleT ref_rect)
	{

		int r_value, g_value ,b_value;

		float object_depth[5];
	
		PointCloudDataT ref_cloud_data;
		
//		ref_cloud_data.cloud_ptr(new PointCloudT ());

		int cloud_width = input_point_cloud->width;
		int cloud_height = input_point_cloud->height;

		int center_x = ref_rect.x + ref_rect.width/2 ;
		int center_y = ref_rect.y + ref_rect.height/2 ;

		double center_object_depth = 0.0;

		PointsT input_points = input_point_cloud->points;

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

		double min_z_value = center_object_depth - object_offset;
		double max_z_value = center_object_depth + object_offset;

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
				
					if(center_object_depth >= min_z_value && center_object_depth <= max_z_value )
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

			prev_x = ref_cloud_centroid[0];
			prev_y = ref_cloud_centroid[1];
			prev_z = ref_cloud_centroid[2];

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

	void downsample(const PointCloudConstPtrT &cloud, PointCloudT &result, double leaf_size)
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

	  float downsampling_grid_size =  0.01;

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

	  PointCloudPtrT transformed_cloud (new PointCloudT());
	  PointCloudPtrT transformed_cloud_downsampled (new PointCloudT());

	  pcl::compute3DCentroid<PointT> (*target_point_cloud, c);
	  transform.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
	  pcl::transformPointCloud<PointT> (*target_point_cloud, *transformed_cloud, transform.inverse());

	  downsample(transformed_cloud, *transformed_cloud_downsampled, downsampling_grid_size);

	  //set reference model and trans
	  particle_tracker->setReferenceCloud (transformed_cloud_downsampled);
	  particle_tracker->setTrans (transform);

	}


	PointCloudDataT crop_cloud(const PointCloudConstPtrT &cloud, double x_prev, double y_prev, double z_prev)
	{

		int r_value, g_value ,b_value;
		double x_value,y_value,z_value;
	
		PointCloudDataT cropped_cloud_data;
		
		//cropped_cloud_data.cloud_ptr(new PointCloudT ());

		int cloud_width = cloud->width;
		int cloud_height = cloud->height;

		double min_x_value = x_prev - length_offset ;
		double max_x_value = x_prev + length_offset ;

		double min_y_value = y_prev - height_offset ;
		double max_y_value = y_prev + height_offset ;

		double min_z_value = z_prev - depth_offset ;
		double max_z_value = z_prev + depth_offset ;

		double min_center_x_value = x_prev - object_offset/2.0 ;
		double max_center_x_value = x_prev + object_offset/2.0 ;

		double min_center_y_value = y_prev - object_offset/2.0;
		double max_center_y_value = y_prev + object_offset/2.0 ;


		double center_object_depth = 0.0;

		PointsT input_points = cloud->points;

		PointT input_point;

		int center_points = 0 ;
	
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
		
			    	if(x_value > min_x_value && x_value < max_x_value && y_value > min_y_value && y_value < max_y_value
				   && z_value > min_z_value && z_value < max_z_value)
				{
					
					if(x_value > min_center_x_value && x_value < max_center_x_value && y_value > min_center_y_value && y_value < max_center_y_value)
					{
						cropped_cloud_data.r_avg += r_value;
						cropped_cloud_data.g_avg += g_value;
						cropped_cloud_data.b_avg += b_value;
						center_points++;
					}

					cropped_cloud_data.cloud_ptr->points.push_back(input_point);
					cropped_cloud_data.number_of_points++;
				}
			}
	    	}

		if ( cropped_cloud_data.number_of_points > 5 )
		{

			cropped_cloud_data.r_avg /= center_points;
			cropped_cloud_data.g_avg /= center_points;
			cropped_cloud_data.b_avg /= center_points;

			Eigen::Vector4f cropped_cloud_centroid;

			pcl::compute3DCentroid<PointT> (*cropped_cloud_data.cloud_ptr, cropped_cloud_centroid);

			cropped_cloud_data.x_center = cropped_cloud_centroid[0];
			cropped_cloud_data.y_center = cropped_cloud_centroid[1];
			cropped_cloud_data.z_center = cropped_cloud_centroid[2];

			PointT point_min;
			PointT point_max;

			pcl::getMinMax3D(*cropped_cloud_data.cloud_ptr,point_min,point_max);

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

	bool track()
	{

		PointCloudDataT tracked_cloud_data = crop_cloud(input_point_cloud, prev_x, prev_y, prev_z);

		if(tracked_cloud_data.is_valid)
		{
			particle_tracker->setInputCloud(tracked_cloud_data.cloud_ptr);
			particle_tracker->compute();

		}
		else
		{
			reset();
			return false;
		}

		if(coherence->target_input_->size() == 0)
		{
			reset();
			return false;	
		}
	
		ParticleCloudPtrT particles = particle_tracker->getParticles ();

	        for (size_t i = 0; i < particles->points.size (); i++)
		{
			PointT point;
			point.x = particles->points[i].x;
			point.y = particles->points[i].y;
			point.z = particles->points[i].z;
			object_particle_cloud->points.push_back (point);
		}


	       sensor_msgs::PointCloud2::Ptr object_cloud (new sensor_msgs::PointCloud2); 
	       pcl::toROSMsg ( *object_particle_cloud, *object_cloud);
	       object_cloud->header.frame_id = "world";
	       object_cloud_pub.publish(object_cloud);

	       sensor_msgs::PointCloud2::Ptr tracked_cloud (new sensor_msgs::PointCloud2); 
	       pcl::toROSMsg ( *tracked_cloud_data.cloud_ptr, *tracked_cloud);
	       tracked_cloud->header.frame_id = "world";
	       tracked_cloud_pub.publish(tracked_cloud);
	 
	       ParticleT result = particle_tracker->getResult ();
	       Eigen::Affine3f transformation = particle_tracker->toEigenMatrix (result);

	       transformation.translation () += Eigen::Vector3f (0.0f, 0.0f, -0.005f);
	       PointCloudPtrT result_cloud (new PointCloudT ());
	       pcl::transformPointCloud<PointT> (*(particle_tracker->getReferenceCloud ()), *result_cloud, transformation);
	  
	       Eigen::Vector4f c;
	       pcl::compute3DCentroid<PointT> (*result_cloud, c);

	       prev_x = c[0];
	       prev_y = c[1];
	       prev_z = c[2];

	       object_centroid_msg.point.x = c[0];
	       object_centroid_msg.point.y = c[1];
	       object_centroid_msg.point.z = c[2];
	       object_centroid_msg.header.frame_id = "world";
	       object_centroid_msg.header.stamp = ros::Time::now();

	       object_centroid_pub.publish(object_centroid_msg);


	       //TODO Relocalization
	       double pixeldiff = abs(tracked_cloud_data.r_avg - target_cloud_data.r_avg) + abs(tracked_cloud_data.g_avg - target_cloud_data.g_avg)
				   + abs(tracked_cloud_data.b_avg - target_cloud_data.b_avg) ;


	}

	void reset()
	{
		particle_tracker->resetTracking();
		object_detected = false;
		prev_x=0.0;
		prev_y=0.0;
		prev_z=0.0;
		return;

	}

	void run()
	{
		if(!object_detected)
		{
			if(object_detection())
			{
				target_point_cloud = target_cloud_data.cloud_ptr;
				init_particle_filter();
			}
		}
		else
		{
			track();
		}


	}


};

int main( int argc, char** argv )
{    
    //Initialize the object tracker node
    ros::init(argc, argv, "object_tracker_node");

   // ros::NodeHandle nh;
    //object_tracker hand_tracker("bottom", NumberOfFeaturesToUse, FocalLength_X, FocalLength_Y, PrincipalPoint_X, PrincipalPoint_Y, Distortion);
    ros::spin();
    return 0;
}
