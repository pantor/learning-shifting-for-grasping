#include <iostream>
#include <memory>

#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <bin_picking/config.hpp>
#include <bin_picking/GetDepthImage.h>
#include <bin_picking/GetImages.h>
#include <ensenso/config.hpp>
#include <ensenso/ensenso.hpp>


class EnsensoNode {
  std::unique_ptr<Ensenso> camera;

  bool getDepthImage(bin_picking::GetDepthImage::Request &req, bin_picking::GetDepthImage::Response &res) {
    const cv::Mat depth_image = camera->takeDepthImage();
    const sensor_msgs::ImagePtr depth_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO16, depth_image).toImageMsg();

    res.depth_image = *(depth_msg_ptr);
    return true;
  }

  bool getImages(bin_picking::GetImages::Request &req, bin_picking::GetImages::Response &res) {
    const std::pair<cv::Mat, cv::Mat> images = camera->takeImages();

    const sensor_msgs::ImagePtr color_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO16, images.first).toImageMsg();
    const sensor_msgs::ImagePtr depth_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO16, images.second).toImageMsg();

    res.color_image = *(color_msg_ptr);
    res.depth_image = *(depth_msg_ptr);
    return true;
  }

public:
  EnsensoNode(EnsensoConfig config) {
    camera = std::make_unique<Ensenso>(config);

    ros::NodeHandle node_handle;
    ros::ServiceServer service_depth = node_handle.advertiseService("ensenso/depth_image", &EnsensoNode::getDepthImage, this);
    ros::ServiceServer service = node_handle.advertiseService("ensenso/images", &EnsensoNode::getImages, this);
    ros::spin();
  }
};


int main(int argc, char *argv[]) {
  ros::init(argc, argv, "ensenso_node");

  Config config {ros::package::getPath("bin_picking") + "/config.yaml"};

  EnsensoNode node {config.ensenso};

  return 0;
}
