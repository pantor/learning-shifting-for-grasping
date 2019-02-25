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
#include <realsense/config.hpp>
#include <realsense/realsense.hpp>


class RealsenseNode {
  std::unique_ptr<Realsense> camera;

  bool getDepthImage(bin_picking::GetDepthImage::Request &req, bin_picking::GetDepthImage::Response &res) {
    const cv::Mat depth_image = camera->takeDepthImage();
    const sensor_msgs::ImagePtr depth_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO16, depth_image).toImageMsg();

    res.depth_image = *(depth_msg_ptr);
    return true;
  }

  bool getImages(bin_picking::GetImages::Request &req, bin_picking::GetImages::Response &res) {
    const std::pair<cv::Mat, cv::Mat> images = camera->takeImages();
    cv::imwrite("/tmp/testcolor.png", images.second);

    const sensor_msgs::ImagePtr depth_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO16, images.first).toImageMsg();
    const sensor_msgs::ImagePtr color_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::RGB8, images.second).toImageMsg();

    res.depth_image = *(depth_msg_ptr);
    res.color_image = *(color_msg_ptr);
    return true;
  }

public:
  RealsenseNode(RealsenseConfig config) {
    camera = std::make_unique<Realsense>(config);

    ros::NodeHandle node_handle;
    ros::ServiceServer service_depth = node_handle.advertiseService("realsense/depth_image", &RealsenseNode::getDepthImage, this);
    ros::ServiceServer service = node_handle.advertiseService("realsense/images", &RealsenseNode::getImages, this);
    ros::spin();
  }
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "realsense_node");

  window app(752, 480, "");

  RealsenseConfig config; // {ros::package::getPath("bin_picking") + "/config.yaml"};
  RealsenseNode node {config};

  return 0;
}