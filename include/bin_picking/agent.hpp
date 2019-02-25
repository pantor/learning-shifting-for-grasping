#pragma once

#include <chrono>
#include <iostream>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <frankr/geometry.hpp>
#include <frankr/gripper.hpp>
#include <frankr/robot.hpp>
#include <ensenso/ensenso.hpp>
#include <realsense/realsense.hpp>
#include <bin_picking/action.hpp>
#include <bin_picking/config.hpp>
#include <bin_picking/depth_image.hpp>
#include <bin_picking/inference.hpp>
#include <bin_picking/io.hpp>
#include <bin_picking/parameter.hpp>
#include <bin_picking/GetDepthImage.h>
#include <bin_picking/GetImages.h>



class Agent {
  DepthImage getEnsensoDepthImage() {
    cv::Mat image;
    if (config.use_ensenso_node) {
      bin_picking::GetDepthImage srv;
      ensenso_depth_client.call(srv);
      image = cv_bridge::toCvCopy(srv.response.depth_image, sensor_msgs::image_encodings::MONO16)->image;
    } else {
      image = ensenso->takeDepthImage();
    }
    return DepthImage(image, config.ensenso.pixel_size, config.ensenso.min_depth, config.ensenso.max_depth, "ed");
  }

  std::pair<DepthImage, DepthImage> getRealsenseImages() {
    std::pair<cv::Mat, cv::Mat> images;

    if (config.use_ensenso_node) {
      bin_picking::GetImages srv;
      realsense_depth_color_client.call(srv);
      images.first = cv_bridge::toCvCopy(srv.response.depth_image, sensor_msgs::image_encodings::MONO16)->image;
      images.second = cv_bridge::toCvCopy(srv.response.color_image, sensor_msgs::image_encodings::RGB8)->image;
    } else {
      images = realsense->takeImages();
    }
    return std::make_pair(
      DepthImage(images.first, config.realsense.pixel_size, config.realsense.min_depth, config.realsense.max_depth, "rd"),
      DepthImage(images.second, config.realsense.pixel_size, config.realsense.min_depth, config.realsense.max_depth, "rc")
    );
  }

protected:
  std::unique_ptr<Robot> robot;
  std::unique_ptr<Gripper> gripper;
  std::unique_ptr<Ensenso> ensenso;
  std::unique_ptr<Realsense> realsense;
  std::shared_ptr<Io> grasp_io;
  std::shared_ptr<Io> push_io;
  std::shared_ptr<Inference> grasp_inference;
  std::shared_ptr<Inference> push_inference;

  ros::ServiceClient ensenso_depth_client;
  ros::ServiceClient realsense_depth_client;
  ros::ServiceClient realsense_depth_color_client;
  ros::NodeHandle node_handle;
  ros::Subscriber sub_wrench;
  ros::Subscriber sub_states;
  ros::AsyncSpinner spinner {2};


  const Config config;

  Bin current_bin;

  int count_all_episodes {0};
  int count_episodes_in_current_bin {0};


  Eigen::Affine3d getBinAffine(const Bin bin) const {
    return config.bin_frames.at(bin);
  }

  Eigen::Affine3d getBinCameraAffine(const Bin bin) const {
    return Affine(0.0, 0.0, config.image_distance_from_pose) * getBinAffine(bin);
  }

  Eigen::Affine3d getBinReleaseAffine(const Bin bin) const {
    return Affine(0.0, 0.0, config.move_down_distance_for_release) * getBinAffine(bin);
  }

  Bin getNextBin(const Bin bin) const {
    switch (bin) {
      case Bin::Left: return Bin::Right;
      case Bin::Right: return Bin::Left;
      default: return Bin::Left;
    }
  }

  Eigen::Affine3d relativeSafePose(const Eigen::Affine3d& current_pose) const {
    Eigen::Affine3d affine = Eigen::Affine3d::Identity();
    affine.translation()(2) = std::max(0.0, 0.16 - current_pose.translation()(2)); // [m], min height
    return affine;
  }

  std::vector<DepthImage> takeImages() {
    switch (config.camera) {
      case Camera::Realsense: {
        auto realsense_images = getRealsenseImages();
        return { realsense_images.first, realsense_images.second };
      }
      case Camera::Both:{
        auto ensenso_image = getEnsensoDepthImage();
        auto realsense_images = getRealsenseImages();
        return { ensenso_image, realsense_images.first, realsense_images.second };
      }
      case Camera::Ensenso:
      default: {
        auto ensenso_image = getEnsensoDepthImage();
        return { ensenso_image };
      }
    }
  }

public:
  explicit Agent(const std::string& config_filename): config(config_filename)  {
    robot = std::make_unique<Robot>("panda_arm");
    robot->velocity_rel = config.general_dynamics_rel;
    robot->acceleration_rel = config.general_dynamics_rel;

    gripper = std::make_unique<Gripper>("172.16.0.2");
    gripper->gripper_speed = config.gripper_speed;

    if (!config.use_ensenso_node) {
      ensenso = std::make_unique<Ensenso>(config.ensenso);
    }

    grasp_io = std::make_shared<Io>(config.database_url, config.learning_url, config.grasp_database);
    push_io = std::make_shared<Io>(config.database_url, config.learning_url, config.push_database);

    if (config.use_cpp_inference) {
      std::vector<double> lower_random_pose(6);
      Eigen::VectorXd::Map(&lower_random_pose[0], 6) = Vector(config.lower_random_affine_before_action);

      std::vector<double> upper_random_pose(6);
      Eigen::VectorXd::Map(&upper_random_pose[0], 6) = Vector(config.upper_random_affine_before_action);

      grasp_inference = std::make_shared<Inference>(config.grasp_model, config.bin_data, lower_random_pose, upper_random_pose);
      grasp_inference->name = "grasp";
      
      if (config.push_objects) {
        push_inference = std::make_shared<Inference>(config.push_model, config.bin_data, lower_random_pose, upper_random_pose);
        push_inference->name = "push";
        push_inference->setSizeOriginalCropped({240, 240});
        push_inference->a_space = Inference::linspace(-3.0, 3.0, 26);
        push_inference->number_actions = 2;
      }
    }

    sub_wrench = node_handle.subscribe("franka_state_controller/F_ext", 10, &Robot::wrenchCallback, robot.get());
    sub_states = node_handle.subscribe("franka_state_controller/franka_states", 10, &Robot::stateCallback, robot.get());

    ensenso_depth_client = node_handle.serviceClient<bin_picking::GetDepthImage>("ensenso/depth_image");
    realsense_depth_client = node_handle.serviceClient<bin_picking::GetDepthImage>("realsense/depth_image");
    realsense_depth_color_client = node_handle.serviceClient<bin_picking::GetImages>("realsense/images");

    spinner.start();

    current_bin = config.start_bin;
  }

  void run() {
    for (auto epoch: config.epochs) {
      for (int current_episode_in_epoch = 0; current_episode_in_epoch < epoch.number_cycles; current_episode_in_epoch++) {
        if (!ros::ok()) {
          return;
        }

        std::cout << "---" << std::endl;

        runEpisode(epoch, current_episode_in_epoch);
      }
    }
  }

  virtual void reset() = 0;

  virtual void runEpisode(const Epoch epoch, int current_episode_in_epoch) = 0;
};
