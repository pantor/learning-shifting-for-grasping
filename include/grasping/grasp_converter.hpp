#pragma once

#include <bin_picking/action.hpp>
#include <grasping/grasp.hpp>


struct GraspConverter {
  std::vector<double> gripper_classes;

  BinData bin;
  bool pose_should_be_inside_bin;

  bool checkSafety(Grasp& grasp) {
    bool inside_bin = bin.isPoseInside(grasp.pose) || !pose_should_be_inside_bin; // so insideBin only if pose should be inside bin

    if (grasp.found == 1 && inside_bin) {
      grasp.found = 1;
      return true;
    } else if (!inside_bin) {
      std::cout << "Pose not inside bin." << std::endl;
      grasp.found = -1;
    } else {
      grasp.found = 0;
    }
    return false;
  }

  double calculatePoseZ(const DepthImage& image, const RobotPose& pose) {
    DepthImage area_image = dimg::getAreaOfInterest(image, pose, cv::Size{752, 480}, bin.background_color, cv::INTER_NEAREST);

    // Get distance at gripper for possible collisions
    const double gripper_one_side_size = 0.5 * image.pixel_size * (pose.d + 0.002); // [px]
    cv::Size side_gripper_image_size (image.pixel_size * 0.024, image.pixel_size * 0.024);

    DepthImage area_center = dimg::crop(area_image, cv::Size(image.pixel_size * 0.01, image.pixel_size * 0.03));
    DepthImage area_center_sm = dimg::crop(area_image, cv::Size(image.pixel_size * 0.003, image.pixel_size * 0.003));
    DepthImage area_left = dimg::crop(area_image, side_gripper_image_size, cv::Point2d(-gripper_one_side_size, 0));
    DepthImage area_right = dimg::crop(area_image, side_gripper_image_size, cv::Point2d(gripper_one_side_size, 0));
    cv::Mat area_left_mask = (area_left.image > 255); // 16bit
    cv::Mat area_right_mask = (area_right.image > 255); // 16bit

    double min_area_center_sm, max_area_center_sm;
    cv::minMaxLoc(area_center_sm.image, &min_area_center_sm, &max_area_center_sm);
    double z_raw = area_center_sm.depthFromValue(max_area_center_sm);

    if (std::isnan(z_raw)) {
      double min_area_center, max_area_center;
      cv::minMaxLoc(area_center.image, &min_area_center, &max_area_center);
      z_raw = area_center.depthFromValue(max_area_center);
    }

    double min_area_left, max_area_left;
    cv::minMaxLoc(area_left.image, &min_area_left, &max_area_left, 0, 0, area_left_mask);
    double z_raw_left = area_left.depthFromValue(min_area_left);

    double min_area_right, max_area_right;
    cv::minMaxLoc(area_right.image, &min_area_right, &max_area_right, 0, 0, area_right_mask);
    double z_raw_right = area_right.depthFromValue(min_area_right);

    z_raw += 0.025; // [m] 0.021
    double z_raw_collision = std::max(z_raw_left, z_raw_right) - 0.01; // [m]
    const double z = std::min(z_raw, z_raw_collision); // Get the maximum [m] for impedance mode
    return -z_raw;
  }

  Grasp convert(const InferenceResult& inf, const DepthImage& dimg) {
    Grasp result;
    result.pose.x = inf.x;
    result.pose.y = inf.y;
    result.pose.a = inf.a;
    result.pose.d = gripper_classes[inf.index];
    result.pose.z = calculatePoseZ(dimg, result.pose); // [m]
    result.prob = inf.prob;
    result.probstd = inf.probstd;
    result.method = inf.method;
    result.found = true;
    checkSafety(result);
    return result;
  }

  InferenceResult convertBack(const Grasp& grasp) {
    InferenceResult inf;
    inf.x = grasp.pose.x;
    inf.y = grasp.pose.y;
    inf.a = grasp.pose.a;

    auto const it = std::min_element(gripper_classes.begin(), gripper_classes.end(), [=](double x, double y) {
      return std::abs(x - grasp.pose.d) < std::abs(y - grasp.pose.d);
    });
    inf.index = std::distance(gripper_classes.begin(), it);
    inf.prob = grasp.prob;
    inf.probstd = grasp.probstd;
    inf.method = grasp.method;
    return inf;
  }
};
