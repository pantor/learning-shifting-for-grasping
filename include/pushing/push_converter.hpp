#pragma once

#include <bin_picking/action.hpp>
#include <pushing/push.hpp>


struct PushConverter {
  BinData bin;
  bool pose_should_be_inside_bin;

  bool checkSafety(Push& push) {
    RobotPose end_pose = push.pose;
    end_pose.x += std::cos(end_pose.a) * 0.024; // [m]
    end_pose.y += std::sin(end_pose.a) * 0.024; // [m]

    bool start_inside_bin = bin.isPoseInside(push.pose);
    bool end_inside_bin = bin.isPoseInside(end_pose);

    bool inside_bin = (start_inside_bin && end_inside_bin) || !pose_should_be_inside_bin; // so insideBin only if pose should be inside bin

    if (push.found == 1 && inside_bin) {
      push.found = 1;
      return true;
    } else if (!inside_bin) {
      std::cout << "Pose not inside bin." << std::endl;
      push.found = -1;
    } else {
      push.found = 0;
    }
    return false;
  }

  double calculatePoseZ(const DepthImage& image, const RobotPose& pose) {
    DepthImage area_image = dimg::getAreaOfInterest(image, pose, cv::Size{752, 480}, bin.background_color);

    DepthImage area_center = dimg::crop(area_image, cv::Size(image.pixel_size * 0.012, image.pixel_size * 0.012));
    DepthImage area_center_sm = dimg::crop(area_image, cv::Size(image.pixel_size * 0.005, image.pixel_size * 0.005));

    double min_area_center_sm, max_area_center_sm;
    cv::minMaxLoc(area_center_sm.image, &min_area_center_sm, &max_area_center_sm);
    double z_raw = area_center_sm.depthFromValue(max_area_center_sm);

    if (std::isnan(z_raw)) {
      double min_area_center, max_area_center;
      cv::minMaxLoc(area_center.image, &min_area_center, &max_area_center);
      z_raw = area_center.depthFromValue(max_area_center);
    }

    z_raw += 0.007; // [m]
    return -z_raw; // [m]
  }

  Push convert(const InferenceResult& inf, const DepthImage& image) {
    Push result;
    result.pose.x = inf.x;
    result.pose.y = inf.y;
    result.pose.a = inf.a;
    result.pose.d = 0.0;
    result.direction = Push::Direction::Left; // (inf.index == 0) ? Push::Direction::Up : Push::Direction::Left;
    result.pose.z = calculatePoseZ(image, result.pose); // [m]
    result.prob = inf.prob;
    result.probstd = inf.probstd;
    result.method = inf.method;
    result.found = true;
    checkSafety(result);
    return result;
  }

  InferenceResult convertBack(const Push& push) {
    InferenceResult inf;
    inf.x = push.pose.x;
    inf.y = push.pose.y;
    inf.a = push.pose.a;
    inf.index = (push.direction == Push::Direction::Up) ? 0 : 1;
    inf.prob = push.prob;
    inf.probstd = push.probstd;
    inf.method = push.method;
    return inf;
  }
};
