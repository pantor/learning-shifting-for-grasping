#pragma once

#include <opencv2/opencv.hpp>

#include <bin_picking/action.hpp>
#include <bin_picking/image_utils.hpp>


struct BinData {
  cv::Point2f top_left; // [m]
  cv::Point2f bottom_right; // [m]
  cv::Scalar background_color; // {255 * 159}


  BinData() { }
  BinData(cv::Point2f top_left, cv::Point2f bottom_right, cv::Scalar background_color): top_left(top_left), bottom_right(bottom_right), background_color(background_color) { }

  bool isPoseInside(const RobotPose& pose) {
    const double gripper_one_side_size = 0.5 * (pose.d + 0.002); // [m]
    cv::Point2d gripper_jaw1 = dimg::rotate(cv::Point2d(gripper_one_side_size, 0), -pose.a); // [m]
    cv::Point2d gripper_jaw2 = dimg::rotate(cv::Point2d(-gripper_one_side_size, 0), -pose.a); // [m]

    const double pose_x = -pose.y; // [m]
    const double pose_y = -pose.x; // [m]

    cv::Point2d gripper_b1 (pose_x + gripper_jaw1.x, pose_y + gripper_jaw1.y); // [m]
    cv::Point2d gripper_b2 (pose_x + gripper_jaw2.x, pose_y + gripper_jaw2.y); // [m]

    const bool jaw1_inside_bin = (top_left.x < gripper_b1.x && gripper_b1.x < bottom_right.x && top_left.y < gripper_b1.y && gripper_b1.y < bottom_right.y);
    const bool jaw2_inside_bin = (top_left.x < gripper_b2.x && gripper_b2.x < bottom_right.x && top_left.y < gripper_b2.y && gripper_b2.y < bottom_right.y);

    // TODO: angles with b and c != 0
    bool start_point_inside_bin = true;

    return (jaw1_inside_bin && jaw2_inside_bin && start_point_inside_bin);
  }
};
