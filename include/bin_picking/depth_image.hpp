#pragma once

#include <cmath>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <bin_picking/bin.hpp>
#include <bin_picking/image_utils.hpp>


struct DepthImage {
  double pixel_size; // [px/m]
  double min_depth; // [m]
  double max_depth; // [m]

  int max_value{255 * 255};

  cv::Mat image;
  std::string suffix;

  explicit DepthImage(std::string image_path, double pixel_size, double min_depth, double max_depth): image(cv::imread(image_path, 0)), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), suffix("") { }
  explicit DepthImage(cv::Mat image, double pixel_size, double min_depth, double max_depth): image(image), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), suffix("") { }
  explicit DepthImage(cv::Mat image, double pixel_size, double min_depth, double max_depth, std::string suffix): image(image), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), suffix(suffix) { }

  DepthImage(const DepthImage& depth_image) {
    pixel_size = depth_image.pixel_size;
    min_depth = depth_image.min_depth;
    max_depth = depth_image.max_depth;
    max_value = depth_image.max_value;

    image = depth_image.image.clone();
    suffix = depth_image.suffix;
  }

  double depthFromValue(double value) const {
    return max_depth + value / max_value * (min_depth - max_depth);
  }

  double valueFromDepth(double depth) const {
    return (depth - max_depth) / (min_depth - max_depth) * max_value;
  }

  double positionFromIndex(int idx, int length) const {
    return ((idx + 0.5) - double(length) / 2) / pixel_size;
  }

  int indexFromPosition(double position, int length) const {
    return ((position * pixel_size) + (length / 2) - 0.5);
  }

  cv::Point2d transformToImageSpace(const Eigen::Affine3d& point, const Eigen::Affine3d& pose, const cv::Size& size) const {
    Eigen::Affine3d affine = pose * point;
    return cv::Point2d(size.width / 2, size.height / 2) + pixel_size * cv::Point2d(-affine(1, 3), -affine(0, 3));
  }
};


namespace dimg {

inline DepthImage crop(const DepthImage& depth_image, cv::Size size, cv::Point2d vec) {
  DepthImage result = depth_image;
  int x = (depth_image.image.cols - size.width) / 2 + vec.y;
  int y = (depth_image.image.rows - size.height) / 2 + vec.x;
  result.image = depth_image.image(cv::Rect(x, y, size.width, size.height));
  return result;
}

inline DepthImage crop(const DepthImage& depth_image, cv::Size size) {
  return crop(depth_image, size, cv::Point2d(0.0, 0.0));
}

inline DepthImage getAreaOfInterest(const DepthImage& depth_image, const RobotPose& pose, const cv::Size& size, cv::Scalar background_color, cv::InterpolationFlags interpolation) {
  cv::Point center_image {depth_image.image.cols / 2, depth_image.image.rows / 2};
  auto trans = getTransformation(depth_image.pixel_size * pose.y, depth_image.pixel_size * pose.x, -pose.a, center_image);

  cv::Mat result;
  cv::warpAffine(depth_image.image, result, trans, size, interpolation, cv::BORDER_CONSTANT, background_color);
  return DepthImage(result, depth_image.pixel_size, depth_image.min_depth, depth_image.max_depth);
}

inline DepthImage getAreaOfInterest(const DepthImage& depth_image, const RobotPose& pose, const cv::Size& size, cv::Scalar background_color) {
  return getAreaOfInterest(depth_image, pose, size, background_color, cv::INTER_LINEAR);
}

inline void drawLine(DepthImage& depth_image, const RobotPose& pose, cv::Point2d pt1, cv::Point2d pt2, cv::Scalar color, int thickness) {
  cv::Point2d center (depth_image.image.size[1] / 2 - depth_image.pixel_size * pose.y, depth_image.image.size[0] / 2 - depth_image.pixel_size * pose.x);
  cv::Point2d pt1_rot = rotate(pt1, -pose.a);
  cv::Point2d pt2_rot = rotate(pt2, -pose.a);
  cv::line(depth_image.image, center + pt1_rot, center + pt2_rot, color, thickness);
}

inline void drawLine(DepthImage& depth_image, const RobotPose& pose, cv::Point2d pt1, cv::Point2d pt2, cv::Scalar color) {
  drawLine(depth_image, pose, pt1, pt2, color, 1);
}

inline void drawRotatedRect(DepthImage& depth_image, const RobotPose& pose, cv::Size size, cv::Scalar color) {
  std::array<cv::Point2d, 4> vecs = {
    cv::Point2d(-size.height / 2, size.width / 2),
    cv::Point2d(size.height / 2, size.width / 2),
    cv::Point2d(size.height / 2, -size.width / 2),
    cv::Point2d(-size.height / 2, -size.width / 2)
  };
  for (int i = 0; i < vecs.size(); i++) {
    drawLine(depth_image, pose, vecs[i % vecs.size()], vecs[(i + 1) % vecs.size()], color, 2);
  }
}

inline DepthImage drawPose(const DepthImage &depth_image, const RobotPose& pose) {
  cv::Mat color (depth_image.image.size(), CV_16UC3);
  cv::cvtColor(depth_image.image, color, cv::COLOR_GRAY2BGR);

  auto result = DepthImage(color, depth_image.pixel_size, depth_image.min_depth, depth_image.max_depth);

  double gripper_px = result.pixel_size * (pose.d + 0.001);
  cv::Scalar color_rect (255 * 255, 0, 0); // Blue
  cv::Scalar color_lines (0, 0, 255 * 255); // Red
  cv::Scalar color_direction (0, 255 * 255, 0); // Green

  drawRotatedRect(result, pose, cv::Size(200, 200), color_rect); // Cropped input of CNN

  drawLine(result, pose, cv::Point2d(gripper_px / 2, result.pixel_size * 0.012), cv::Point2d(gripper_px / 2, -result.pixel_size * 0.012), color_lines);
  drawLine(result, pose, cv::Point2d(-gripper_px / 2, result.pixel_size * 0.012), cv::Point2d(-gripper_px / 2, -result.pixel_size * 0.012), color_lines);
  drawLine(result, pose, cv::Point2d(gripper_px / 2, 0), cv::Point2d(-gripper_px / 2, 0), color_lines);
  drawLine(result, pose, cv::Point2d(0, result.pixel_size * 0.006), cv::Point2d(0, -result.pixel_size * 0.006), color_lines);
  return result;
}

inline DepthImage drawAroundBin(const DepthImage& depth_image, BinData bin) {
  DepthImage result = depth_image;

  cv::Point center (result.image.cols / 2, result.image.rows / 2);
  cv::Point bin_top_left_px = result.pixel_size * bin.top_left;
  cv::Point bin_bottom_right_px = result.pixel_size * bin.bottom_right;

  cv::rectangle(result.image, cv::Point(0, 0), cv::Point(center.x + bin_top_left_px.x, result.image.rows), bin.background_color, -1);
  cv::rectangle(result.image, cv::Point(result.image.cols, 0), cv::Point(center.x + bin_bottom_right_px.x, result.image.rows), bin.background_color, -1);
  cv::rectangle(result.image, cv::Point(0, 0), cv::Point(result.image.cols, center.y + bin_top_left_px.y), bin.background_color, -1);
  cv::rectangle(result.image, cv::Point(0, center.y + bin_bottom_right_px.y), cv::Point(result.image.cols, result.image.rows), bin.background_color, -1);
  return result;
}

inline DepthImage drawAroundBin(const DepthImage& depth_image, BinData bin, const Eigen::Affine3d& pose, const Eigen::Affine3d& bin_frame) {
  DepthImage result = depth_image;

  Eigen::Affine3d bin_top_left_affine, bin_top_right_affine, bin_bottom_left_affine, bin_bottom_right_affine;
  bin_top_left_affine = Eigen::Translation<double, 3>(bin.top_left.x, bin.top_left.y, -0.28); // [m]
  bin_top_right_affine = Eigen::Translation<double, 3>(bin.bottom_right.x, bin.top_left.y, -0.28); // [m]
  bin_bottom_left_affine = Eigen::Translation<double, 3>(bin.top_left.x, bin.bottom_right.y - 0.005, -0.28); // [m]
  bin_bottom_right_affine = Eigen::Translation<double, 3>(bin.bottom_right.x, bin.bottom_right.y - 0.005, -0.28); // [m]

  Eigen::Affine3d border_top_left_affine, border_top_right_affine, border_bottom_left_affine, border_bottom_right_affine;
  border_top_left_affine = Eigen::Translation<double, 3>(-10.0, -10.0, -0.28); // [m]
  border_top_right_affine = Eigen::Translation<double, 3>(10.0, -10.0, -0.28); // [m]
  border_bottom_left_affine = Eigen::Translation<double, 3>(-10.0, 10.0, -0.28); // [m]
  border_bottom_right_affine = Eigen::Translation<double, 3>(10.0, 10.0, -0.28); // [m]

  Eigen::Affine3d diff = pose.inverse() * bin_frame;
  std::vector<std::vector<cv::Point>> pts = {{
    result.transformToImageSpace(border_top_right_affine, diff, result.image.size()),
    result.transformToImageSpace(border_top_left_affine, diff, result.image.size()),
    result.transformToImageSpace(border_bottom_left_affine, diff, result.image.size()),
    result.transformToImageSpace(border_bottom_right_affine, diff, result.image.size()),
  }, {
    result.transformToImageSpace(bin_top_right_affine, diff, result.image.size()),
    result.transformToImageSpace(bin_top_left_affine, diff, result.image.size()),
    result.transformToImageSpace(bin_bottom_left_affine, diff, result.image.size()),
    result.transformToImageSpace(bin_bottom_right_affine, diff, result.image.size()),
  }};

  std::vector<double> bin_depths = {
    -(diff * bin_top_right_affine).translation()(2),
    -(diff * bin_top_left_affine).translation()(2),
    -(diff * bin_bottom_left_affine).translation()(2),
    -(diff * bin_bottom_right_affine).translation()(2),
  };
  double min_depth = *std::min_element(bin_depths.begin(), bin_depths.end());

  cv::fillPoly(result.image, pts, cv::Scalar(result.valueFromDepth(min_depth)));
  return result;
}

}
