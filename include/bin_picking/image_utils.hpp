#pragma once

#include <opencv2/opencv.hpp>

namespace dimg {

inline cv::Point2d rotate(const cv::Point2d& point, double a) {
  return cv::Point2d(point.x * std::cos(a) - point.y * std::sin(a), point.x * std::sin(a) + point.y * std::cos(a));
}

inline cv::Mat crop(const cv::Mat& image, cv::Size size, cv::Point2d vec) {
  int x = (image.cols - size.width) / 2 + vec.y;
  int y = (image.rows - size.height) / 2 + vec.x;
  return image(cv::Rect(x, y, size.width, size.height));
}

inline cv::Mat crop(const cv::Mat& image, cv::Size size) {
  return crop(image, size, cv::Point2d(0.0, 0.0));
}

inline cv::Mat getTransformation(double x, double y, double a, cv::Point center) {
  auto rot_mat = cv::getRotationMatrix2D(cv::Point(center.x - x, center.y - y), a * 180.0 / M_PI, 1.0); // [deg]
  rot_mat.at<double>(0, 2) += x;
  rot_mat.at<double>(1, 2) += y;
  return rot_mat;
}

}
