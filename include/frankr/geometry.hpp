#pragma once

#define BIN_PICKING_GEOMETRY

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

#include <geometry_msgs/PoseStamped.h>

#include <franka_msgs/FrankaState.h>

#include <Eigen/Geometry>
#include <unsupported/Eigen/EulerAngles>



using Euler = Eigen::EulerAngles<double, Eigen::EulerSystemZYX>;


inline const Eigen::Affine3d Affine(double x, double y, double z, double a = 0.0, double b = 0.0, double c = 0.0) {
  Eigen::Affine3d affine;
  affine = Eigen::Translation<double, 3>(x, y, z) * Euler(a, b, c).toRotationMatrix();
  return affine;
}

inline const Eigen::Affine3d Affine(const geometry_msgs::Pose& pose) {
  auto position = pose.position;
  auto orientation = pose.orientation;

  Eigen::Translation<double, 3> t {position.x, position.y, position.z};
  Eigen::Quaternion<double> q {orientation.w, orientation.x, orientation.y, orientation.z};

  Eigen::Affine3d result;
  result = t * q;
  return result;
}

inline const geometry_msgs::Pose Pose(const Eigen::Affine3d& affine) {
  Eigen::Quaternion<double> q;
  q = affine.rotation();

  geometry_msgs::Pose result;
  result.position.x = affine.translation()(0);
  result.position.y = affine.translation()(1);
  result.position.z = affine.translation()(2);
  result.orientation.x = q.x();
  result.orientation.y = q.y();
  result.orientation.z = q.z();
  result.orientation.w = q.w();
  return result;
}

inline const Eigen::Matrix<double, 6, 1> Vector(const Eigen::Affine3d& affine) {
  Eigen::Matrix<double, 6, 1> result;
  result << affine.translation(), static_cast<Euler>(affine.rotation()).angles();
  return result;
}

inline const Eigen::Affine3d getRandomAffine(const Eigen::Affine3d& max_random_affine) {
  std::random_device r;
  std::default_random_engine engine(r());

  Eigen::Matrix<double, 6, 1> max = Vector(max_random_affine);
  Eigen::Matrix<double, 6, 1> random;

  for (int i = 0; i < 6; i++) {
    std::uniform_real_distribution<double> distribution(-max(i), max(i));
    random(i) = distribution(engine);
  }

  return Affine(random(0), random(1), random(2), random(3), random(4), random(5));
}
