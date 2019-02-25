#pragma once

#include <frankr/geometry.hpp>


struct Waypoint {
  enum class ReferenceType {
    ABSOLUTE,
    RELATIVE
  };

  Eigen::Affine3d target_affine;

  ReferenceType reference_type {ReferenceType::ABSOLUTE};

  Waypoint(Eigen::Affine3d target_affine): target_affine(target_affine) { }
  Waypoint(Eigen::Affine3d target_affine, ReferenceType reference_type): target_affine(target_affine), reference_type(reference_type) { }
};
