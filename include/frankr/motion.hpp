#pragma once

#include <frankr/geometry.hpp>


struct MotionData {
  double velocity_rel {1.0};
  double acceleration_rel {1.0};

  bool check_z_force_condition {false};
  double max_z_force;

  bool check_xy_force_condition {false};
  double max_xy_force;

  bool did_break {false};
  std::function<void()> break_callback;

  MotionData& withDynamics(double dynamics_rel) {
    this->velocity_rel = dynamics_rel;
    this->acceleration_rel = dynamics_rel;
    return *this;
  }

  MotionData& withZForceCondition(double max_z_force) {
    this->check_z_force_condition = true;
    this->max_z_force = max_z_force;
    return *this;
  }

  MotionData& withXYForceCondition(double max_xy_force) {
    this->check_xy_force_condition = true;
    this->max_xy_force = max_xy_force;
    return *this;
  }

  MotionData& withBreakCallback(std::function<void()> break_callback) {
    this->break_callback = break_callback;
    return *this;
  }
};
