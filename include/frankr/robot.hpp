#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

#include <geometry_msgs/WrenchStamped.h>
#include <franka_msgs/FrankaState.h>
#include <franka_control/ErrorRecoveryAction.h>
#include <franka_control/ErrorRecoveryActionGoal.h>
#include <franka_control/ErrorRecoveryGoal.h>

#include <frankr/geometry.hpp>
#include <frankr/motion.hpp>
#include <frankr/waypoint.hpp>


class Robot: moveit::planning_interface::MoveGroupInterface {

  moveit::planning_interface::MoveGroupInterface::Plan my_plan;

  // Break conditions
  bool is_moving {false};
  std::shared_ptr<MotionData> current_motion_data;

  // Interrupt sleep for position hold motion
  std::mutex mutex;
  std::condition_variable wait_condition_variable;
  bool flag {false};

  template<typename T>
  auto restartMoveItIfCommandFails(T lambda, long int timeout) {
    auto future = std::async(std::launch::async, lambda);
    auto status = future.wait_for(std::chrono::seconds(timeout));
    if (status == std::future_status::timeout) {
      restartMoveIt();
      return lambda();
    }
    return future.get();
  }


public:
  double velocity_rel {1.0};
  double acceleration_rel {1.0};

  actionlib::SimpleActionClient<franka_control::ErrorRecoveryAction> ac{"franka_control/error_recovery", true};

  Robot(std::string name);

  void stateCallback(const franka_msgs::FrankaState& msg);
  void wrenchCallback(const geometry_msgs::WrenchStamped& msg);

  Eigen::Affine3d currentPose(const Eigen::Affine3d& frame);

  bool isMoving();

  void restartMoveIt();
  bool recoverFromErrors();

  bool moveJoints(const std::array<double, 7>& joint_values, const MotionData& data = MotionData());
  bool moveJoints(const std::array<double, 7>& joint_values, MotionData& data);

  bool movePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data = MotionData());
  bool movePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data);

  bool moveRelativePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data = MotionData());
  bool moveRelativePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data);

  bool moveWaypointsPtp(const Eigen::Affine3d& frame, const std::vector<Waypoint>& waypoints, MotionData& data);

  bool moveCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data = MotionData());
  bool moveCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data);

  bool moveRelativeCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data = MotionData());
  bool moveRelativeCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data);

  bool moveWaypointsCartesian(const Eigen::Affine3d& frame, const std::vector<Waypoint>& waypoints, MotionData& data);

  bool positionHold(double duration, const MotionData& data = MotionData()); // [s]
  bool positionHold(double duration, MotionData& data); // [s]
};
