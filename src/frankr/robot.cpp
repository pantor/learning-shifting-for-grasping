#include <frankr/robot.hpp>


Robot::Robot(std::string name): moveit::planning_interface::MoveGroupInterface(name) { }

void Robot::stateCallback(const franka_msgs::FrankaState& msg) {
  if (msg.current_errors.cartesian_reflex) {
    std::cout << "Cartesian reflex error!" << std::endl;
  }
}

void Robot::wrenchCallback(const geometry_msgs::WrenchStamped& msg) {
  if (is_moving && !current_motion_data->did_break) {
    if (current_motion_data->check_z_force_condition && (std::pow(msg.wrench.force.z, 2) > std::pow(current_motion_data->max_z_force, 2))) {
      std::cout << "Exceeded z force." << std::endl;
      this->stop();
      current_motion_data->did_break = true;
      this->is_moving = false;

      this->flag = true;
      wait_condition_variable.notify_all();
    }

    if (current_motion_data->check_xy_force_condition && (std::pow(msg.wrench.force.x, 2) + std::pow(msg.wrench.force.y, 2) > std::pow(current_motion_data->max_xy_force, 2))) {
      std::cout << "Exceeded xy force." << std::endl;
      this->stop();
      current_motion_data->did_break = true;
      this->is_moving = false;

      this->flag = true;
      wait_condition_variable.notify_all();
    }
  }
}

Eigen::Affine3d Robot::currentPose(const Eigen::Affine3d& frame) {
  return restartMoveItIfCommandFails([&]() { return Affine(this->getCurrentPose().pose) * frame.inverse(); }, 5); // [s]
}

bool Robot::isMoving() {
  return is_moving;
}

void Robot::restartMoveIt() {
  // Set /move_group node to respawn=true
  std::cout << "Restart MoveIt!" << std::endl;
  // system("rosnode kill /move_group"); // Soft kill by ROS
  system("kill $(ps ax | grep [/]move_group | awk '{print $1}')");
  std::this_thread::sleep_for(std::chrono::seconds(4));
  std::cout << "Restarted MoveIt" << std::endl;
}

bool Robot::recoverFromErrors() {
  ac.waitForServer();

  franka_control::ErrorRecoveryGoal goal;
  ac.sendGoal(goal);

  return ac.waitForResult(ros::Duration(5.0));
}


bool Robot::moveJoints(const std::array<double, 7>& joint_values, const MotionData& data) {
  MotionData data_non_const = data;
  moveJoints(joint_values, data_non_const);
}

bool Robot::moveJoints(const std::array<double, 7>& joint_values, MotionData& data) {
  current_motion_data = std::make_shared<MotionData>(data);
  this->setMaxVelocityScalingFactor(velocity_rel * data.velocity_rel);
  this->setMaxAccelerationScalingFactor(acceleration_rel * data.acceleration_rel);

  std::vector<double> joint_values_vector {joint_values.begin(), joint_values.end()};
  this->setJointValueTarget(joint_values_vector);

  bool execution_success = false;
  if (this->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
    this->is_moving = true;
    auto execution = this->execute(my_plan);
    this->is_moving = false;
    execution_success = (execution == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  } else {
    ROS_FATAL_STREAM("Error in planning motion");
    return false;
  }

  if (current_motion_data->break_callback && current_motion_data->did_break) {
    current_motion_data->break_callback();
  }

  data = *current_motion_data;

  current_motion_data = std::make_shared<MotionData>();
  return execution_success;
}

bool Robot::movePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data) {
  MotionData data_non_const = data;
  return movePtp(frame, affine, data_non_const);
}

bool Robot::movePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data) {
  return moveWaypointsPtp(frame, { Waypoint(affine) }, data);
}

bool Robot::moveRelativePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data) {
  MotionData data_non_const = data;
  return moveRelativePtp(frame, affine, data_non_const);
}

bool Robot::moveRelativePtp(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data) {
  return moveWaypointsPtp(frame, { Waypoint(affine, Waypoint::ReferenceType::RELATIVE) }, data);
}

bool Robot::moveWaypointsPtp(const Eigen::Affine3d& frame, const std::vector<Waypoint>& waypoints, MotionData& data) {
  EigenSTL::vector_Affine3d affines {};
  for (auto waypoint: waypoints) {
    if (waypoint.reference_type == Waypoint::ReferenceType::ABSOLUTE) {
      affines.push_back(waypoint.target_affine * frame);
    } else {
      Eigen::Affine3d base_affine = affines.empty() ? Affine(this->getCurrentPose().pose) * frame : affines.back() * frame;

      base_affine.translate(waypoint.target_affine.translation());
      base_affine = base_affine * frame.inverse();
      base_affine.rotate(waypoint.target_affine.rotation());

      affines.push_back(base_affine);
    }
  }

  current_motion_data = std::make_shared<MotionData>(data);
  this->setMaxVelocityScalingFactor(velocity_rel * data.velocity_rel);
  this->setMaxAccelerationScalingFactor(acceleration_rel * data.acceleration_rel);

  bool execution_success = false;
  for (auto affine: affines) {
    this->setPoseTarget(affine);

    if (this->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
      this->stop();
      this->is_moving = true;
      auto execution = this->execute(my_plan);
      this->is_moving = false;
      if (execution == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
        execution_success = true;
      }
    } else {
      ROS_FATAL_STREAM("Error in planning motion");
      return false;
    }
  }

  if (current_motion_data->break_callback && current_motion_data->did_break) {
    current_motion_data->break_callback();
  }

  data = *current_motion_data;

  current_motion_data = std::make_shared<MotionData>();
  return execution_success;
}

bool Robot::moveCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data) {
  MotionData data_non_const = data;
  return moveCartesian(frame, affine, data_non_const);
}

bool Robot::moveCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data) {
  return moveWaypointsCartesian(frame, { Waypoint(affine) }, data);
}

bool Robot::moveRelativeCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, const MotionData& data) {
  MotionData data_non_const = data;
  return moveRelativeCartesian(frame, affine, data_non_const);
}

bool Robot::moveRelativeCartesian(const Eigen::Affine3d& frame, const Eigen::Affine3d& affine, MotionData& data) {
  return moveWaypointsCartesian(frame, { Waypoint(affine, Waypoint::ReferenceType::RELATIVE) }, data);
}

bool Robot::moveWaypointsCartesian(const Eigen::Affine3d& frame, const std::vector<Waypoint>& waypoints, MotionData& data) {
  geometry_msgs::Pose start_pose = this->getCurrentPose().pose;

  std::vector<geometry_msgs::Pose> poses;
  poses.push_back(start_pose);

  for (auto waypoint: waypoints) {
    if (waypoint.reference_type == Waypoint::ReferenceType::ABSOLUTE) {
      poses.push_back(Pose(waypoint.target_affine * frame));
    } else {
      Eigen::Affine3d base_affine = Affine(poses.back()) * frame;

      base_affine.translate(waypoint.target_affine.translation());
      base_affine = base_affine * frame.inverse();
      base_affine.rotate(waypoint.target_affine.rotation());

      poses.push_back(Pose(base_affine));
    }
  }

  current_motion_data = std::make_shared<MotionData>(data);

  moveit_msgs::RobotTrajectory trajectory;
  bool execution_success = false;

  double fraction = restartMoveItIfCommandFails([&]() { return this->computeCartesianPath(poses, 0.01, 0.0, trajectory); }, 10); // [s]

  if (fraction >= 0.0) {
    robot_trajectory::RobotTrajectory rt(this->getCurrentState()->getRobotModel(), this->getName());
    rt.setRobotTrajectoryMsg(*this->getCurrentState(), trajectory);

    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    iptp.computeTimeStamps(rt, velocity_rel * data.velocity_rel, acceleration_rel * data.acceleration_rel);

    rt.getRobotTrajectoryMsg(trajectory);
    my_plan.trajectory_ = trajectory;

    this->stop();
    this->is_moving = true;

    auto execute_return = restartMoveItIfCommandFails([&]() { return this->execute(my_plan); }, 10); // [s]

    this->is_moving = false;
    execution_success = (execute_return == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  } else {
    ROS_FATAL_STREAM("Error in planning motion");
    return false;
  }

  if (current_motion_data->break_callback && current_motion_data->did_break) {
    current_motion_data->break_callback();
  }

  data = *current_motion_data;
  current_motion_data = std::make_shared<MotionData>();
  return execution_success;
}

bool Robot::positionHold(double duration, const MotionData& data) {
  MotionData data_non_const = data;
  positionHold(duration, data_non_const);
}

bool Robot::positionHold(double duration, MotionData& data) {
  current_motion_data = std::make_shared<MotionData>(data);

  this->is_moving = true;
  std::unique_lock<std::mutex> lock(mutex);
  wait_condition_variable.wait_for(lock, std::chrono::duration<double, std::milli>(1000 * duration), [this]() { return flag; } ); // [s]
  this->is_moving = false;

  data = *current_motion_data;

  current_motion_data = std::make_shared<MotionData>();
  return true;
}
