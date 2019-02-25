#include <frankr/gripper.hpp>


Gripper::Gripper(std::string fci_ip): franka::Gripper(fci_ip) { }

double Gripper::width() const {
  auto state = ((franka::Gripper*) this)->readOnce();
  return state.width + width_calibration;
}

bool Gripper::homing() const {
  return ((franka::Gripper*) this)->homing();
}

bool Gripper::stop() const {
  return ((franka::Gripper*) this)->stop();
}

bool Gripper::is_grasping() const {
  const bool libfranka_is_grasped = ((franka::Gripper*) this)->readOnce().is_grasped;
  const bool width_is_grasped = std::abs(this->width() - last_clamp_width) < 0.003;
  return libfranka_is_grasped && width_is_grasped;
}

bool Gripper::move(double width) { // [m]
  try {
    return ((franka::Gripper*) this)->move(width - width_calibration, gripper_speed); // [m] [m/s]
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    this->stop();
    this->homing();
    return ((franka::Gripper*) this)->move(width - width_calibration, gripper_speed); // [m] [m/s]
  }
}

std::future<bool> Gripper::moveAsync(double width) { // [m]
  return std::async(std::launch::async, &Gripper::move, this, width - width_calibration);
}

void Gripper::open() {
  move(max_width);
}

bool Gripper::clamp() {
  const bool success = this->grasp(min_width, gripper_speed, gripper_force, 0.0, 1.0); // [m] [m/s] [N] [m] [m]
  last_clamp_width = this->width();
  return success;
}

void Gripper::release(double width) { // [m]
  try {
    this->stop();
    this->move(width);
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    this->homing();
    this->stop();
    this->move(width);
  }
}
