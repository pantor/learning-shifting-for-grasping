#pragma once

#include <cereal/archives/json.hpp>

#include <bin_picking/action.hpp>


struct Grasp: public Action {
  Grasp() {}
  Grasp(Action action): Action(action) { }
};


struct GraspResult: public ActionResult {
  explicit GraspResult(): ActionResult() { }

  Grasp grasp;
  RobotPose final;

  template<class Archive>
  void serialize(Archive & archive) {
    archive(CEREAL_NVP(id), CEREAL_NVP(reward), CEREAL_NVP(collision), cereal::make_nvp("action", grasp), CEREAL_NVP(final));
  }
};
