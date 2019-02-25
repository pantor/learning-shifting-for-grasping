#pragma once

#include <cereal/archives/json.hpp>

#include <bin_picking/action.hpp>


struct Push: public Action {
  enum class Direction {
    Up,
    Left,
  } direction {Direction::Up};

  bool reverse {false};

  Push() {}
  Push(Action action): Action(action) { }

  template<class Archive>
  void serialize(Archive & archive) {
    archive(CEREAL_NVP(pose), CEREAL_NVP(found), CEREAL_NVP(prob), CEREAL_NVP(probstd), CEREAL_NVP(method), CEREAL_NVP(direction));
  }
};


struct PushResult: public ActionResult {
  explicit PushResult(): ActionResult() { }

  Push push;
  RobotPose start;
  RobotPose end;

  double probstart;
  double probend;

  template<class Archive>
  void serialize(Archive & archive) {
    archive(CEREAL_NVP(id), CEREAL_NVP(reward), CEREAL_NVP(collision), cereal::make_nvp("action", push), CEREAL_NVP(start), CEREAL_NVP(end), CEREAL_NVP(probstart), CEREAL_NVP(probend));
  }
};


namespace cereal {
  template <class Archive> inline
  std::string save_minimal(Archive const &, Push::Direction const & direction) {
    return (direction == Push::Direction::Up) ? "0" : "1";
  }

  template <class Archive> inline
  void load_minimal(Archive const &, Push::Direction & t, std::string const & value) {
    t = (value == "0") ? Push::Direction::Up : Push::Direction::Left;
  }
}