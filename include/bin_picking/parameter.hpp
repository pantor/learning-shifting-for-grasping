#pragma once

#include <map>
#include <string>


enum class Bin {
  Left,
  Right,
};

enum class Camera {
  Ensenso,
  Realsense,
  Both,
};

enum class Mode {
  Measure,
  Evaluate,
  Perform,
};

enum class SelectionMethod {
  Max,
  Min,
  Top5,
  Bottom5,
  Random, // Random from pose number generator
  RandomInference, // Random from neural network inference
  Uncertain,
  NotZero,
  Prob,
  Bayes,
  BayesTop,
  BayesProb,
  None,
};

enum class GraspType {
  Default,
  Specific,
  Type,
};


inline Bin binName(const std::string& name) {
  std::map<std::string, Bin> map_name_bin {
    {"LEFT", Bin::Left},
    {"RIGHT", Bin::Right},
  };
  return map_name_bin.at(name);
}

inline Camera cameraName(const std::string& name) {
  std::map<std::string, Camera> map_name_camera {
    {"ENSENSO", Camera::Ensenso},
    {"REALSENSE", Camera::Realsense},
    {"BOTH", Camera::Both},
  };
  return map_name_camera.at(name);
}

inline Mode modeName(const std::string& name) {
  std::map<std::string, Mode> map_name_mode {
    {"MEASURE", Mode::Measure},
    {"EVALUATE", Mode::Evaluate},
    {"PERFORM", Mode::Perform},
  };
  return map_name_mode.at(name);
}

inline GraspType graspTypeName(const std::string& name) {
  std::map<std::string, GraspType> grasp_type_name_mode {
    {"DEFAULT", GraspType::Default},
    {"SPECIFIC", GraspType::Specific},
    {"TYPE", GraspType::Type},
  };
  return grasp_type_name_mode.at(name);
}

inline std::string selectionMethodName(const SelectionMethod& method) {
  const std::map<SelectionMethod, std::string> map_selection_method_name {
    {SelectionMethod::Max, "MAX"},
    {SelectionMethod::Min, "MIN"},
    {SelectionMethod::Top5, "TOP_5"},
    {SelectionMethod::Bottom5, "BOTTOM_5"},
    {SelectionMethod::Random, "RANDOM"},
    {SelectionMethod::RandomInference, "RANDOM_INFERENCE"},
    {SelectionMethod::Uncertain, "UNCERTAIN"},
    {SelectionMethod::NotZero, "NOT_ZERO"},
    {SelectionMethod::Prob, "PROB"},
    {SelectionMethod::Bayes, "BAYES"},
    {SelectionMethod::BayesTop, "BAYES_TOP"},
    {SelectionMethod::BayesProb, "BAYES_PROB"},
    {SelectionMethod::None, "NONE"},
  };
  return map_selection_method_name.at(method);
}

inline SelectionMethod selectionMethodName(const std::string& name) {
  std::map<std::string, SelectionMethod> map_name_selection_method {
    {"MAX", SelectionMethod::Max},
    {"MIN", SelectionMethod::Min},
    {"TOP_5", SelectionMethod::Top5},
    {"BOTTOM_5", SelectionMethod::Bottom5},
    {"RANDOM", SelectionMethod::Random},
    {"RANDOM_INFERENCE", SelectionMethod::RandomInference},
    {"UNCERTAIN", SelectionMethod::Uncertain},
    {"NOT_ZERO", SelectionMethod::NotZero},
    {"PROB", SelectionMethod::Prob},
    {"BAYES", SelectionMethod::Bayes},
    {"BAYES_TOP", SelectionMethod::BayesTop},
    {"BAYES_PROB", SelectionMethod::BayesProb},
    {"NONE", SelectionMethod::None},
  };
  return map_name_selection_method.at(name);
}


// Serialize selection method for cereal
namespace cereal {
  template <class Archive> inline
  std::string save_minimal(Archive const &, SelectionMethod const & method) {
    return selectionMethodName(method);
  }

  template <class Archive> inline
  void load_minimal(Archive const &, SelectionMethod & t, std::string const & value) {
    t = selectionMethodName(value);
  }
}
