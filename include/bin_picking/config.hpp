#pragma once

#include <yaml-cpp/yaml.h>

#include <ensenso/config.hpp>
#include <realsense/config.hpp>
#include <frankr/geometry.hpp>
#include <bin_picking/bin.hpp>
#include <bin_picking/epoch.hpp>
#include <bin_picking/parameter.hpp>


namespace YAML {

template <> struct convert<Eigen::Affine3d> {
  static Node encode(const Eigen::Affine3d &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, Eigen::Affine3d &rhs) {
    auto vec = node.as<std::vector<double>>();
    rhs = Affine(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]);
    return true;
  }
};

template <> struct convert<Bin> {
  static Node encode(const Bin &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, Bin &rhs) {
    rhs = binName(node.as<std::string>());
    return true;
  }
};

template <> struct convert<Camera> {
  static Node encode(const Camera &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, Camera &rhs) {
    rhs = cameraName(node.as<std::string>());
    return true;
  }
};

template <> struct convert<Mode> {
  static Node encode(const Mode &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, Mode &rhs) {
    rhs = modeName(node.as<std::string>());
    return true;
  }
};

template <> struct convert<GraspType> {
  static Node encode(const GraspType &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, GraspType &rhs) {
    rhs = graspTypeName(node.as<std::string>());
    return true;
  }
};

template <> struct convert<BinData> {
  static Node encode(const BinData &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, BinData &rhs) {
    rhs.top_left = cv::Point2f{node["top_left"][0].as<float>(), node["top_left"][1].as<float>()};
    rhs.bottom_right = cv::Point2f{node["bottom_right"][0].as<float>(), node["bottom_right"][1].as<float>()};
    rhs.background_color = cv::Scalar{255 * node["height"].as<double>()};
    return true;
  }
};

template <> struct convert<Epoch> {
  static Node encode(const Epoch &) { throw std::runtime_error("Not implemented!"); }

  static bool decode(const Node &node, Epoch &rhs) {
    rhs = Epoch(node["number_cycles"].as<int>(), selectionMethodName(node["method_primary"].as<std::string>()));
    if (node["percentage_secondary"]) {
      rhs.percentage_secondary = node["percentage_secondary"].as<double>();
    }
    if (node["method_secondary"]) {
      rhs.selection_method_secondary = selectionMethodName(node["method_secondary"].as<std::string>());
    }
    return true;
  }
};

}



struct Config {
  std::string database_url;
  std::string learning_url;

  std::string grasp_database;
  std::string grasp_model;

  std::string push_database;
  std::string push_model;

  std::vector<Epoch> epochs;

  Bin start_bin;
  Camera camera;
  Mode mode;
  GraspType grasp_type;

  bool use_ensenso_node;
  bool use_cpp_inference;
  bool continuous_inference;
  bool check_grasp_second_time;
  bool adjust_grasp_second_time;
  bool show_live_actions;
  bool show_live_heatmap;

  std::string evaluation_result;
  int change_bin_at_number_of_success_grasps;
  int number_objects_in_bin;

  int change_bin_at_number_of_failed_grasps;
  double change_bin_at_max_probability;

  bool wait_for_force;
  bool change_bins;
  bool push_objects;
  bool release_in_other_bin;
  bool release_as_fast_as_possible;
  bool random_pose_before_release;
  bool set_zero_reward;
  bool home_gripper;

  bool take_direct_images;
  bool take_after_image;
  bool take_side_images;

  double image_distance_from_pose;
  double approach_distance_from_pose;
  double move_down_distance_for_release;
  double general_force_condition_threshold;
  double general_dynamics_rel;
  double approach_dynamics_rel;
  double gripper_speed;
  double measurement_gripper_force;
  double performance_gripper_force;

  bool train_model;
  int train_model_every_number_cycles;

  EnsensoConfig ensenso;
  RealsenseConfig realsense;

  std::vector<double> gripper_classes;

  BinData bin_data;

  std::vector<double> bin_size_rect_top_left;
  std::vector<double> bin_size_rect_bottom_right;

  Eigen::Affine3d lower_random_affine_before_action;
  Eigen::Affine3d upper_random_affine_before_action;
  Eigen::Affine3d max_random_affine_before_release;


  // Pushing
  double grasp_push_threshold;
  double push_empty_threshold;
  double push_distance;
  std::string secondary_model;


  // Constants
  const Eigen::Affine3d camera_frame {Affine(-0.079, -0.0005, 0.011, -M_PI_4, 0.0, -M_PI)};
  const Eigen::Affine3d gripper_frame {Affine(0.0, 0.0, 0.18, -M_PI_4, 0.0, -M_PI)};

  const std::map<Bin, const Eigen::Affine3d> bin_frames {
    {Bin::Left, Affine(0.480, -0.125, 0.01, M_PI_2)},
    {Bin::Right, Affine(0.480, 0.125, 0.01, M_PI_2)}
  };

  const std::map<Bin, const std::array<double, 7>> bin_joint_values {
    {Bin::Left, {-1.8119446041276943, 1.1791089121678338, 1.7571002245448795, -2.141621800118565, -1.1433693913722132, 1.6330460616663416, -0.4321716643888135}},
    {Bin::Right, {-1.4637412426804741, 1.0494154046592876, 1.7926908288289254, -2.283032105735691, -1.0354444001306924, 1.7528634854002052, 0.043251646500343466}}
  };

  explicit Config(const std::string& config_file) {
    YAML::Node config = YAML::LoadFile(config_file);

    database_url = config["url"]["database"].as<std::string>();
    learning_url = config["url"]["learning"].as<std::string>();

    grasp_database = config["grasp_database"].as<std::string>();
    grasp_model = config["grasp_model"].as<std::string>() + "-sm";

    push_database = config["push_database"].as<std::string>();
    push_model = config["push_model"].as<std::string>() + "-sm";

    epochs = config["epochs"].as<std::vector<Epoch>>();

    start_bin = config["start_bin"].as<Bin>();
    camera = config["camera"].as<Camera>();
    mode = config["mode"].as<Mode>();
    grasp_type = config["grasp_type"].as<GraspType>();

    use_ensenso_node = config["use_ensenso_node"].as<bool>();
    use_cpp_inference = config["use_cpp_inference"].as<bool>();

    continuous_inference = config["continuous_inference"].as<bool>();
    check_grasp_second_time = config["check_grasp_second_time"].as<bool>();
    adjust_grasp_second_time = config["adjust_grasp_second_time"].as<bool>();
    show_live_actions = config["show_live_actions"].as<bool>();
    show_live_heatmap = config["show_live_heatmap"].as<bool>();

    evaluation_result = config["evaluation_result"].as<std::string>();
    change_bin_at_number_of_success_grasps = config["change_bin_at_number_of_success_grasps"].as<int>();
    number_objects_in_bin = config["number_objects_in_bin"].as<int>();

    change_bin_at_number_of_failed_grasps = config["change_bin_at_number_of_failed_grasps"].as<int>();
    change_bin_at_max_probability = config["change_bin_at_max_probability"].as<double>();

    wait_for_force = config["wait_for_force"].as<bool>();
    push_objects = config["push_objects"].as<bool>();
    change_bins = config["change_bins"].as<bool>();
    release_in_other_bin = config["release_in_other_bin"].as<bool>();
    release_as_fast_as_possible = config["release_as_fast_as_possible"].as<bool>();
    random_pose_before_release = config["random_pose_before_release"].as<bool>();
    set_zero_reward = config["set_zero_reward"].as<bool>();
    home_gripper = config["home_gripper"].as<bool>();

    take_direct_images = config["take_direct_images"].as<bool>();
    take_after_image = config["take_after_image"].as<bool>();
    take_side_images = config["take_side_images"].as<bool>();

    image_distance_from_pose = config["image_distance_from_pose"].as<double>();
    approach_distance_from_pose = config["approach_distance_from_pose"].as<double>();
    move_down_distance_for_release = config["move_down_distance_for_release"].as<double>();
    general_force_condition_threshold = config["general_force_condition_threshold"].as<double>();
    general_dynamics_rel = config["general_dynamics_rel"].as<double>();
    approach_dynamics_rel = config["approach_dynamics_rel"].as<double>();
    gripper_speed = config["gripper_speed"].as<double>();
    measurement_gripper_force = config["measurement_gripper_force"].as<double>();
    performance_gripper_force = config["performance_gripper_force"].as<double>();

    train_model = config["train_model"].as<bool>();
    train_model_every_number_cycles = config["train_model_every_number_cycles"].as<int>();

    ensenso.id = config["ensenso"]["id"].as<std::string>();
    ensenso.pixel_size = config["ensenso"]["pixel_size"].as<double>();
    ensenso.min_depth = config["ensenso"]["min_depth"].as<double>();
    ensenso.max_depth = config["ensenso"]["max_depth"].as<double>();
    // ensenso.auto_exposure = config["ensenso"]["auto_exposure"].as<bool>();
    ensenso.use_open_gl = config["ensenso"]["use_open_gl"].as<bool>();
    ensenso.use_cuda = config["ensenso"]["use_cuda"].as<bool>();
    ensenso.cuda_device = config["ensenso"]["cuda_device"].as<int>();

    gripper_classes = config["gripper_classes"].as<std::vector<double>>();

    bin_data = config["bin_data"].as<BinData>();

    lower_random_affine_before_action = config["lower_random_pose"].as<Eigen::Affine3d>();
    upper_random_affine_before_action = config["upper_random_pose"].as<Eigen::Affine3d>();
    max_random_affine_before_release = config["max_random_affine_before_release"].as<Eigen::Affine3d>();

    // Pushing
    grasp_push_threshold = config["grasp_push_threshold"].as<double>();
    push_empty_threshold = config["push_empty_threshold"].as<double>();
    push_distance = config["push_distance"].as<double>();
    secondary_model = config["secondary_model"].as<std::string>();

    if (mode == Mode::Perform) {
      general_dynamics_rel *= 2.5;
      approach_dynamics_rel *= 0.8;
      gripper_speed *= 3.0;

      approach_distance_from_pose *= 0.7;
    }
  }
};
