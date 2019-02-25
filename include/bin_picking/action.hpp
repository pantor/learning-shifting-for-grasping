#pragma once

#include <chrono>
#include <ctime>
#include <iostream>

#include <cereal/archives/json.hpp>

#include <bin_picking/parameter.hpp>


struct RobotPose {
  double x {0.0};
  double y {0.0};
  double z {0.0};
  double a {0.0};
  double b {0.0};
  double c {0.0};

  double d {0.0};

#ifdef BIN_PICKING_GEOMETRY
  RobotPose() { }

  RobotPose(const Eigen::Affine3d& affine) {
    auto vector = Vector(affine);
    x = vector(0);
    y = vector(1);
    z = vector(2);
    a = vector(3);
    b = vector(4);
    c = vector(5);
  }

  Eigen::Affine3d toAffine() {
    return Affine(x, y, z, a, b, c);
  }
#endif // BIN_PICKING_GEOMETRY_HPP

  template<class Archive>
  void serialize(Archive & archive) {
    archive(CEREAL_NVP(x), CEREAL_NVP(y), CEREAL_NVP(z), CEREAL_NVP(a), CEREAL_NVP(b), CEREAL_NVP(c), CEREAL_NVP(d));
  }
};


struct Action {
  RobotPose pose;

  int found {false};

  double prob {0.0};
  double probstd {0.0};

  SelectionMethod method {SelectionMethod::None};

  template<class Archive>
  void serialize(Archive & archive) {
    archive(CEREAL_NVP(pose), CEREAL_NVP(found), CEREAL_NVP(prob), CEREAL_NVP(probstd), CEREAL_NVP(method));
  }
};


struct ActionResult {
  std::string id;
  bool save {true};

  double reward {0.0};
  bool collision {false};

  explicit ActionResult() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    auto bt = *std::localtime(&timer);

    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y-%m-%d-%H-%M-%S-");
    oss << std::setfill('0') << std::setw(3) << ms.count(); // Add [ms] to id
    id = oss.str();
  }
};
