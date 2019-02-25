#pragma once

#include <cereal/archives/json.hpp>

#include <bin_picking/parameter.hpp>


struct InferenceResult {
  double x;
  double y;
  double a;

  int index;

  double prob {-1.0};
  double probstd {0.0};

  SelectionMethod method {SelectionMethod::None};

  template<class Archive>
  void serialize(Archive & archive) {
    archive(CEREAL_NVP(x), CEREAL_NVP(y), CEREAL_NVP(a), CEREAL_NVP(index), CEREAL_NVP(prob), CEREAL_NVP(probstd), CEREAL_NVP(method));
  }
};
