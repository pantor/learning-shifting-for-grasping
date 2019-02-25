#pragma once

#include <chrono>
#include <cmath>
#include <random>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <bin_picking/action.hpp>
#include <bin_picking/inference_result.hpp>
#include <bin_picking/parameter.hpp>
#include <bin_picking/depth_image.hpp>
#include <bin_picking/image_utils.hpp>


class Net;


class Inference {
  template<typename T> struct TopComp {
    TopComp(const std::vector<T>& v): _v(v) {}
    bool operator()(T a, T b) { return _v[a] > _v[b]; }
    const std::vector<T>& _v;
  };

  template<typename T> struct BottomComp {
    BottomComp(const std::vector<T>& v): _v(v) {}
    bool operator()(T a, T b) { return _v[a] < _v[b]; }
    const std::vector<T>& _v;
  };

  std::string model_path;

  double resolution_factor {2.0};
  cv::Size size_input {752, 480};
  cv::Size size_rotated {160, 160};
  cv::Size size_output {32, 32};
  cv::Size size_cropped {110, 110};
  cv::Size size_original_cropped {200, 200};

  // Calculated in setSizeOriginalCropped
  cv::Size size_resized;
  std::array<double, 2> scale_factors;

  std::mt19937 random_generator;
  std::vector<double> lower_random_pose;
  std::vector<double> upper_random_pose;

  BinData bin;
  std::shared_ptr<Net> net;

  void createImageTensor(const DepthImage& depth_image, float* ptr);

public:
  std::string name {""};
  std::vector<double> a_space; // [rad]
  int number_actions {3};

  bool image_inpainting {false};

  std::array<int, 4> unravelIndex(int index, const std::array<int, 4>& shape) const {
    std::array<int, 4> result;
    int c = index;
    for (int i = shape.size() - 1; i >= 0; i--) {
      result[i] = c % shape[i];
      c = (c - result[i]) / shape[i];
    }
    return result;
  }

  template<typename T>
  static inline std::vector<double> linspace(T start_in, T end_in, int num_in) {
    std::vector<double> result;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return result; }
    if (num == 1) {
      result.push_back(start);
      return result;
    }

    double delta = (end - start) / (num - 1);
    for (int i = 0; i < num - 1; i += 1) {
      result.push_back(start + delta * i);
    }
    result.push_back(end);
    return result;
  }

  template<typename T=float>
  int selectIndex(const std::vector<T>& prob_vec, SelectionMethod method) {
    switch (method) {
      case SelectionMethod::Max: {
        return std::distance(prob_vec.begin(), std::max_element(prob_vec.begin(), prob_vec.end()));
      } break;
      case SelectionMethod::Min: {
        return std::distance(prob_vec.begin(), std::min_element(prob_vec.begin(), prob_vec.end()));
      } break;
      case SelectionMethod::Top5: {
        std::vector<int> vec_index(prob_vec.size());
        std::iota(std::begin(vec_index), std::end(vec_index), 0);
        std::partial_sort(vec_index.begin(), vec_index.begin() + 5, vec_index.end(), TopComp<T>(prob_vec));
        std::uniform_int_distribution<std::mt19937::result_type> dist_vec(0, 5 - 1);
        return vec_index[dist_vec(random_generator)];
      } break;
      case SelectionMethod::Bottom5: {
        std::vector<int> vec_index(prob_vec.size());
        std::iota(std::begin(vec_index), std::end(vec_index), 0);
        std::partial_sort(vec_index.begin(), vec_index.begin() + 5, vec_index.end(), BottomComp<T>(prob_vec));
        std::uniform_int_distribution<std::mt19937::result_type> dist_vec(0, 5 - 1);
        return vec_index[dist_vec(random_generator)];
      } break;
      case SelectionMethod::Random:
      case SelectionMethod::RandomInference: {
        std::uniform_int_distribution<std::mt19937::result_type> dist_vec(0, prob_vec.size() - 1);
        return dist_vec(random_generator);
      } break;
      case SelectionMethod::Uncertain: {
        T reference = 0.5;
        auto i = std::min_element(prob_vec.begin(), prob_vec.end(), [=](T x, T y) { return std::abs(x - reference) < std::abs(y - reference); });
        return std::distance(prob_vec.begin(), i);
      } break;
      case SelectionMethod::Prob: {
        std::vector<T> prob_vec_mutable(prob_vec.size());
        if (*std::min_element(prob_vec.begin(), prob_vec.end()) < 0.0) {
          std::transform(prob_vec.begin(), prob_vec.end(), prob_vec_mutable.begin(), [](double x){ return x * x; });
          std::discrete_distribution<> dist_vec(prob_vec_mutable.begin(), prob_vec_mutable.end());
          return dist_vec(random_generator);
        }
        std::discrete_distribution<> dist_vec(prob_vec.begin(), prob_vec.end());
        return dist_vec(random_generator);
      } break;
      default: {
        std::cerr << "Selection method not implemented." << std::endl;
        std::exit(0);
      }
    }
  }

  void setSizeOriginalCropped(const cv::Size& size) {
    size_original_cropped = size;
    size_resized = {size_input.width * size_output.width / size_original_cropped.width, size_input.height * size_output.height / size_original_cropped.height};
    scale_factors = {{resolution_factor * double(size_original_cropped.width) / size_output.width, resolution_factor * double(size_original_cropped.height) / size_output.height}};
  }

  explicit Inference() { }
  explicit Inference(const std::string& model_path, const BinData& bin);
  explicit Inference(const std::string& model_path, const BinData& bin, const std::vector<double>& lower_random_pose, const std::vector<double>& upper_random_pose);

  InferenceResult getRandom();

  InferenceResult infer(const DepthImage& depth_image, SelectionMethod method);
  InferenceResult inferDual(const DepthImage& depth_image, const DepthImage& raw_image, SelectionMethod method);
  double inferAtPose(const DepthImage& depth_image, const InferenceResult& inf);
  InferenceResult inferAroundPose(const DepthImage& depth_image, const RobotPose& pose, SelectionMethod method);
  InferenceResult inferAroundPoseDiscountedTime(const DepthImage& depth_image, const RobotPose& pose, SelectionMethod method);

  cv::Mat drawHeatmap(const DepthImage& depth_image);
};
