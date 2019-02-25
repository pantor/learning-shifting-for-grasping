#include <bin_picking/inference.hpp>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>


namespace tf = tensorflow;

class Net {
  tf::SessionOptions session_options;
  tf::RunOptions run_options;

public:
  tf::SavedModelBundle bundle;

  Net(const std::string& model) {
    tf::ConfigProto &config = session_options.config;
    config.mutable_gpu_options()->set_visible_device_list("0");
    config.mutable_gpu_options()->set_allow_growth(true);
    config.set_allow_soft_placement(true);

    tf::LoadSavedModel(session_options, run_options, model, {tf::kSavedModelTagServe}, &bundle);
  }
};



Inference::Inference(const std::string& model_path, const BinData& bin, const std::vector<double>& lower_random_pose, const std::vector<double>& upper_random_pose): model_path(model_path), bin(bin), lower_random_pose(lower_random_pose), upper_random_pose(upper_random_pose) {
  setSizeOriginalCropped(size_original_cropped);
  a_space = linspace(-1.484, 1.484, 20);
  random_generator.seed(std::random_device()());

  net = std::make_shared<Net>(model_path);
}

Inference::Inference(const std::string& model_path, const BinData& bin): Inference(model_path, bin, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) { }

void Inference::createImageTensor(const DepthImage& depth_image, float* ptr) {
  cv::Mat image_bin = dimg::drawAroundBin(depth_image, bin).image;

  cv::Mat image_resized {size_resized, image_bin.type()};
  cv::resize(image_bin, image_resized, size_resized);

  // Image inpainting
  if (image_inpainting) {
    cv::Mat mask = cv::Mat::zeros(image_resized.size(), CV_16U);
    mask = (image_resized == 0);
    cv::inpaint(image_resized, mask, image_resized, 3, cv::INPAINT_TELEA);
  }

  cv::Mat image_transformed {size_rotated, image_resized.type()};

  int idx {0};
  for (double a: a_space) {
    auto rot_mat = cv::getRotationMatrix2D(cv::Point(size_resized.width / 2, size_resized.height / 2), a * 180.0 / M_PI, 1.0); // [deg]
    rot_mat.at<double>(0, 2) += (size_rotated.width - size_resized.width) / 2;
    rot_mat.at<double>(1, 2) += (size_rotated.height - size_resized.height) / 2;

    cv::warpAffine(image_resized, image_transformed, rot_mat, size_rotated, cv::INTER_LINEAR, cv::BORDER_CONSTANT, bin.background_color);

    cv::Mat image_cropped = dimg::crop(image_transformed, size_cropped);
    image_cropped.convertTo(image_cropped, CV_32F);
    image_cropped /= (255.0 * 255.0);
    std::copy_n(image_cropped.begin<float>(), image_cropped.total(), ptr + idx * image_cropped.total());
    idx += 1;
  }
}

InferenceResult Inference::getRandom() {
  std::uniform_real_distribution<double> x_dist(lower_random_pose[0], upper_random_pose[0]);
  std::uniform_real_distribution<double> y_dist(lower_random_pose[1], upper_random_pose[1]);
  std::uniform_real_distribution<double> a_dist(lower_random_pose[3], upper_random_pose[3]);
  std::uniform_real_distribution<double> b_dist(lower_random_pose[4], upper_random_pose[4]);
  std::uniform_real_distribution<double> c_dist(lower_random_pose[5], upper_random_pose[5]);
  std::uniform_int_distribution<int> index_dist(0, number_actions - 1);

  InferenceResult result;
  result.x = x_dist(random_generator); // [m]
  result.y = y_dist(random_generator); // [m]
  result.a = a_dist(random_generator); // [rad]
  result.index = index_dist(random_generator);
  return result;
}

cv::Mat Inference::drawHeatmap(const DepthImage& depth_image) {
  tf::Tensor image_tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<int>(a_space.size()), size_cropped.width, size_cropped.height, 1}));
  std::vector<tf::Tensor> output_tensor {20};

  createImageTensor(depth_image, image_tensor.flat<float>().data());

  net->bundle.session->Run({{"image:0", image_tensor}}, {"prob:0"}, {}, &output_tensor);

  const auto prob = output_tensor[0].tensor<float, 4>();

  cv::Mat image_rgb;
  cv::cvtColor(depth_image.image / 255.0, image_rgb, cv::COLOR_GRAY2BGR);
  image_rgb.convertTo(image_rgb, CV_8UC3);

  cv::Size size_first {520, 520};
  cv::Mat prob_all = cv::Mat::zeros(size_first, CV_32F);

  cv::Mat prob_a;
  for (int a_idx = 0; a_idx < a_space.size(); a_idx++) {
    prob_a = cv::Mat::zeros(cv::Size(40, 40), CV_32F);
    for (int i = 0; i < 40; i++) {
      for (int j = 0; j < 40; j++) {
        prob_a.at<float>(i, j) = std::max(prob(a_idx, i, j, 0), std::max(prob(a_idx, i, j, 1), prob(a_idx, i, j, 2)));
      }
    }

    auto rot_mat = cv::getRotationMatrix2D(cv::Point(20, 20), -a_space[a_idx] * 180.0 / M_PI, 1.0);
    cv::warpAffine(prob_a, prob_a, rot_mat, cv::Size(40, 40), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::resize(prob_a, prob_a, size_first);
    prob_all += prob_a;
  }

  prob_all *= 255.0 / a_space.size();
  prob_all.convertTo(prob_all, CV_8UC3);

  cv::Rect cropped_rect {0, 20, 520, 480};
  cv::Mat prob_all_cropped = prob_all(cropped_rect);
  cv::copyMakeBorder(prob_all_cropped, prob_all, 0, 0, int((752 - size_first.width) / 2), int((752 - size_first.width) / 2), cv::BORDER_CONSTANT, 0);

  cv::Mat heatmap;
  cv::applyColorMap(prob_all, heatmap, cv::COLORMAP_JET);
  float alpha {0.5};
  return (1 - alpha) * image_rgb + alpha * heatmap;
}

InferenceResult Inference::infer(const DepthImage& depth_image, SelectionMethod method) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  if (method == SelectionMethod::Random) {
    return getRandom();
  }

  tf::Tensor image_tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<int>(a_space.size()), size_cropped.width, size_cropped.height, 1}));
  createImageTensor(depth_image, image_tensor.flat<float>().data());

  std::vector<tf::Tensor> output_tensor {20};
  net->bundle.session->Run({{"image:0", image_tensor}}, {"prob:0"}, {}, &output_tensor);

  const auto prob = output_tensor[0].tensor<float, 4>();
  std::vector<float> prob_vec(prob.data(), prob.data() + prob.size());
  const std::array<int, 4> shape {static_cast<int>(prob.dimension(0)), static_cast<int>(prob.dimension(1)), static_cast<int>(prob.dimension(2)), static_cast<int>(prob.dimension(3))};
  // TODO Implement BAYES in C++

  int index_vec = selectIndex(prob_vec, method);
  std::array<int, 4> index = unravelIndex(index_vec, shape);

  // Calculate grasp point in robot system
  double a = -a_space[index[0]];
  cv::Point2d vec = dimg::rotate(cv::Point2d(depth_image.positionFromIndex(index[1], shape[1]), depth_image.positionFromIndex(index[2], shape[2])), a);

  InferenceResult result;
  result.x = -vec.x * scale_factors[0]; // [m]
  result.y = -vec.y * scale_factors[1]; // [m]
  result.a = a;
  result.index = index[3];
  result.prob = prob_vec[index_vec];
  result.method = method;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Inference " << name << " prob: " << result.prob << " method: " << selectionMethodName(method) << std::endl;
  std::cout << "Inference time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
  return result;
}

InferenceResult Inference::inferDual(const DepthImage& depth_image, const DepthImage& raw_image, SelectionMethod method) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  if (method == SelectionMethod::Random) {
    return getRandom();
  }

  tf::Tensor depth_image_tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<int>(a_space.size()), size_cropped.width, size_cropped.height, 1}));
  tf::Tensor raw_image_tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<int>(a_space.size()), size_cropped.width, size_cropped.height, 1}));
  createImageTensor(depth_image, depth_image_tensor.flat<float>().data());
  createImageTensor(raw_image, raw_image_tensor.flat<float>().data());

  std::vector<tf::Tensor> output_tensor {20};
  net->bundle.session->Run({{"depth_image:0", depth_image_tensor}, {"raw_image:0", raw_image_tensor}}, {"prob:0"}, {}, &output_tensor);

  const auto prob = output_tensor[0].tensor<float, 4>();
  std::vector<float> prob_vec(prob.data(), prob.data() + prob.size());
  const std::array<int, 4> shape {static_cast<int>(prob.dimension(0)), static_cast<int>(prob.dimension(1)), static_cast<int>(prob.dimension(2)), static_cast<int>(prob.dimension(3))};
  // TODO Implement BAYES in C++

  int index_vec = selectIndex(prob_vec, method);
  std::array<int, 4> index = unravelIndex(index_vec, shape);

  // Calculate grasp point in robot system
  double a = -a_space[index[0]];
  cv::Point2d vec = dimg::rotate(cv::Point2d(depth_image.positionFromIndex(index[1], shape[1]), depth_image.positionFromIndex(index[2], shape[2])), a);

  InferenceResult result;
  result.x = -vec.x * scale_factors[0]; // [m]
  result.y = -vec.y * scale_factors[1]; // [m]
  result.a = a;
  result.index = index[3];
  result.prob = prob_vec[index_vec];
  result.method = method;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Inference " << name << " prob: " << result.prob << " method: " << selectionMethodName(method) << std::endl;
  std::cout << "Inference time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
  return result;
}

double Inference::inferAtPose(const DepthImage& depth_image, const InferenceResult& inf) {
  // DepthImage dimg_bin = dimg::drawAroundBin(depth_image, bin);

  RobotPose pose;
  pose.x = inf.x;
  pose.y = inf.y;
  pose.a = inf.a;

  DepthImage image_transformed = dimg::getAreaOfInterest(depth_image, pose, size_input, bin.background_color);

  cv::Mat image_resized {size_resized, depth_image.image.type()};
  cv::resize(image_transformed.image, image_resized, size_resized);
  cv::Mat image_output = dimg::crop(image_resized, size_output);

  image_output.convertTo(image_output, CV_32F);
  image_output /= (255.0 * 255.0);

  tf::Tensor image_tensor(tf::DT_FLOAT, tf::TensorShape({1, size_output.width, size_output.height, 1}));
  std::copy_n(image_output.begin<float>(), image_output.total(), image_tensor.flat<float>().data());

  std::vector<tf::Tensor> output_tensor {1};
  net->bundle.session->Run({{"image:0", image_tensor}}, {"prob:0"}, {}, &output_tensor);

  const auto prob = output_tensor[0].tensor<float, 4>();
  std::vector<float> prob_vec(prob.data(), prob.data() + prob.size());
  return prob_vec[inf.index];
}

// SelectionMethod either MAX or TOP_5
InferenceResult Inference::inferAroundPose(const DepthImage& depth_image, const RobotPose& pose, SelectionMethod method) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  tf::Tensor image_tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<int>(a_space.size()), size_cropped.width, size_cropped.height, 1}));
  createImageTensor(depth_image, image_tensor.flat<float>().data());

  std::vector<tf::Tensor> output_tensor {20};
  net->bundle.session->Run({{"image:0", image_tensor}}, {"prob:0"}, {}, &output_tensor);

  const auto prob = output_tensor[0].tensor<float, 4>();
  std::vector<float> prob_vec(prob.data(), prob.data() + prob.size());
  const std::array<int, 4> shape {static_cast<int>(prob.dimension(0)), static_cast<int>(prob.dimension(1)), static_cast<int>(prob.dimension(2)), static_cast<int>(prob.dimension(3))};
  // TODO Implement BAYES in C++

  {
    cv::Size_<double> window_size {0.13, 0.13}; // 240.0 / depth_image.pixel_size, 240.0 / depth_image.pixel_size};

    for (int i = 0; i < prob_vec.size(); i+= 1) {
      std::array<int, 4> index = unravelIndex(i, shape);

      double a = -a_space[index[0]];
      cv::Point2d vec = dimg::rotate(cv::Point2d(depth_image.positionFromIndex(index[1], shape[1]), depth_image.positionFromIndex(index[2], shape[2])), a);

      cv::Point2d center = cv::Point2d{-scale_factors[0] * vec.x, -scale_factors[1] * vec.y}; // [m]
      center.x -= pose.x;
      center.y -= pose.y;
      cv::Point2d center_rot = dimg::rotate(center, pose.a);

      // Outside the rectangle
      if (center_rot.x < -window_size.width / 2 || window_size.width / 2 < center_rot.x || center_rot.y < -window_size.height / 2 || window_size.height / 2 < center_rot.y) {
        prob_vec[i] = 0.0;
      }
    }
  }

  int index_vec = selectIndex(prob_vec, method);
  std::array<int, 4> index = unravelIndex(index_vec, shape);

  // Calculate grasp point in robot system
  double a = -a_space[index[0]];
  cv::Point2d vec = dimg::rotate(cv::Point2d(depth_image.positionFromIndex(index[1], shape[1]), depth_image.positionFromIndex(index[2], shape[2])), a);

  InferenceResult result;
  result.x = -vec.x * scale_factors[0]; // [m]
  result.y = -vec.y * scale_factors[1]; // [m]
  result.a = a;
  result.index = index[3];
  result.prob = prob_vec[index_vec];
  result.method = method;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Inference " << name << " prob: " << result.prob << " method: " << selectionMethodName(method) << std::endl;
  std::cout << "Inference time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
  return result;
}

InferenceResult Inference::inferAroundPoseDiscountedTime(const DepthImage& depth_image, const RobotPose& pose, SelectionMethod method) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // DepthImage depth_image_bin = dimg::drawAroundBin(depth_image, bin, global_pose);

  tf::Tensor image_tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<int>(a_space.size()), size_cropped.width, size_cropped.height, 1}));
  createImageTensor(depth_image, image_tensor.flat<float>().data());

  std::vector<tf::Tensor> output_tensor {20};
  net->bundle.session->Run({{"image:0", image_tensor}}, {"prob:0"}, {}, &output_tensor);

  const auto prob = output_tensor[0].tensor<float, 4>();
  std::vector<float> prob_vec(prob.data(), prob.data() + prob.size());
  const std::array<int, 4> shape {static_cast<int>(prob.dimension(0)), static_cast<int>(prob.dimension(1)), static_cast<int>(prob.dimension(2)), static_cast<int>(prob.dimension(3))};

  std::vector<float> new_prob_vec(prob_vec.size());

  {
    double translation_velocity {0.5}; // [m/s]
    double rotation_velocity {0.5}; // [rad/s]
    double gripper_velocity {0.06}; // [m/s]

    double reward_discount_per_time {10.0}; // [grasp probability / s]

    std::array<double, 3> gripper_classes {0.05, 0.07, 0.086}; // [m]

    for (int i = 0; i < prob_vec.size(); i+= 1) {
      std::array<int, 4> index = unravelIndex(i, shape);

      double a = -a_space[index[0]];
      cv::Point2d vec = dimg::rotate(cv::Point2d(depth_image.positionFromIndex(index[1], shape[1]), depth_image.positionFromIndex(index[2], shape[2])), a);
      cv::Point2d point = cv::Point2d{-scale_factors[0] * vec.x, -scale_factors[1] * vec.y};

      double time_to_move {0.0};
      time_to_move += cv::norm(point - cv::Point2d{pose.x, pose.y}) / translation_velocity;
      time_to_move += std::abs(a - pose.a) / rotation_velocity;
      time_to_move += std::abs(gripper_classes.at(index[3]) - pose.d) / gripper_velocity;

      new_prob_vec[i] = std::max(prob_vec[i] - reward_discount_per_time * time_to_move, 0.0);
    }
  }

  int index_vec = selectIndex(new_prob_vec, method);
  std::array<int, 4> index = unravelIndex(index_vec, shape);

  // Calculate grasp point in robot system
  double a = -a_space[index[0]];
  cv::Point2d vec = dimg::rotate(cv::Point2d(depth_image.positionFromIndex(index[1], shape[1]), depth_image.positionFromIndex(index[2], shape[2])), a);

  InferenceResult result;
  result.x = -vec.x * scale_factors[0]; // [m]
  result.y = -vec.y * scale_factors[1]; // [m]
  result.a = a;
  result.index = index[3];
  result.prob = prob_vec[index_vec];
  result.method = method;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Inference " << name << " prob: " << result.prob << " method: " << selectionMethodName(method) << std::endl;
  std::cout << "Inference time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
  return result;
}
