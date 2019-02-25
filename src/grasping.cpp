#include <bin_picking/agent.hpp>
#include <bin_picking/depth_image.hpp>

#include <grasping/grasp.hpp>
#include <grasping/grasp_converter.hpp>

#include <pushing/push.hpp>
#include <pushing/push_converter.hpp>


class Grasping: public Agent {
  int count_successfull_grasps {0};
  int count_successfull_grasps_in_current_bin {0};
  int count_failed_grasps_since_last_success {0};

  std::thread continuous_inference_thread;
  std::mutex prob_mutex;
  float current_prob; // For CHECK_GRASP_SECOND_TIME

  GraspConverter grasp_converter;
  PushConverter push_converter;

  double getMaxGraspProb(const std::vector<DepthImage>& images) {
    return grasp_inference->infer(images[0], SelectionMethod::Max).prob;
  }

  double getMaxGraspProbAroundPose(const std::vector<DepthImage>& images, const RobotPose& pose) {
    return grasp_inference->inferAroundPose(images[0], pose, SelectionMethod::Max).prob;
  }

public:
  explicit Grasping(const std::string& config_filename): Agent(config_filename) {
    gripper->gripper_force = std::map<Mode, double>{
      {Mode::Measure, config.measurement_gripper_force},
      {Mode::Evaluate, config.performance_gripper_force},
      {Mode::Perform, config.performance_gripper_force}
    }[config.mode];

    grasp_converter.bin = config.bin_data;
    grasp_converter.gripper_classes = config.gripper_classes;
    grasp_converter.pose_should_be_inside_bin = true;

    push_converter.bin = config.bin_data;
    push_converter.pose_should_be_inside_bin = true;
    
    if (config.continuous_inference) {
      continuous_inference_thread = std::thread(&Grasping::continuousInference, this);
      // continuous_inference_thread.join();
    }
  }

  void continuousInference() {
    if (!config.use_cpp_inference) {
      std::cout << "Use C++ for continuous Inference" << std::endl;
      std::exit(0);
    }

    if (config.show_live_actions || config.show_live_heatmap) {
      cv::namedWindow("image");
    }

    Grasp new_grasp;

    while (ros::ok()) {
      prob_mutex.lock();
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

      // Measure robot pose directly before depth image
      auto robot_pose_future = std::async(std::launch::async, [&]() { return robot->currentPose(config.camera_frame); });
      auto gripper_width_future = std::async(std::launch::async, [&]() { return gripper->width(); });

      auto images = takeImages();

      gripper_width_future.wait();

      // new_grasp = inference->infer(images[0], SelectionMethod::MAX);

      new_grasp.pose.x = config.camera_frame.translation()(0);
      new_grasp.pose.y = config.camera_frame.translation()(1);
      new_grasp.pose.a = 0.0;
      new_grasp.pose.d = gripper_width_future.get();
      // new_grasp.prob = inference->inferAtPose(images[0], grasp_converter.convertBack(new_grasp));

      current_prob = new_grasp.prob;
      prob_mutex.unlock();

      if (config.show_live_actions) {
        robot_pose_future.wait();
        auto robot_pose = robot_pose_future.get();

        auto shown_image = dimg::drawAroundBin(images[0], config.bin_data, robot_pose, config.bin_frames.at(current_bin) * Affine(0.0, 0.0, 0.0, -M_PI_2));
        // std::cout << "Action prob: " << new_grasp.prob << std::endl;

        if (new_grasp.prob > 0.2) {
          shown_image = dimg::drawPose(shown_image, new_grasp.pose);
        }

        cv::imshow("image", shown_image.image);
        cv::waitKey(1);
      } else if (config.show_live_heatmap) {
        auto shown_image = grasp_inference->drawHeatmap(images[0]);

        cv::imshow("image", shown_image);
        cv::waitKey(1);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Image time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    }
  }

  void shutdown() {
    continuous_inference_thread.join();
  }

  void reset() {
    robot->recoverFromErrors();
    gripper->stop();

    robot->moveRelativeCartesian(config.gripper_frame, relativeSafePose(robot->currentPose(config.gripper_frame)), MotionData().withDynamics(0.3));
    robot->moveJoints(config.bin_joint_values.at(current_bin), MotionData().withDynamics(0.7));
    robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));

    if (config.mode != Mode::Perform && config.home_gripper) {
      // gripper->homing();
    }
  }

  void runEpisode(const Epoch epoch, int current_episode_in_epoch) {
    auto grasp_result = GraspResult();

    // Move to camera position, but with elbow configuration
    robot->recoverFromErrors();
    robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));
    if (config.mode != Mode::Perform || current_episode_in_epoch % 8 == 0) {
      robot->moveJoints(config.bin_joint_values.at(current_bin));
      robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));
    }

    // Take side images
    if (config.take_side_images) {
      Eigen::Affine3d center_of_rotation = getBinAffine(current_bin);
      double distance = 0.295;
      center_of_rotation.translation()(2) += 0.065;

      for (double b: std::vector<double>({-0.5, 0.0})) {
        auto side_affine = center_of_rotation * Affine(0, 0, 0, 0, b, 0) * Affine(0, 0, distance);
        robot->moveCartesian(config.camera_frame, side_affine);
        auto robot_pose = robot->currentPose(config.camera_frame);
        auto images = takeImages();
        images[0] = dimg::drawAroundBin(images[0], config.bin_data, robot_pose, config.bin_frames.at(current_bin) * Affine(0.0, 0.0, 0.0, -M_PI_2));
        grasp_io->uploadImage(grasp_result.id, images[0], "s-" + std::to_string((int)(b * 100)) + "-" + std::to_string(0));
      }

      /* for (double c: {-0.5, -0.25, 0.25, 0.5}) {
        auto side_affine = center_of_rotation * Affine(0, 0, 0, 0, 0, c) * Affine(0, 0, distance);
        robot->moveCartesian(config.camera_frame, side_affine);
        grasp_io->uploadImage(grasp_result.id, takeImages()[0], "s-" + std::to_string(0) + "-" + std::to_string((int)(c * 100)));
      } */

      robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));
      grasp_result.grasp.method = SelectionMethod::None;
      grasp_io->saveResult(grasp_result);
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1000)); // [ms]
      return;
    }

    // Wait for force
    if (config.wait_for_force) {
      std::cout << "Waiting for force before next grasp..." << std::endl;
      auto position_hold_data = MotionData().withZForceCondition(10.0).withXYForceCondition(10.0); // [N]
      robot->positionHold(120.0, position_hold_data); // [s]
      if (!position_hold_data.did_break) {
        std::cout << "Exit as no force could be detected." << std::endl;
        std::exit(0);
      }
      robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));
    }

    SelectionMethod current_method;
    if (config.mode == Mode::Measure) {
      current_method = epoch.getSelectionMethod();
    } else {
      current_method = epoch.getSelectionMethodPerform(count_failed_grasps_since_last_success);
    }

    if (config.mode == Mode::Evaluate) {
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1000)); // [ms]
    }

    Eigen::Affine3d image_frame = robot->currentPose(config.camera_frame);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto images = takeImages();

    Grasp new_grasp;
    if (config.use_cpp_inference) {
      InferenceResult inference_result = grasp_inference->infer(images[0], current_method);
      // InferenceResult inference_result = grasp_inference->inferDual(images[0], images[1], current_method);
      new_grasp = grasp_converter.convert(inference_result, images[0]);
    } else {
      new_grasp = static_cast<Grasp>(grasp_io->infer(images[0], current_method));
    }
    new_grasp.method = current_method;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Image + inference detection time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    for (const DepthImage image: images) {
      auto image_future = grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-v");
      image_future.wait();
    }
    grasp_io->saveAttempt(grasp_result.id, new_grasp);

    // Push objects
    while (config.push_objects && new_grasp.prob < config.grasp_push_threshold && new_grasp.found == 1) {
      auto push_result = PushResult();

      // Actual push
      SelectionMethod current_push_method;
      if (config.mode == Mode::Measure) {
        current_push_method = epoch.getSelectionMethod();
      } else {
        current_push_method = epoch.getSelectionMethodPerform(count_failed_grasps_since_last_success);
      }

      InferenceResult push_inference_result = push_inference->infer(images[0], current_push_method);

      Push new_push = push_converter.convert(push_inference_result, images[0]);
      if (new_push.prob < config.push_empty_threshold) {
        break;
      }

      push_result.probstart = getMaxGraspProbAroundPose(images, new_push.pose);

      for (const DepthImage image: images) {
        auto image_future = push_io->uploadImage(push_result.id, image, image.suffix + "-v");
        image_future.wait();
      }
      push_io->saveAttempt(push_result.id, new_push);

      if (config.set_zero_reward) new_push.found = -1;
      push_result.push = new_push;

      RobotPose start_pose = new_push.pose;
      if (start_pose.a < -1.5) {
        start_pose.a += M_PI;
        new_push.reverse = true;
      } else if (start_pose.a > 1.5) {
        start_pose.a -= M_PI;
        new_push.reverse = true;
      }

      std::future<bool> gripper_success_future = gripper->moveAsync(0.0);

      Eigen::Affine3d new_grasp_frame = image_frame * start_pose.toAffine();
      Eigen::Affine3d new_grasp_approach_frame = Affine(0, 0, config.approach_distance_from_pose) * new_grasp_frame;

      robot->moveCartesian(config.gripper_frame, new_grasp_approach_frame);

      gripper_success_future.wait_for(std::chrono::seconds(10)); // [s]

      auto approach_motion_data = MotionData().withDynamics(config.approach_dynamics_rel).withZForceCondition(6.0); // [N]
      robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, -config.approach_distance_from_pose), approach_motion_data);
      if (approach_motion_data.did_break) {
        robot->recoverFromErrors();
      }
      robot->recoverFromErrors();

      push_result.start = RobotPose(image_frame.inverse() * robot->currentPose(config.gripper_frame));

      // Actual push movement
      Eigen::Affine3d push_transformation = Affine(config.push_distance, 0.0, 0.0);
      if (new_push.reverse) {
        push_transformation = push_transformation.inverse();
      }

      auto push_motion_data = MotionData().withDynamics(config.approach_dynamics_rel).withXYForceCondition(14.0); // [N]
      robot->moveRelativeCartesian(config.gripper_frame, push_transformation, push_motion_data);
      robot->recoverFromErrors();

      push_result.end = RobotPose(image_frame.inverse() * robot->currentPose(config.gripper_frame));

      auto up_motion_data = MotionData().withDynamics(0.7).withZForceCondition(15.0); // [N]
      bool move_up_execution_successful = robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, config.approach_distance_from_pose), up_motion_data);
      if (up_motion_data.did_break || !move_up_execution_successful) {
        robot->recoverFromErrors();
        robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, config.approach_distance_from_pose));
      }

      robot->moveRelativeCartesian(config.gripper_frame, relativeSafePose(robot->currentPose(config.gripper_frame)));

      robot->moveCartesian(config.camera_frame, image_frame);

      auto images_after = takeImages();
      for (const DepthImage image: images_after) {
        auto image_future = push_io->uploadImage(push_result.id, image, image.suffix + "-after");
      }
      push_result.probend = getMaxGraspProbAroundPose(images_after, new_push.pose);
      push_result.reward = push_result.probend - push_result.probstart;


      if (config.mode == Mode::Measure) {
        current_method = epoch.getSelectionMethod();
      } else {
        current_method = epoch.getSelectionMethodPerform(count_failed_grasps_since_last_success);
      }
      InferenceResult inference_result = grasp_inference->infer(images_after[0], current_method);
      new_grasp = grasp_converter.convert(inference_result, images_after[0]);
      new_grasp.method = current_method;

      for (const DepthImage image: images_after) {
        auto image_future = grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-v");
        image_future.wait();
      }
      if (push_result.save && ros::ok() && config.mode == Mode::Measure) {
        push_io->saveResult(push_result);
      }
      images = images_after;

      grasp_io->saveAttempt(grasp_result.id, new_grasp);
    }

    if (config.set_zero_reward) new_grasp.found = -1;
    grasp_result.grasp = new_grasp;

    // Check if bin is empty
    bool bin_empty = epoch.selectionMethodShouldBeHigh(current_method) && new_grasp.prob < config.change_bin_at_max_probability;
    if (bin_empty) {
      std::cout << "Bin empty!" << std::endl;
    }

    // found = 1: ok, 0: error, -1: not found
    if (new_grasp.found == 0 || bin_empty) {
      std::cout << "Pose ignored." << std::endl;
      grasp_result.save = false;
    } else if (new_grasp.found == 1) {
      Eigen::Affine3d new_grasp_frame = image_frame * new_grasp.pose.toAffine();
      Eigen::Affine3d new_grasp_approach_frame = Affine(0, 0, config.approach_distance_from_pose) * new_grasp_frame;

      if (config.mode == Mode::Measure && config.take_direct_images) {
        robot->moveCartesian(config.camera_frame, Affine(0, 0, 0.308) * new_grasp_frame);
        for (const DepthImage image: takeImages()) {
          grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-direct");
        }
      }

      std::future<bool> gripper_success_future = gripper->moveAsync(new_grasp.pose.d);

      robot->moveCartesian(config.gripper_frame, new_grasp_approach_frame);
      if (config.mode == Mode::Measure && config.take_direct_images) {
        for (const DepthImage image: takeImages()) {
          grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-pre");
        }
      }
      auto future_status = gripper_success_future.wait_for(std::chrono::seconds(10)); // [s]

      // Try to get rid of gripper errors
      if (future_status == std::future_status::timeout) {
        gripper->homing();
        gripper->move(new_grasp.pose.d);
      }

      if (config.check_grasp_second_time && config.use_cpp_inference) {
        prob_mutex.lock();
        if (current_prob < 0.01 * new_grasp.prob) {
          std::cout << "Break at prob: " << current_prob << std::endl;
          prob_mutex.unlock();
          return;
        }
        prob_mutex.unlock();
      }

      if (config.adjust_grasp_second_time && config.use_cpp_inference) {
        for (int i = 0; i < 2; i++) {
          takeImages();
        }

        auto adjusted_images = takeImages();

        RobotPose adjusted_pose;
        adjusted_pose.x = config.camera_frame.translation()(0);
        adjusted_pose.y = config.camera_frame.translation()(1);
        adjusted_pose.a = 0.0;
        adjusted_pose.d = new_grasp.pose.d;

        InferenceResult inference_result = grasp_inference->inferAroundPoseDiscountedTime(adjusted_images[0], adjusted_pose, SelectionMethod::Max);
        Grasp adjusted_grasp = grasp_converter.convert(inference_result, adjusted_images[0]);

        if (adjusted_grasp.prob > new_grasp.prob * 0.9) {
          Eigen::Affine3d adjusted_image_frame = robot->currentPose(config.camera_frame);
          Eigen::Affine3d adjusted_grasp_frame = adjusted_image_frame * adjusted_grasp.pose.toAffine();
          Eigen::Affine3d adjusted_grasp_approach_frame = Affine(0, 0, config.approach_distance_from_pose) * adjusted_grasp_frame;

          std::future<bool> adjusted_gripper_success_future = gripper->moveAsync(adjusted_grasp.pose.d);

          robot->moveCartesian(config.gripper_frame, adjusted_grasp_approach_frame);

          adjusted_gripper_success_future.wait_for(std::chrono::seconds(10)); // [s]
        }
      }

      auto approach_motion_data = MotionData().withDynamics(config.approach_dynamics_rel).withZForceCondition(10.0); // [N]
      robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, -config.approach_distance_from_pose), approach_motion_data);
      if (approach_motion_data.did_break) {
        robot->recoverFromErrors();
        robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, 0.001), MotionData().withDynamics(0.4));
        grasp_result.collision = true;
      }

      grasp_result.final = RobotPose(image_frame.inverse() * robot->currentPose(config.gripper_frame));

      bool first_grasp_succesfull = gripper->clamp();
      if (first_grasp_succesfull) {
        std::cout << "Grasp successfull at first." << std::endl;
        robot->recoverFromErrors();

        auto up_motion_data = MotionData().withDynamics(0.7).withZForceCondition(15.0); // [N]
        bool move_up_execution_successful = robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, config.approach_distance_from_pose), up_motion_data);
        if (up_motion_data.did_break || !move_up_execution_successful) {
          gripper->release(gripper->width() + 0.002); // [m]

          robot->recoverFromErrors();
          robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, config.approach_distance_from_pose));
          robot->moveRelativeCartesian(config.gripper_frame, relativeSafePose(robot->currentPose(config.gripper_frame)));
          gripper->move(gripper->max_width);
        } else {
          if (config.take_after_image && !config.release_in_other_bin) {
            robot->moveCartesian(config.camera_frame, image_frame);
            for (const DepthImage image: takeImages()) {
              grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-after");
            }
          }

          Eigen::Affine3d possible_random_affine = Eigen::Affine3d::Identity();
          if (config.random_pose_before_release) {
            possible_random_affine = getRandomAffine(config.max_random_affine_before_release);
          }

          robot->recoverFromErrors();
          if (config.release_in_other_bin) {
            if (config.release_as_fast_as_possible) {
              auto change_bin_motion_data = MotionData();
              robot->moveWaypointsCartesian(config.gripper_frame, {
                Waypoint(Affine(0.480, -0.05, 0.180, M_PI / 2), Waypoint::ReferenceType::ABSOLUTE)
              }, change_bin_motion_data);
            } else {
              auto change_bin_motion_data = MotionData();
              robot->moveWaypointsCartesian(config.gripper_frame, {
                Waypoint(Affine(0.480, 0.0, 0.240, M_PI / 2), Waypoint::ReferenceType::ABSOLUTE),
                Waypoint(getBinReleaseAffine(getNextBin(current_bin)) * possible_random_affine, Waypoint::ReferenceType::ABSOLUTE)
              }, change_bin_motion_data);
            }   
          } else {
            robot->moveCartesian(config.gripper_frame, getBinReleaseAffine(current_bin) * possible_random_affine);
          }

          if (gripper->is_grasping()) {
            grasp_result.reward = 1.0;
            grasp_result.final.d = gripper->width();
          }

          if (config.mode == Mode::Perform) {
            gripper->release(grasp_result.final.d + 0.008); // [m], open just a little bit more
          } else {
            gripper->release(new_grasp.pose.d + 0.002); // [m], open like beginning
            robot->moveRelativeCartesian(config.gripper_frame, relativeSafePose(robot->currentPose(config.gripper_frame)));
          }

          if (config.mode == Mode::Measure && config.take_after_image && config.release_in_other_bin) {
            robot->moveCartesian(config.camera_frame, image_frame);
            for (const DepthImage image: takeImages()) {
              grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-after");
            }
          }
        }
      } else {
        gripper->release(gripper->width() + 0.002); // [m]

        robot->recoverFromErrors();
        robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, config.approach_distance_from_pose));

        if (config.mode == Mode::Measure && config.take_after_image) {
          robot->moveCartesian(config.camera_frame, image_frame);
          for (const DepthImage image: takeImages()) {
             grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-after");
          }
        }
      }

      count_episodes_in_current_bin += 1;
      count_all_episodes += 1;
    } else if (new_grasp.found == -1) {
      std::cout << "Pose not found." << std::endl;
      grasp_result.collision = true;

      if (config.take_after_image) {
        robot->moveCartesian(config.camera_frame, image_frame);
        for (const DepthImage image: takeImages()) {
          grasp_io->uploadImage(grasp_result.id, image, image.suffix + "-after");
        }
      }
    }

    if (grasp_result.reward > 0.0) {
      count_failed_grasps_since_last_success = 0;
      count_successfull_grasps += 1;
      count_successfull_grasps_in_current_bin += 1;
    } else {
      count_failed_grasps_since_last_success += 1;
    }

    if (grasp_result.save && ros::ok() && config.mode == Mode::Measure) {
      grasp_io->saveResult(grasp_result);
    }

    std::cout << "Episode ID: " << grasp_result.id << std::endl;
    std::cout << "Episodes (done): " << count_all_episodes << std::endl;
    std::cout << "Episodes in epoch (success / done / total): " << count_successfull_grasps << " / " << current_episode_in_epoch + 1 << " / " << epoch.number_cycles << std::endl;
    std::cout << "Last success: " << count_failed_grasps_since_last_success << " cycles ago." << std::endl;

    // Retrain network
    if (config.train_model && current_episode_in_epoch % config.train_model_every_number_cycles == 0 && current_episode_in_epoch > 0 && config.mode == Mode::Measure) {
      grasp_io->trainModel();
    }

    // Change bin
    if (
      (config.mode == Mode::Evaluate && count_successfull_grasps_in_current_bin == config.change_bin_at_number_of_success_grasps)
      || (config.mode != Mode::Evaluate && (count_failed_grasps_since_last_success >= config.change_bin_at_number_of_failed_grasps || bin_empty) && config.change_bins)
    ) {
      if (config.mode == Mode::Evaluate) {
        grasp_io->saveEvalResult(config.evaluation_result, count_episodes_in_current_bin);
      }

      count_episodes_in_current_bin = 0;
      count_successfull_grasps_in_current_bin = 0;
      count_failed_grasps_since_last_success = 0;

      current_bin = getNextBin(current_bin);
      std::cout << "Switch to other bin." << std::endl;

      if (config.mode != Mode::Perform && config.home_gripper) {
        gripper->homing();
      }
    }
  }
};


int main(int argc, char *argv[]) {
  ros::init(argc, argv, "bin_picking");

  Grasping agent {ros::package::getPath("bin_picking") + "/config.yaml"};
  agent.reset();
  agent.run();
  agent.reset();

  ros::shutdown();
  return 0;
}
