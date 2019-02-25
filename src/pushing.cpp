#include <bin_picking/agent.hpp>
#include <bin_picking/depth_image.hpp>

#include <pushing/push.hpp>
#include <pushing/push_converter.hpp>


class Pushing: public Agent {
  PushConverter push_converter;

  double getMaxGraspProb(const std::vector<DepthImage>& images) {
    return grasp_inference->infer(images[0], SelectionMethod::Max).prob;
  }

  double getMaxGraspProbAroundPose(const std::vector<DepthImage>& images, const RobotPose& pose) {
    return grasp_inference->inferAroundPose(images[0], pose, SelectionMethod::Max).prob;
  }

public:
  explicit Pushing(const std::string& config_filename): Agent(config_filename) {
    push_converter.bin = config.bin_data;
    push_converter.pose_should_be_inside_bin = true;

    push_inference->setSizeOriginalCropped({240, 240});
    push_inference->a_space = Inference::linspace(-3.0, 3.0, 26);
    push_inference->number_actions = 2;

    if (config.use_cpp_inference) {
      grasp_inference = std::make_shared<Inference>(config.secondary_model + "-sm", config.bin_data);
    }
  }

  void reset() {
    robot->recoverFromErrors();
    gripper->stop();

    robot->moveRelativeCartesian(config.gripper_frame, relativeSafePose(robot->currentPose(config.gripper_frame)), MotionData().withDynamics(0.3));
    robot->moveJoints(config.bin_joint_values.at(current_bin), MotionData().withDynamics(0.7));
    robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));

    gripper->move(0.0);
  }

  void runEpisode(Epoch epoch, int current_episode_in_epoch) {
    auto push_result = PushResult();

    // Move to camera position, but with elbow configuration
    robot->recoverFromErrors();
    robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));
    if (config.mode != Mode::Perform) {
      robot->moveJoints(config.bin_joint_values.at(current_bin));
      robot->moveCartesian(config.camera_frame, getBinCameraAffine(current_bin));
    }

    Eigen::Affine3d image_frame = robot->currentPose(config.camera_frame);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto images = takeImages();

    double max_grasp_prob = getMaxGraspProb(images);

    SelectionMethod current_method;
    if (config.mode == Mode::Measure) {
      if (epoch.selection_method_primary == SelectionMethod::Top5 || epoch.selection_method_primary == SelectionMethod::Bottom5) {
        epoch.selection_method_primary = (max_grasp_prob < 0.7) ? SelectionMethod::Top5 : SelectionMethod::Bottom5;
      }
      current_method = epoch.getSelectionMethod();

    } else {
      current_method = epoch.getSelectionMethodPerform(0.0);
    }

    std::cout << "Get push inference" << std::endl;
    Push new_push;
    if (config.use_cpp_inference) {
      InferenceResult inference_result = push_inference->infer(images[0], current_method);
      new_push = push_converter.convert(inference_result, images[0]);
    } else {
      new_push = static_cast<Push>(push_io->infer(images[0], current_method));
    }

    push_result.probstart = getMaxGraspProbAroundPose(images, new_push.pose);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Image + inference detection time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    for (const DepthImage image: images) {
      auto image_future = push_io->uploadImage(push_result.id, image, image.suffix + "-v");
      image_future.wait();
    }
    push_io->saveAttempt(push_result.id, new_push);

    if (config.set_zero_reward) new_push.found = -1;
    push_result.push = new_push;

    if (new_push.found == 0) {
      std::cout << "Pose ignored." << std::endl;
      push_result.save = false;

    } else if (new_push.found == 1) {
      RobotPose start_pose = new_push.pose;
      if (start_pose.a < -1.5) {
        start_pose.a += M_PI;
        new_push.reverse = true;
      } else if (start_pose.a > 1.5) {
        start_pose.a -= M_PI;
        new_push.reverse = true;
      }

      Eigen::Affine3d new_grasp_frame = image_frame * start_pose.toAffine();
      Eigen::Affine3d new_grasp_approach_frame = Affine(0, 0, config.approach_distance_from_pose) * new_grasp_frame;

      robot->moveCartesian(config.gripper_frame, new_grasp_approach_frame);

      auto approach_motion_data = MotionData().withDynamics(config.approach_dynamics_rel).withZForceCondition(6.0); // [N]
      robot->moveRelativeCartesian(config.gripper_frame, Affine(0.0, 0.0, -config.approach_distance_from_pose), approach_motion_data);
      if (approach_motion_data.did_break) {
        robot->recoverFromErrors();
        push_result.collision = true;
      }
      robot->recoverFromErrors();

      push_result.start = RobotPose(image_frame.inverse() * robot->currentPose(config.gripper_frame));

      // Actual push movement
      Eigen::Affine3d push_transformation = Affine(config.push_distance, 0.0, 0.0);
      if (new_push.direction == Push::Direction::Left) {
        push_transformation = Affine(0.0, config.push_distance, 0.0);
      }
      if (new_push.reverse) {
        push_transformation = push_transformation.inverse();
      }

      auto push_motion_data = MotionData().withDynamics(config.approach_dynamics_rel).withZForceCondition(10.0); // [N]
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

      count_episodes_in_current_bin += 1;
      count_all_episodes += 1;

    } else if (new_push.found == -1) {
      std::cout << "Pose not found." << std::endl;
      push_result.collision = true;
    }

    if (push_result.save && ros::ok() && config.mode == Mode::Measure) {
      push_io->saveResult(push_result);
    }

    std::cout << "---" << std::endl;
    std::cout << "Episode ID: " << push_result.id << std::endl;
    std::cout << "Episodes (done): " << count_all_episodes << std::endl;

    // Change bin
    if (count_episodes_in_current_bin >= config.change_bin_at_number_of_failed_grasps && config.change_bins) {
      current_bin = getNextBin(current_bin);
      std::cout << "Switch to other bin." << std::endl;
    }
  }
};


int main(int argc, char *argv[]) {
  ros::init(argc, argv, "bin_picking");

  Pushing agent {ros::package::getPath("bin_picking") + "/config.yaml"};
  agent.reset();
  agent.run();
  agent.reset();

  ros::shutdown();
  return 0;
}
