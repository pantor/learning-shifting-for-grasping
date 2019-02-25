#include <ensenso/ensenso.hpp>


Ensenso::Ensenso(EnsensoConfig config): min_depth(config.min_depth), max_depth(config.max_depth) {
  std::cout << "Opening ensenso..." << std::endl;

  try {
    nxLibInitialize(true); // Init nx library

    // Create an object referencing the camera's tree item, for easier access:
    camera = root[itmCameras][itmBySerialNo][config.id];
    if (!camera.exists() || (camera[itmType] != valStereo)) {
      std::cout << "Ensenso not found. Please connect a single stereo camera to your computer.";
      return;
    }

    // Open camera
    NxLibCommand open(cmdOpen);
    open.parameters()[itmCameras] = config.id;
    open.execute();

    factor_depth = 1.0 / (min_depth - max_depth) * (65025. - 1);

    configureDepthCaptureParams(config);
    configureRawCaptureParams(config);

    configureCapture(config);
  } catch (NxLibException& e) {
    printf("An NxLib API error with code %d (%s) occurred while accessing item %s.\n", e.getErrorCode(), e.getErrorText().c_str(), e.getItemPath().c_str());
    if (e.getErrorCode() == NxLibExecutionFailed) printf("/Execute:\n%s\n", NxLibItem(itmExecute).asJson(true).c_str());
    std::exit(-1);
  }

  std::cout << "Ensenso open." << std::endl;
}

Ensenso::~Ensenso() {
  std::cout << "Closing ensenso." << std::endl;

  NxLibCommand (cmdClose).execute();
  nxLibFinalize();

  std::cout << "Ensenso closed." << std::endl;
}

void Ensenso::configureCapture(EnsensoConfig config) {
  camera[itmParameters][itmCapture][itmTargetBrightness] = config.target_brightness;
  camera[itmParameters][itmCapture][itmAutoExposure] = config.auto_exposure;
  camera[itmParameters][itmCapture][itmAutoGain] = config.auto_gain;
  camera[itmParameters][itmCapture][itmExposure] = (double)config.exposure_time;
  camera[itmParameters][itmCapture][itmGain] = (double)config.gain;
  camera[itmParameters][itmCapture][itmProjector] = config.projector;
  camera[itmParameters][itmCapture][itmFrontLight] = config.front_light;

  camera[itmParameters][itmDisparityMap][itmStereoMatching][itmMethod] = config.stereo_matching_method;
  camera[itmParameters][itmDisparityMap][itmMeasurementVolume][itmFar][itmLeftBottom][2] = max_depth * 1.25 * 1000; // [mm]
  camera[itmParameters][itmDisparityMap][itmMeasurementVolume][itmNear][itmLeftBottom][2] = 0.16 * 1000; // [mm]

  root[itmParameters][itmCUDA][itmEnabled] = config.use_cuda;
  root[itmParameters][itmCUDA][itmDevice] = config.cuda_device;

  root[itmParameters][itmRenderPointMap][itmPixelSize] = 0.5; // [mm / px]
  root[itmParameters][itmRenderPointMap][itmScaling] = 0.9;
  root[itmParameters][itmRenderPointMap][itmSize][0] = 752; // [px]
  root[itmParameters][itmRenderPointMap][itmSize][1] = 480; // [px]
  root[itmParameters][itmRenderPointMap][itmUseOpenGL] = config.use_open_gl;

  // Only Ensenso
  // root[itmParameters][itmRenderPointMap][itmViewPose][itmTranslation][0] = 49; // [mm]

  // Ensenso + RealSense
  root[itmParameters][itmRenderPointMap][itmViewPose][itmTranslation][0] = 47; // [mm]
  root[itmParameters][itmRenderPointMap][itmViewPose][itmTranslation][1] = -3; // [mm]


  for (int i = 0; i < 3; i++) {
    NxLibCommand (cmdCapture).execute();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void Ensenso::configureRawCaptureParams(EnsensoConfig config) {
  raw_capture_config = config;
  raw_capture_config.front_light = true;
  raw_capture_config.projector = false;
  raw_capture_config.target_brightness = 100;
}

void Ensenso::configureDepthCaptureParams(EnsensoConfig config) {
  depth_capture_config = config;
}

/* cv::Mat Ensenso::takeDepthImage() {
  try {
    configureCapture(depth_capture_config);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    NxLibCommand (cmdCapture).execute();
    NxLibCommand (cmdComputeDisparityMap).execute();
    NxLibCommand (cmdComputePointMap).execute();

    cv::Mat helper_image = cv::Mat(cv::Size(752, 480), CV_32FC3);

    camera[itmImages][itmPointMap].getBinaryData(helper_image, 0);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Ensenso image time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    cv::Mat depth_image = cv::Mat(helper_image.size(), CV_16UC1);

    for (int i = 0; i < helper_image.rows; i++) {
      for (int j = 0; j < helper_image.cols; j++) {
        const cv::Vec3f px = helper_image.at<cv::Vec3f>(i, j);
        if (!std::isnan(px.val[0])) { // Check if value is valid
          depth_image.at<unsigned short>(i, j) = clampLimits<unsigned short, float>(factor_depth * (px.val[2] / 1000.0 - max_depth)); // [mm]
        } else { // In case of NaN use 0
          depth_image.at<unsigned short>(i, j) = 0; // NaN
        }
      }
    }
    return depth_image;

  } catch (NxLibException& e) {
    printf("An NxLib API error with code %d (%s) occurred while accessing item %s.\n", e.getErrorCode(), e.getErrorText().c_str(), e.getItemPath().c_str());
    if (e.getErrorCode() == NxLibExecutionFailed) printf("/Execute:\n%s\n", NxLibItem(itmExecute).asJson(true).c_str());
    std::exit(-1);
  }
}

cv::Mat Ensenso::takeColorImage() {
  try {
    configureCapture(raw_capture_config);

    NxLibCommand (cmdCapture).execute();
    NxLibCommand (cmdRectifyImages).execute();

    cv::Mat raw_image;
    camera[itmImages][itmRectified][itmLeft].getBinaryData(raw_image, 0);
    return raw_image;

  } catch (NxLibException& e) {
    printf("An NxLib API error with code %d (%s) occurred while accessing item %s.\n", e.getErrorCode(), e.getErrorText().c_str(), e.getItemPath().c_str());
    if (e.getErrorCode() == NxLibExecutionFailed) printf("/Execute:\n%s\n", NxLibItem(itmExecute).asJson(true).c_str());
    std::exit(-1);
  }
} */

cv::Mat Ensenso::takeDepthImage() {
  try {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    configureCapture(depth_capture_config);

    NxLibCommand (cmdCapture).execute();
    NxLibCommand (cmdComputeDisparityMap).execute();
    NxLibCommand (cmdComputePointMap).execute();
    NxLibCommand (cmdRenderPointMap).execute();

    cv::Mat helper_image;
    root[itmImages][itmRenderPointMap].getBinaryData(helper_image, 0);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Ensenso image time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    cv::Mat depth_image = cv::Mat(helper_image.size(), CV_16UC1);

    for (int i = 0; i < helper_image.rows; i++) {
      for (int j = 0; j < helper_image.cols; j++) {
        const cv::Vec3f px = helper_image.at<cv::Vec3f>(i, j);
        if (!std::isnan(px.val[0])) { // Check if value is valid
          depth_image.at<unsigned short>(i, j) = clampLimits<unsigned short, float>(factor_depth * (px.val[2] / 1000.0 - max_depth)); // [mm]
        } else { // In case of NaN use 0
          depth_image.at<unsigned short>(i, j) = 0; // NaN
        }
      }
    }
    return depth_image;

  } catch (NxLibException& e) {
    printf("An NxLib API error with code %d (%s) occurred while accessing item %s.\n", e.getErrorCode(), e.getErrorText().c_str(), e.getItemPath().c_str());
    if (e.getErrorCode() == NxLibExecutionFailed) printf("/Execute:\n%s\n", NxLibItem(itmExecute).asJson(true).c_str());
    std::exit(-1);
  }
}

std::pair<cv::Mat, cv::Mat> Ensenso::takeImages() {
  try {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    configureCapture(raw_capture_config);

    NxLibCommand (cmdCapture).execute();
    NxLibCommand (cmdRectifyImages).execute();

    cv::Mat raw_image;
    camera[itmImages][itmRectified][itmLeft].getBinaryData(raw_image, 0);


    configureCapture(depth_capture_config);

    NxLibCommand (cmdCapture).execute();
    NxLibCommand (cmdComputeDisparityMap).execute();
    NxLibCommand (cmdComputePointMap).execute();
    NxLibCommand (cmdRenderPointMap).execute();

    cv::Mat disparity_image = cv::Mat(raw_image.size(), CV_16SC1);
    camera[itmImages][itmDisparityMap].getBinaryData(disparity_image, 0);

    cv::Mat point_map;
    camera[itmImages][itmPointMap].getBinaryData(point_map, 0);

    cv::Mat rendered_point_map;
    root[itmImages][itmRenderPointMap].getBinaryData(rendered_point_map, 0);


    // Calculate point map with inpaint
    cv::Mat disparity_image_float;
    disparity_image.convertTo(disparity_image_float, CV_32FC1, 1./255);

    cv::Mat mask = cv::Mat::zeros(disparity_image.size(), CV_16UC1);
    mask = (disparity_image == -32768);
    cv::inpaint(disparity_image_float, mask, disparity_image_float, 3, cv::INPAINT_NS);

    disparity_image_float.convertTo(disparity_image, CV_16SC1, 255.0);
    camera[itmImages][itmDisparityMap].setBinaryData(disparity_image);


    NxLibCommand (cmdComputePointMap).execute();

    cv::Mat point_map_inpaint;
    camera[itmImages][itmPointMap].getBinaryData(point_map_inpaint, 0);

    cv::Mat depth_image = cv::Mat(rendered_point_map.size(), CV_16UC1);
    cv::Mat raw_rendered_image = cv::Mat::zeros(point_map.size(), CV_16UC1);
    cv::Mat raw_rendered_depth_image = cv::Mat::zeros(point_map.size(), CV_16UC1);

    for (int i = 0; i < point_map.rows; i++) {
      for (int j = 0; j < point_map.cols; j++) {
        const cv::Vec3f px = point_map.at<cv::Vec3f>(i, j);
        const cv::Vec3f px_rendered = rendered_point_map.at<cv::Vec3f>(i, j);
        const unsigned char px_raw = raw_image.at<unsigned char>(i, j);

        if (!std::isnan(px_rendered.val[0])) { // Check if value is valid
          depth_image.at<unsigned short>(i, j) = clampLimits<unsigned short, float>(factor_depth * (px_rendered.val[2] / 1000.0 - max_depth)); // [mm]
        } else { // In case of NaN use 0
          depth_image.at<unsigned short>(i, j) = 0; // NaN
        }

        if (std::isnan(px.val[0])) continue;

        int i_new = (int)(2 * (px.val[1] - 0) + point_map.rows / 2);
        int j_new = (int)(2 * (px.val[0] - 49) + point_map.cols / 2);
        int new_value = 255 * px_raw;

        if (i_new < 0 || i_new >= point_map.rows) continue;
        if (j_new < 0 || j_new >= point_map.cols) continue;
        
        if (new_value == 0) continue;
        if (raw_rendered_depth_image.at<unsigned short>(i_new, j_new) > px.val[2]) continue;

        raw_rendered_depth_image.at<unsigned short>(i_new, j_new) = px.val[2];

        raw_rendered_image.at<unsigned short>(i_new, j_new) = new_value;
        raw_rendered_image.at<unsigned short>(std::max(i_new - 1, 0), j_new) = new_value;
        raw_rendered_image.at<unsigned short>(i_new, std::max(j_new - 1, 0)) = new_value;
        // raw_rendered_image.at<unsigned short>(std::min(i_new + 1, point_map.rows - 1), std::min(j_new + 1, point_map.cols - 1)) = new_value;
      }
    }

    for (int i = 0; i < point_map_inpaint.rows; i++) {
      for (int j = 0; j < point_map_inpaint.cols; j++) {
        const cv::Vec3f px = point_map_inpaint.at<cv::Vec3f>(i, j);
        const unsigned char px_raw = raw_image.at<unsigned char>(i, j);

        if (std::isnan(px.val[0])) continue;

        int i_new = (int)(2 * (px.val[1] - 0) + point_map_inpaint.rows / 2);
        int j_new = (int)(2 * (px.val[0] - 49) + point_map_inpaint.cols / 2);

        if (raw_rendered_image.at<unsigned short>(i_new, j_new) != 0) continue;

        int new_value = 255 * px_raw;

        if (i_new < 0 || i_new >= point_map_inpaint.rows) continue;
        if (j_new < 0 || j_new >= point_map_inpaint.cols) continue;
        
        if (new_value == 0) continue;
        if (raw_rendered_depth_image.at<unsigned short>(i_new, j_new) > px.val[2]) continue;

        raw_rendered_image.at<unsigned short>(i_new, j_new) = new_value;
        raw_rendered_image.at<unsigned short>(std::max(i_new - 1, 0), j_new) = new_value;
        raw_rendered_image.at<unsigned short>(i_new, std::max(j_new - 1, 0)) = new_value;
        // raw_rendered_image.at<unsigned short>(std::min(i_new + 1, point_map_inpaint.rows - 1), std::min(j_new + 1, point_map_inpaint.cols - 1)) = new_value;
      }
    }

    /* R = np.array([[1, 0, 0, -236.442], [0, 1, 0, -240.596], [0, 0, 0, 589.4], [0, 0, 0.00986395, 2.62967]])

    x = [69.75, 27.0, 200.0, 1] # real world x, y, z, 1

    res = np.matmul(np.linalg.inv(R), x)
    res /= res[3]

    a = [69.75 * R[2, 3] / 200.0 - R[0, 3], 27.0 * R[2, 3] / 200.0 - R[1, 3]]

    assert(res[0] < a[0] * (1 + 1e-6) and res[0] > a[0] * (1 - 1e-6))
    assert(res[1] < a[1] * (1 + 1e-6) and res[1] > a[1] * (1 - 1e-6))

    a */

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Ensenso image time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    return std::make_pair(raw_rendered_image, depth_image);
  } catch (NxLibException& e) {
    printf("An NxLib API error with code %d (%s) occurred while accessing item %s.\n", e.getErrorCode(), e.getErrorText().c_str(), e.getItemPath().c_str());
    if (e.getErrorCode() == NxLibExecutionFailed) printf("/Execute:\n%s\n", NxLibItem(itmExecute).asJson(true).c_str());
    std::exit(-1);
  }
}
