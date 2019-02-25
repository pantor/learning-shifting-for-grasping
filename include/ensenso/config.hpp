#pragma once

struct EnsensoConfig {
  std::string id;
  double pixel_size;
  double min_depth;
  double max_depth;

  // Default settings
  bool auto_exposure {true};
  double exposure_time {0}; // [ms]
  int target_brightness {200};

  bool auto_gain {true};
  double gain {0};

  bool projector {true};
  bool front_light {false};

  std::string stereo_matching_method {"SgmAligned"};

  bool use_open_gl {true};
  bool use_cuda {true};
  int cuda_device {1};
};
