#pragma once

#include <fstream>

#include <cpr/cpr.h>
#include <opencv2/opencv.hpp>
#include <cereal/archives/json.hpp>

#include <bin_picking/inference_result.hpp>
#include <bin_picking/parameter.hpp>


class Io {
  const std::string database_url;
  const std::string learning_url;
  const std::string database;

public:
  Io(const std::string& database_url, const std::string& learning_url, const std::string& database): database_url(database_url), learning_url(learning_url), database(database) { }

  std::future<cpr::Response> uploadImage(const std::string& id, const DepthImage& depth_image, const std::string& suffix) {
    std::vector<unsigned char> buf;
    cv::imencode(".png", depth_image.image, buf);
    return cpr::PostAsync(cpr::Url{database_url + "upload-image"},
      cpr::Parameters{{"database", database}, {"id", id}, {"suffix", suffix}},
      cpr::Body{ std::string{buf.begin(), buf.end()} },
      cpr::Header{{"Content-Type", "text/plain"}}
    );
  }

  template<class T>
  bool saveResult(const T result) const {
    std::stringstream ss;
    {
      cereal::JSONOutputArchive oarchive(ss);
      oarchive(CEREAL_NVP(result));
    }

    auto r = cpr::Post(cpr::Url{database_url + "new-result"}, cpr::Parameters{{"database", database}}, cpr::Payload{{"json", ss.str()}});
    return (r.status_code == 200);
  }

  template<class T>
  bool saveAttempt(const std::string& id, const T action) const {
    std::stringstream ss;
    {
      cereal::JSONOutputArchive oarchive(ss);
      oarchive(CEREAL_NVP(id), CEREAL_NVP(action));
    }

    auto r = cpr::Post(cpr::Url{database_url + "new-attempt"}, cpr::Parameters{{"database", database}}, cpr::Payload{{"json", ss.str()}});
    return (r.status_code == 200);
  }

  Action infer(const DepthImage& depth_image, const SelectionMethod& selection_method) const {
    std::vector<unsigned char> buf;
    cv::imencode(".png", depth_image.image, buf);
    auto r = cpr::Post(cpr::Url{learning_url + "infer"},
      cpr::Parameters{{"method", selectionMethodName(selection_method)}},
      cpr::Body{ std::string{buf.begin(), buf.end()} },
      cpr::Header{{"Content-Type", "text/plain"}}
    );

    if (r.text.empty()) {
      std::cout << "Could not infer from learning server." << std::endl;
      std::exit(1);
    }

    std::stringstream ss;
    ss.str(r.text);
    cereal::JSONInputArchive iarchive(ss);

    Action action;
    iarchive(CEREAL_NVP(action));
    return action;
  }

  bool trainModel() const {
    auto r = cpr::Post(cpr::Url{learning_url + "train-model"});
    return (r.status_code == 200);
  }

  bool restoreModel() const {
    auto r = cpr::Post(cpr::Url{learning_url + "restore-model"});
    return (r.status_code == 200);
  }

  void saveEvalResult(std::string file_path, int number_tries) const {
    std::ofstream outfile;

    outfile.open(file_path, std::ios_base::app);
    outfile << number_tries << std::endl;
    outfile.close();
  }
};
