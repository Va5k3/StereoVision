#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "stereovision/Config.hpp"

namespace stereovision {

struct Calibration {
    double focal_lenght_;
    double cx, cy, baseline, doffs;
};

struct Images {
    cv::Mat left_img;
    cv::Mat right_img;
    cv::Mat left_img_gray_;
    cv::Mat right_img_gray_;
};

class StereoVision {
    Config config_;
    Calibration calib_{};
    Images image_;
    cv::Mat disparity_map;
    cv::Mat depth_map;

    void loadCalibration();
    void initStereoMatcher();

    cv::Ptr<cv::StereoMatcher> stereo_;

public:
    explicit StereoVision(const Config& config);
    void loadImages();
    void compute_disparity();
    void compute_depth();
    void visualize_results(bool saveFig = false);
    void save_disparity_map();
};

}  // namespace stereovision
