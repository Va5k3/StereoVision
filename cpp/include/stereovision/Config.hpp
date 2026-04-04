#pragma once
#include <string>

namespace stereovision {

enum class StereoMethod {
    BM,
    SGBM
};

struct Config {
    std::string dataset_path_ = "all/data/curule1";
    StereoMethod stereo_method_ = StereoMethod::SGBM;

    int num_disparities_ = 196;
    int block_size_ = 7;

    bool use_gray_ = true;
    bool apply_median_filter_ = true;

    double max_depth_meters_ = 20;

    bool save_results_ = true;
    std::string output_dir_ = "./stereo_output";
    double display_scale_ = 0.4;
};

}  // namespace stereovision
