#include "stereovision/StereoVision.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace stereovision {
namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

cv::Matx33d parseCameraMatrix(const std::string& text) {
    static const std::regex number_regex(R"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)");
    std::sregex_iterator it(text.begin(), text.end(), number_regex);
    const std::sregex_iterator end;
    std::array<double, 9> values{};
    int idx = 0;
    for (; it != end && idx < 9; ++it, ++idx) {
        values[static_cast<size_t>(idx)] = std::stod(it->str());
    }
    if (idx < 9) {
        throw std::runtime_error("Invalid camera matrix in calib.txt (need 9 numbers)");
    }
    return {values[0], values[1], values[2], values[3], values[4], values[5],
            values[6], values[7], values[8]};
}

float percentile95(const cv::Mat& src, const cv::Mat& mask) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < src.rows; ++y) {
        const float* row_ptr = src.ptr<float>(y);
        const uchar* mask_ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (mask_ptr[x] != 0) {
                values.push_back(row_ptr[x]);
            }
        }
    }
    if (values.empty()) {
        return 1.0F;
    }
    const size_t kth = (95U * (values.size() - 1U)) / 100U;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(kth), values.end());
    return std::max(1.0F, values[kth]);
}

}  // namespace

StereoVision::StereoVision(const Config& config) : config_(config) {
    loadCalibration();
    initStereoMatcher();
}

void StereoVision::loadCalibration() {
    const std::filesystem::path dataset(config_.dataset_path_);
    const std::filesystem::path calib_path = dataset / "calib.txt";

    if (!std::filesystem::is_directory(dataset))
        throw std::invalid_argument("Dataset path not found: " + config_.dataset_path_);
    if (!std::filesystem::is_regular_file(calib_path))
        throw std::invalid_argument("calib.txt not found under: " + config_.dataset_path_);

    std::ifstream in(calib_path);
    if (!in)
        throw std::runtime_error("Failed to open calib.txt");

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#')
            continue;

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));

        if (key == "cam0") {
            cv::Matx33d cam0 = parseCameraMatrix(value);
            calib_.focal_lenght_ = cam0(0, 0);
            calib_.cx = cam0(0, 2);
            calib_.cy = cam0(1, 2);
        } else if (key == "baseline") {
            calib_.baseline = std::stod(value) / 1000.0;
        } else if (key == "doffs") {
            calib_.doffs = std::stod(value);
        }
    }
}

void StereoVision::initStereoMatcher() {
    int num_disparities = config_.num_disparities_;
    int block_size = config_.block_size_;

    num_disparities = ((num_disparities + 15) / 16) * 16;
    block_size = std::max(3, block_size);
    if (block_size % 2 == 0) {
        ++block_size;
    }

    const char* method_label = "SGBM";

    if (config_.stereo_method_ == StereoMethod::BM) {
        method_label = "BM";
        auto bm = cv::StereoBM::create(num_disparities, block_size);
        bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
        bm->setPreFilterSize(9);
        bm->setPreFilterCap(31);
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(15);
        bm->setSpeckleRange(32);
        bm->setSpeckleWindowSize(100);
        stereo_ = bm;
    } else {
        auto sgbm = cv::StereoSGBM::create(0, num_disparities, block_size);
        sgbm->setP1(8 * 3 * block_size * block_size);
        sgbm->setP2(32 * 3 * block_size * block_size);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(100);
        sgbm->setSpeckleRange(32);
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        stereo_ = sgbm;
    }

    std::cout << "Initialized " << method_label << " with num_disparities=" << num_disparities
              << ", block_size=" << block_size << '\n';
}

void StereoVision::loadImages() {
    const std::filesystem::path dataset(config_.dataset_path_);
    std::filesystem::path left_path = dataset / "im0.png";
    std::filesystem::path right_path = dataset / "im1.png";

    if (!std::filesystem::exists(left_path)) {
        left_path = dataset / "im2.png";
        right_path = dataset / "im6.png";
    }
    if (!std::filesystem::exists(left_path) || !std::filesystem::exists(right_path)) {
        throw std::runtime_error("No stereo pair found in " + config_.dataset_path_);
    }

    image_.left_img = cv::imread(left_path.string(), cv::IMREAD_COLOR);
    image_.right_img = cv::imread(right_path.string(), cv::IMREAD_COLOR);
    if (image_.left_img.empty() || image_.right_img.empty()) {
        throw std::runtime_error("Failed to load images");
    }
    if (image_.left_img.size() != image_.right_img.size()) {
        throw std::runtime_error("Left/right image sizes do not match");
    }

    if (config_.use_gray_) {
        cv::cvtColor(image_.left_img, image_.left_img_gray_, cv::COLOR_BGR2GRAY);
        cv::cvtColor(image_.right_img, image_.right_img_gray_, cv::COLOR_BGR2GRAY);
    } else {
        image_.left_img_gray_ = image_.left_img;
        image_.right_img_gray_ = image_.right_img;
    }

    std::cout << "Loaded images: " << left_path.filename().string() << ", " << right_path.filename().string()
              << " (" << image_.left_img.cols << "x" << image_.left_img.rows << ")\n";
}

void StereoVision::compute_disparity() {
    if (image_.left_img_gray_.empty() || image_.right_img_gray_.empty()) {
        throw std::runtime_error("Images not loaded. Call loadImages() first.");
    }
    if (stereo_.empty()) {
        throw std::runtime_error("Stereo matcher not initialized.");
    }

    cv::Mat raw;
    stereo_->compute(image_.left_img_gray_, image_.right_img_gray_, raw);

    raw.convertTo(disparity_map, CV_32F, 1.0 / 16.0);
    cv::max(disparity_map, 0.0F, disparity_map);

    if (disparity_map.empty())
        throw std::runtime_error("Disparity map was empty");

    if (config_.apply_median_filter_) {
        cv::medianBlur(disparity_map, disparity_map, 5);
    }

    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(disparity_map, &min_v, &max_v);
    std::cout << "Disparity computed: range [" << min_v << ", " << max_v << "]\n";
}

void StereoVision::compute_depth() {
    if (disparity_map.empty()) {
        throw std::runtime_error("Disparity map not computed. Call compute_disparity() first.");
    }

    depth_map = cv::Mat::zeros(disparity_map.size(), CV_32F);
    const cv::Mat valid_mask = disparity_map > 0.0F;
    const cv::Mat denom = disparity_map + static_cast<float>(calib_.doffs);
    const cv::Mat depth_candidate =
        static_cast<float>(calib_.focal_lenght_ * calib_.baseline) / denom;
    depth_candidate.copyTo(depth_map, valid_mask);

    const float max_depth = static_cast<float>(config_.max_depth_meters_);
    cv::threshold(depth_map, depth_map, max_depth, max_depth, cv::THRESH_TRUNC);

    double max_depth_val = 0.0;
    cv::minMaxLoc(depth_map, nullptr, &max_depth_val);
    const int valid_pixels = cv::countNonZero(valid_mask);
    if (valid_pixels > 0) {
        double min_positive = 0.0;
        cv::minMaxLoc(depth_map, &min_positive, nullptr, nullptr, nullptr, valid_mask);
        std::cout << "Depth computed: range [" << min_positive << "m, " << max_depth_val << "m]\n";
    } else {
        std::cout << "Depth computed: no valid pixels\n";
    }
}

void StereoVision::visualize_results(bool saveFig) {
    if (image_.left_img.empty() || disparity_map.empty() || depth_map.empty()) {
        throw std::runtime_error("No results to visualize");
    }

    const std::string& output_dir = config_.output_dir_;
    if (saveFig) {
        std::filesystem::create_directories(output_dir);
    }

    cv::Mat disp_vis;
    const double max_disp = cv::norm(disparity_map, cv::NORM_INF);
    if (max_disp > 0.0) {
        disparity_map.convertTo(disp_vis, CV_8U, 255.0 / max_disp);
    } else {
        disp_vis = cv::Mat::zeros(disparity_map.size(), CV_8U);
    }
    cv::Mat disp_color;
    cv::applyColorMap(disp_vis, disp_color, cv::COLORMAP_JET);

    cv::Mat depth_vis = depth_map.clone();
    const cv::Mat positive_mask = depth_vis > 0.0F;
    cv::Mat depth_color;
    if (cv::countNonZero(positive_mask) > 0) {
        const float p95 = percentile95(depth_vis, positive_mask);
        cv::threshold(depth_vis, depth_vis, p95, p95, cv::THRESH_TRUNC);
        depth_vis.convertTo(depth_vis, CV_8U, 255.0F / p95);
        cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_VIRIDIS);
    } else {
        depth_vis = cv::Mat::zeros(depth_map.size(), CV_8U);
        cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_VIRIDIS);
    }

    int h = image_.left_img.rows;
    int w = image_.left_img.cols;
    cv::Mat info_panel = cv::Mat::zeros(h, w, CV_8UC3);

    const int text_thickness = 2;
    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const std::string method_str =
        (config_.stereo_method_ == StereoMethod::BM) ? "BM" : "SGBM";

    cv::putText(info_panel,
                "Focal: " + std::to_string(static_cast<int>(calib_.focal_lenght_)) + "px",
                {10, 30}, font, 0.7, {255, 255, 255}, text_thickness);
    cv::putText(info_panel,
                "Baseline: " + std::to_string(calib_.baseline * 1000.0) + "mm",
                {10, 60}, font, 0.7, {255, 255, 255}, text_thickness);
    cv::putText(info_panel, "Method: " + method_str, {10, 90}, font, 0.7, {255, 255, 255},
                text_thickness);
    cv::putText(info_panel, "Disparities: " + std::to_string(config_.num_disparities_), {10, 120},
                font, 0.7, {255, 255, 255}, text_thickness);

    cv::Mat left_resized = image_.left_img;
    cv::Mat disp_resized = disp_color;
    cv::Mat depth_resized = depth_color;
    const float disp_scale = static_cast<float>(config_.display_scale_);
    if (disp_scale != 1.0F) {
        const int new_w = static_cast<int>(static_cast<float>(w) * disp_scale);
        const int new_h = static_cast<int>(static_cast<float>(h) * disp_scale);
        cv::resize(image_.left_img, left_resized, {new_w, new_h});
        cv::resize(disp_color, disp_resized, {new_w, new_h});
        cv::resize(depth_color, depth_resized, {new_w, new_h});
        cv::resize(info_panel, info_panel, {new_w, new_h});
        w = new_w;
        h = new_h;
    }

    cv::Mat top_row;
    cv::Mat bottom_row;
    cv::hconcat(std::vector<cv::Mat>{left_resized, disp_resized}, top_row);
    cv::hconcat(std::vector<cv::Mat>{depth_resized, info_panel}, bottom_row);
    cv::Mat visualization;
    cv::vconcat(std::vector<cv::Mat>{top_row, bottom_row}, visualization);

    cv::putText(visualization, "Left Image", {10, 30}, font, 1.0, {255, 255, 255}, 2);
    cv::putText(visualization, "Disparity Map", {w + 10, 30}, font, 1.0, {255, 255, 255}, 2);
    cv::putText(visualization, "Depth Map", {10, h + 30}, font, 1.0, {255, 255, 255}, 2);
    cv::putText(visualization, "Parameters", {w + 10, h + 30}, font, 1.0, {255, 255, 255}, 2);

    cv::imshow("Stereo Vision Pipeline Results", visualization);
    cv::waitKey(0);
    cv::destroyAllWindows();

    if (saveFig) {
        const std::string scene = std::filesystem::path(config_.dataset_path_).filename().string();
        const auto output_path = std::filesystem::path(output_dir) / ("result_" + scene + ".png");
        cv::imwrite(output_path.string(), visualization);
        std::cout << "Visualization saved to " << output_path.string() << '\n';
    }
}

void StereoVision::save_disparity_map() {
    if (disparity_map.empty() || depth_map.empty()) {
        throw std::runtime_error("No disparity/depth data to save");
    }

    const std::string& output_dir = config_.output_dir_;
    std::filesystem::create_directories(output_dir);
    const std::string scene_name = std::filesystem::path(config_.dataset_path_).filename().string();

    double max_disp = 0.0;
    cv::minMaxLoc(disparity_map, nullptr, &max_disp);
    cv::Mat disp_norm;
    if (max_disp > 0.0) {
        disparity_map.convertTo(disp_norm, CV_16U, 65535.0 / max_disp);
    } else {
        disp_norm = cv::Mat::zeros(disparity_map.size(), CV_16U);
    }
    cv::imwrite((std::filesystem::path(output_dir) / (scene_name + "_disp.png")).string(), disp_norm);

    cv::FileStorage disp_fs((std::filesystem::path(output_dir) / (scene_name + "_disp.yml")).string(),
                            cv::FileStorage::WRITE);
    disp_fs << "disparity" << disparity_map;
    disp_fs.release();

    cv::FileStorage depth_fs((std::filesystem::path(output_dir) / (scene_name + "_depth.yml")).string(),
                             cv::FileStorage::WRITE);
    depth_fs << "depth" << depth_map;
    depth_fs.release();

    std::cout << "Results saved to " << output_dir << "/\n";
}

}  // namespace stereovision
