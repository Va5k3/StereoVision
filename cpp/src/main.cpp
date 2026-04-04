#include "StereoVision.hpp"

#include <iostream>

int main(int argc, char** argv) {
    stereovision::Config config;

    if (argc > 1) {
        config.dataset_path_ = argv[1];
    } else {
        // Ako pokrećeš iz learning/build: .. = learning, ../.. = koren workspace-a
        config.dataset_path_ = "../../StereoVision/all/data/artroom1";
    }

    try {
        stereovision::StereoVision pipeline(config);
        pipeline.loadImages();
        pipeline.compute_disparity();
        pipeline.compute_depth();
        pipeline.visualize_results(config.save_results_);
        if (config.save_results_) {
            pipeline.save_disparity_map();
        }
        std::cout << "\nPipeline completed successfully!\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
