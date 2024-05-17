#include "histogram_eq.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }


    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Cannot load image " << argv[1] << std::endl;
        return 1;
    }

    int n_iterations = std::stoi(argv[2]);


    int width = input_image.cols;
    int height = input_image.rows;
    int channels = input_image.channels();
    wbImage_t wb_input_image = wbImage_new(width, height, channels);
    float *input_data = wbImage_getData(wb_input_image);


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                input_data[(i * width + j) * channels + k] = input_image.at<cv::Vec3b>(i, j)[k] / 255.0f;
            }
        }
    }


    wbImage_t wb_output_image = cp::iterative_histogram_equalization(wb_input_image, n_iterations);

    // Converter de volta para cv::Mat
    cv::Mat output_image(height, width, input_image.type());
    float *output_data = wbImage_getData(wb_output_image);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                output_image.at<cv::Vec3b>(i, j)[k] = static_cast<unsigned char>(output_data[(i * width + j) * channels + k] * 255.0f);
            }
        }
    }


    cv::imwrite(argv[3], output_image);


    wbImage_delete(wb_input_image);
    wbImage_delete(wb_output_image);

    std::cout << "Histogram equalization completed successfully." << std::endl;
    return 0;
}
