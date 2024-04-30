//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <omp.h>

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }


    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min));
    }

    std::vector<float> parallel_prefix_sum(const std::vector<float> &arr) {
        if (arr.empty()) return {};

        int n = arr.size();
        std::vector<float> result(n, 0.0f);
        int num_threads = omp_get_max_threads();
        std::vector<float> sums(num_threads, 0.0f);

        // Calculate the local prefix sums for each chunk of the array in parallel
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * n / num_threads;
            int end = (tid + 1) * n / num_threads;

            // Handle the case where n is not evenly divisible by num_threads
            if (tid == num_threads - 1) {
                end = n;
            }

            if (start < end) {
                result[start] = arr[start];
                for (int i = start + 1; i < end; ++i) {
                    result[i] = result[i - 1] + arr[i];
                }
                sums[tid] = result[end - 1];
            }
        }

        // Compute the total sum for each chunk
        for (int i = 1; i < num_threads; ++i) {
            sums[i] += sums[i - 1];
        }

        // In parallel, update each chunk with the sum of the previous chunks
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * n / num_threads;
            int end = (tid + 1) * n / num_threads;

            if (tid == num_threads - 1) {
                end = n;
            }

            if (tid > 0) {
                float add = sums[tid - 1];
                for (int i = start; i < end; ++i) {
                    result[i] += add;
                }
            }
        }

        return result;
    }

    static void histogram_equalization(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       const std::shared_ptr<unsigned char[]> &uchar_image,
                                       const std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

#pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);

#pragma omp parallel for
        for (int i = 0; i < height; i++)
#pragma omp parallel for firstprivate(i)
                for (int j = 0; j < width; j++) {
                    auto idx = i * width + j;
                    auto r = uchar_image[3 * idx];
                    auto g = uchar_image[3 * idx + 1];
                    auto b = uchar_image[3 * idx + 2];
                    gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
                }

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);


        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;

        cdf[0] = prob(histogram[0], size);
        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }

        for (int i = 0; i < size_channels; i++) {
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);
        }

#pragma parallel for
        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {

        // Input 1 2 5 4 9 7 0 1
        // Output: 1 3 8 12 21 28 28 29

        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        wbImage_t output_image = wbImage_new(width, height, channels);
        float *input_image_data = wbImage_getData(input_image);
        float *output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);

            input_image_data = output_image_data;
        }

        return output_image;
    }
}