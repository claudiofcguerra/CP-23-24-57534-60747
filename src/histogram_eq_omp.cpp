//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <omp.h>

namespace cp
{
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float prob(const int x, const int size)
    {
        return static_cast<float>(x) / static_cast<float>(size);
    }


    static unsigned char correct_color(const float cdf_val, const float cdf_min)
    {
        return static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min));
    }


    static void histogram_equalization_omp(const int width, const int height,
                                           const float* input_image_data,
                                           float* output_image_data,
                                           const std::shared_ptr<unsigned char[]>& uchar_image,
                                           const std::shared_ptr<unsigned char[]>& gray_image,
                                           int (&histogram)[HISTOGRAM_LENGTH],
                                           float (&cdf)[HISTOGRAM_LENGTH])
    {
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

#pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);

#pragma omp parallel for
        for (int i = 0; i < height; i++)
#pragma omp parallel for firstprivate(i)
            for (int j = 0; j < width; j++)
            {
                const auto idx = i * width + j;
                const auto r = uchar_image[3 * idx];
                const auto g = uchar_image[3 * idx + 1];
                const auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }

        std::fill_n(histogram, HISTOGRAM_LENGTH, 0);

#pragma omp for reduction(+ : histogram)
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;

        cdf[0] = prob(histogram[0], size);
        const auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
        {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }

        for (int i = 0; i < size_channels; i++)
        {
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);
        }

#pragma parallel for
        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
    }

    static void histogram_equalization_org(const int width, const int height,
                                           const float* input_image_data,
                                           float* output_image_data,
                                           const std::shared_ptr<unsigned char[]>& uchar_image,
                                           const std::shared_ptr<unsigned char[]>& gray_image,
                                           int (&histogram)[HISTOGRAM_LENGTH],
                                           float (&cdf)[HISTOGRAM_LENGTH])
    {
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
                const auto idx = i * width + j;
                const auto r = uchar_image[3 * idx];
                const auto g = uchar_image[3 * idx + 1];
                const auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }

        std::fill_n(histogram, HISTOGRAM_LENGTH, 0);
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;

        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);

        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);

        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
    }

    static void histogram_equalization(const int width, const int height,
                                       const float* input_image_data,
                                       float* output_image_data,
                                       const std::shared_ptr<unsigned char[]>& uchar_image,
                                       const std::shared_ptr<unsigned char[]>& gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH],
                                       const func_type type)
    {
        if (type == func_type::OMP)
        {
            return histogram_equalization_omp(width, height, input_image_data, output_image_data, uchar_image,
                                              gray_image,
                                              histogram, cdf);
        }
        if (type == func_type::CUDA)
        {
            std::cout << "CUDA is not available. Running sequential version." << std::endl;
        }
        return histogram_equalization_org(width, height, input_image_data, output_image_data, uchar_image,
                                          gray_image,
                                          histogram, cdf);
    }

    wbImage_t iterative_histogram_equalization(const wbImage_t& input_image, const int iterations, const func_type type)
    {
        // Input 1 2 5 4 9 7 0 1
        // Output: 1 3 8 12 21 28 28 29

        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        const wbImage_t output_image = wbImage_new(width, height, channels);
        float* input_image_data = wbImage_getData(input_image);
        float* output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        for (int i = 0; i < iterations; i++)
        {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf, type);

            input_image_data = output_image_data;
        }

        return output_image;
    }
}
