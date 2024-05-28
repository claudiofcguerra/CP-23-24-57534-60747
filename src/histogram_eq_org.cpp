//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"

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


    static void greyscale_image_org(const int width, const int height,
                                    const std::shared_ptr<unsigned char[]>& uchar_image,
                                    const std::shared_ptr<unsigned char[]>& gray_image)
    {
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
                const auto idx = i * width + j;
                const auto r = uchar_image[3 * idx];
                const auto g = uchar_image[3 * idx + 1];
                const auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }
    }

    static void create_histogram_org(const std::shared_ptr<unsigned char[]>& gray_image, int (&histogram)[256],
                                     const int size)
    {
        std::fill_n(histogram, HISTOGRAM_LENGTH, 0);
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;
    }

    static void color_correct_and_output(float* output_image_data, const std::shared_ptr<unsigned char[]>& uchar_image,
                                         float (&cdf)[256], const int size_channels, float cdf_min)
    {
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
    }

    static void calculate_cdf_and_fin_min(int (&histogram)[256], float (&cdf)[256], const int size, float& cdf_min)
    {
        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);

        cdf_min = cdf[0];
    }

    static void initialize_uchar_image_array(const float* input_image_data,
                                             const std::shared_ptr<unsigned char[]>& uchar_image,
                                             const int size_channels)
    {
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);
    }

    static void histogram_equalization(const int width, const int height,
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

        initialize_uchar_image_array(input_image_data, uchar_image, size_channels);

        greyscale_image_org(width, height, uchar_image, gray_image);

        create_histogram_org(gray_image, histogram, size);

        float cdf_min;
        calculate_cdf_and_fin_min(histogram, cdf, size, cdf_min);

        color_correct_and_output(output_image_data, uchar_image, cdf, size_channels, cdf_min);
    }


    wbImage_t iterative_histogram_equalization(const wbImage_t& input_image, const int iterations)
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
                                   histogram, cdf);

            input_image_data = output_image_data;
        }

        return output_image;
    }
}
