//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"

namespace cp
{
    constexpr auto HISTOGRAM_LENGTH = 256;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    // https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }


    unsigned char *d_gray_image, *d_uchar_image;
    float* d_input_image_data;

    void cuda_prepare_memory(const int size, const int size_channels, const float* input_image_data)
    {
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_gray_image), size * sizeof(unsigned char)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_input_image_data), size_channels * sizeof(float)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_uchar_image), size_channels * sizeof(unsigned char)));

        gpuErrchk(
            cudaMemcpy(d_input_image_data, input_image_data, size_channels * sizeof(float), cudaMemcpyHostToDevice));
    }

    void cuda_free_memory()
    {
        gpuErrchk(cudaFree(d_gray_image));
        gpuErrchk(cudaFree(d_uchar_image));
    }


    static float prob(const int x, const int size)
    {
        return static_cast<float>(x) / static_cast<float>(size);
    }

    __global__ void initialize_uchar_image_array_kernel(const float* input_image_data, unsigned char* uchar_image,
                                                        int size_channels)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size_channels)
        {
            uchar_image[idx] = static_cast<unsigned char>(255 * input_image_data[idx]);
        }
    }

    __global__ void greyscale_image_org_kernel(const int width, const int height, const unsigned char* uchar_image,
                                               unsigned char* gray_image)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int num_pixels = width * height;
        if (idx < num_pixels)
        {
            const auto r = uchar_image[3 * idx];
            const auto g = uchar_image[3 * idx + 1];
            const auto b = uchar_image[3 * idx + 2];
            gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }

    void initialize_uchar_image_array(const float* d_input_image_data, unsigned char* d_uchar_image, int size_channels)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        initialize_uchar_image_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input_image_data, d_uchar_image, size_channels);
        cudaDeviceSynchronize();
    }

    void greyscale_image(const int width, const int height, const unsigned char* d_uchar_image,
                         unsigned char* d_gray_image)
    {
        int num_pixels = width * height;
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

        greyscale_image_org_kernel<<<blocksPerGrid, threadsPerBlock>>>(width, height, d_uchar_image, d_gray_image);
        cudaDeviceSynchronize();
    }

    static void create_histogram_org(const unsigned char* gray_image, int (&histogram)[256],
                                     const int size)
    {
        std::fill_n(histogram, HISTOGRAM_LENGTH, 0);

        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;
    }

    static void calculate_cdf_and_fin_min(int (&histogram)[256], float (&cdf)[256], const int size, float& cdf_min)
    {
        cdf[0] = prob(histogram[0], size);
        cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
        {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }
    }


    static unsigned char correct_color(const float cdf_val, const float cdf_min)
    {
        return static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min));
    }

    static void color_correct_and_output(float* output_image_data, unsigned char* uchar_image,
                                         float (&cdf)[256], const int size_channels, float cdf_min)
    {
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
                                       float (&cdf)[HISTOGRAM_LENGTH])
    {
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;
        auto* unchar_image = new unsigned char[size_channels];
        auto* gray_imagem = new unsigned char[size_channels];

        initialize_uchar_image_array(d_input_image_data, d_uchar_image, size_channels);

        greyscale_image(width, height, d_uchar_image, d_gray_image);

        gpuErrchk(
            cudaMemcpy(unchar_image, d_uchar_image, size_channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        gpuErrchk(
            cudaMemcpy(gray_imagem, d_gray_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        create_histogram_org(gray_imagem, histogram, size);

        float cdf_min;
        calculate_cdf_and_fin_min(histogram, cdf, size, cdf_min);

        color_correct_and_output(output_image_data, unchar_image, cdf, size_channels, cdf_min);
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

        cuda_prepare_memory(size, size_channels, input_image_data);

        for (int i = 0; i < iterations; i++)
        {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);
            gpuErrchk(
                cudaMemcpy(d_input_image_data, output_image_data, size_channels * sizeof(float), cudaMemcpyHostToDevice
                ));
        }

        cuda_free_memory();

        return output_image;
    }
}
