//
// Created by herve on 13-04-2024.
//
#include <cub/cub.cuh>
#include "histogram_eq.h"

namespace cp
{
    constexpr auto HISTOGRAM_LENGTH = 256;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    // https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    inline void gpuAssert(const cudaError_t code, const char* file, const int line, const bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    int* d_histogram;
    unsigned char *d_gray_image, *d_uchar_image;
    float *d_input_image_data, *d_output_image_data;

    void cuda_prepare_memory(const int size, const int size_channels, const float* input_image_data)
    {
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_gray_image), size * sizeof(unsigned char)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_input_image_data), size_channels * sizeof(float)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_output_image_data), size_channels * sizeof(float)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_uchar_image), size_channels * sizeof(unsigned char)));

        gpuErrchk(cudaMalloc(&d_histogram, HISTOGRAM_LENGTH * sizeof(int)));
        gpuErrchk(cudaMemset(d_histogram, 0, HISTOGRAM_LENGTH * sizeof(int))); // Initialize histogram to zero

        gpuErrchk(
            cudaMemcpy(d_input_image_data, input_image_data, size_channels * sizeof(float), cudaMemcpyHostToDevice));
    }

    void cuda_free_memory()
    {
        gpuErrchk(cudaFree(d_histogram));
        gpuErrchk(cudaFree(d_gray_image));
        gpuErrchk(cudaFree(d_uchar_image));
        gpuErrchk(cudaFree(d_input_image_data));
        gpuErrchk(cudaFree(d_output_image_data));
    }


    static float prob(const int x, const int size)
    {
        return static_cast<float>(x) / static_cast<float>(size);
    }

    __global__ void initialize_uchar_image_array_kernel(const float* input_image_data, unsigned char* uchar_image,
                                                        const int size_channels)
    {
        if (const int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_channels)
        {
            uchar_image[idx] = static_cast<unsigned char>(255 * input_image_data[idx]);
        }
    }

    __global__ void greyscale_image_org_kernel(const int width, const int height, const unsigned char* uchar_image,
                                               unsigned char* gray_image)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (const int num_pixels = width * height; idx < num_pixels)
        {
            const auto r = uchar_image[3 * idx];
            const auto g = uchar_image[3 * idx + 1];
            const auto b = uchar_image[3 * idx + 2];
            gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }

    __device__ unsigned char correct_color(const float cdf_val, const float cdf_min)
    {
        return static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min));
    }

    // CUDA kernel to perform color correction and output
    __global__ void color_correct_and_output_kernel(float* output_image_data, unsigned char* uchar_image,
                                                    const float* cdf, const int size_channels, const float cdf_min)
    {
        if (const int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_channels)
        {
            uchar_image[idx] = correct_color(cdf[uchar_image[idx]], cdf_min);
            output_image_data[idx] = static_cast<float>(uchar_image[idx]) / 255.0f;
        }
    }

    void initialize_uchar_image_array(const float* d_input_image_data, unsigned char* d_uchar_image,
                                      const int size_channels)
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
        const int num_pixels = width * height;
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

        greyscale_image_org_kernel<<<blocksPerGrid, threadsPerBlock>>>(width, height, d_uchar_image, d_gray_image);
        cudaDeviceSynchronize();
    }


    // Host function to call the CUDA kernel
    void color_correct_and_output(float* d_output_image_data, unsigned char* d_uchar_image, const float* cdf,
                                  const int size_channels, const float cdf_min)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        color_correct_and_output_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_output_image_data, d_uchar_image, cdf, size_channels, cdf_min);
        cudaDeviceSynchronize();
    }

    void create_histogram_cub(const unsigned char* d_gray_image, const int size, int* d_histogram
    )
    {
        // Determine temporary device storage requirements for CUB
        const int num_samples = size;
        constexpr float lower_level = 0.0f;
        constexpr float upper_level = 256.0f;
        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;
        constexpr int num_levels = HISTOGRAM_LENGTH + 1;
        const auto d_samples = d_gray_image;

        cub::DeviceHistogram::HistogramEven(
            d_temp_storage, temp_storage_bytes,
            d_samples, d_histogram,
            num_levels, lower_level, upper_level,
            num_samples
        );
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Compute the histogram
        cub::DeviceHistogram::HistogramEven(
            d_temp_storage, temp_storage_bytes,
            d_samples, d_histogram,
            num_levels, lower_level, upper_level,
            num_samples
        );
        // Free temporary storage
        cudaFree(d_temp_storage);
    }

    static void calculate_cdf_and_fin_min(const int (&histogram)[256], float (&cdf)[256], const int size,
                                          float& cdf_min)
    {
        cdf[0] = prob(histogram[0], size);
        cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
        {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }
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

        initialize_uchar_image_array(d_input_image_data, d_uchar_image, size_channels);

        greyscale_image(width, height, d_uchar_image, d_gray_image);

        create_histogram_cub(d_gray_image, size, d_histogram);

        // Copy the histogram back to the host
        gpuErrchk(cudaMemcpy(histogram, d_histogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost));

        float cdf_min;
        calculate_cdf_and_fin_min(histogram, cdf, size, cdf_min);

        color_correct_and_output(d_output_image_data, d_uchar_image, cdf, size_channels, cdf_min);
    }


    wbImage_t iterative_histogram_equalization(const wbImage_t& input_image, const int iterations)
    {
        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        const wbImage_t output_image = wbImage_new(width, height, channels);
        const float* input_image_data = wbImage_getData(input_image);
        float* output_image_data = wbImage_getData(output_image);

        const std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        const std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

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
                cudaMemcpy(d_input_image_data, d_output_image_data, size_channels * sizeof(float),
                    cudaMemcpyDeviceToDevice
                ));
        }

        gpuErrchk(
            cudaMemcpy(output_image_data, d_output_image_data, size_channels * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

        cuda_free_memory();

        return output_image;
    }
}
