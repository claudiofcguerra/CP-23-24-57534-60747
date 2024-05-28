#include <cub/cub.cuh>
#include "histogram_eq.h"

namespace cp
{
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float prob(const int x, const int size)
    {
        return static_cast<float>(x) / static_cast<float>(size);
    }

    void compute_histogram(int* histogram, unsigned char* gray_image, const int size)
    {
        //!    // Declare, allocate, and initialize device-accessible pointers for
        //!    // input samples and output histogram
        //!    int      num_samples;    // e.g., 10
        //!    float*   d_samples;      // e.g., [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5]
        //!    int*     d_histogram;    // e.g., [ -, -, -, -, -, -]
        //!    int      num_levels;     // e.g., 7       (seven level boundaries for six bins)
        //!    float    lower_level;    // e.g., 0.0     (lower sample value boundary of lowest bin)
        //!    float    upper_level;    // e.g., 12.0    (upper sample value boundary of upper bin)

        // Sample on entire image, so num_samples is size.
        const int num_samples = size;
        // unsigned char instead of float because of parameter
        unsigned char* d_samples = gray_image;
        int* d_histogram = histogram;
        // The number of histogram bins is (``num_levels - 1``), so HISTOGRAM_LENGTH + 1 levels will yield
        // the number of bins we need.
        constexpr int num_levels = HISTOGRAM_LENGTH + 1;
        constexpr float lower_level = 0;
        constexpr float upper_level = 256;

        // Determine temporary device storage requirements
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceHistogram::HistogramEven(
            d_temp_storage, temp_storage_bytes,
            d_samples, d_histogram, num_levels,
            lower_level, upper_level, num_samples);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Compute histograms
        cub::DeviceHistogram::HistogramEven(
            d_temp_storage, temp_storage_bytes,
            d_samples, d_histogram, num_levels,
            lower_level, upper_level, num_samples);
    }

    __global__ void correct_color_kernel(const unsigned char* input_image_data, float* output_image_data,
                                         const float* cdf, const float cdf_min, const int size)
    {
        const int tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid < size)
        {
            output_image_data[tid] = static_cast<unsigned char>(255 * (cdf[input_image_data[tid]] - cdf_min) / (1 -
                cdf_min));
        }
    }

    __global__ void greyscale_image_kernel(const float* input_image_data, unsigned char* gray_image,
                                           const int size)
    {
        const int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid < size)
        {
            auto r = static_cast<unsigned char>(255 * input_image_data[3 * tid]);
            auto g = static_cast<unsigned char>(255 * input_image_data[3 * tid + 1]);
            auto b = static_cast<unsigned char>(255 * input_image_data[3 * tid + 2]);
            gray_image[tid] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.007 * b);
        }
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

    // Device arrays
    int* d_histogram;
    unsigned char* d_gray_image;
    float *d_output_image_data, *d_input_image_data;

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

        greyscale_image_kernel<<<(size + 255) / 256, 256>>>(d_input_image_data, d_gray_image, size);

        compute_histogram(d_histogram, d_gray_image, size);

        cudaMemcpy(histogram, d_histogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);

        float cdf_min;
        calculate_cdf_and_fin_min(histogram, cdf, size, cdf_min);
        correct_color_kernel<<<(size + 255) / 256, 256>>>(d_gray_image, d_output_image_data, cdf, cdf_min,
                                                          size_channels);
    }

    void cuda_prepare_memory(const int size, const int size_channels)
    {
        cudaMalloc(reinterpret_cast<void**>(&d_gray_image), size * sizeof(unsigned char));
        cudaMalloc(reinterpret_cast<void**>(&d_histogram), HISTOGRAM_LENGTH * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&d_input_image_data), size_channels * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&d_output_image_data), size_channels * sizeof(float));
    }

    void cuda_free_memory()
    {
        cudaFree(d_gray_image);
        cudaFree(d_histogram);
        cudaFree(d_input_image_data);
        cudaFree(d_output_image_data);
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

        cuda_prepare_memory(size, size_channels);
        cudaMemcpy(d_input_image_data, input_image_data, size_channels * sizeof(float), cudaMemcpyHostToDevice);
        printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

        for (int i = 0; i < iterations; i++)
        {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);
            cudaMemcpy(d_input_image_data, d_output_image_data, size_channels * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(output_image_data, d_output_image_data, size_channels * sizeof(float), cudaMemcpyDeviceToHost);


        cuda_free_memory();

        return output_image;
    }
}
