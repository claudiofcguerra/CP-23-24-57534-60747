#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "histogram_eq.h"

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;


    __global__ void histogram_kernel(const unsigned char *gray_image, int size, int *histogram) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            atomicAdd(&histogram[gray_image[tid]], 1);
        }
    }


    void compute_cdf(const int *histogram, float *cdf, int size) {
        int *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;


        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, histogram, cdf, HISTOGRAM_LENGTH);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, histogram, cdf, HISTOGRAM_LENGTH);
        cudaFree(d_temp_storage);


        thrust::transform(thrust::device, cdf, cdf + HISTOGRAM_LENGTH, cdf, [size] __device__ (float x) {
            return x / size;
        });
    }

    __global__ void equalization_kernel(const unsigned char *input_image_data, float *output_image_data,
                                        const float *cdf, float cdf_min, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            output_image_data[tid] = 255.0f * (cdf[input_image_data[tid]] - cdf_min) / (1.0f - cdf_min);
        }
    }

    template <typename T>
    __global__ void convert_to_gray_kernel(const T *input_image_data, unsigned char *gray_image, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            gray_image[tid] = 0.299f * input_image_data[3 * tid] + 0.587f * input_image_data[3 * tid + 1] + 0.114f * input_image_data[3 * tid + 2];
        }
    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        wbImage_t output_image = wbImage_new(width, height, channels);
        float *input_image_data = wbImage_getData(input_image);
        float *output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);
        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        // Device arrays
        unsigned char *d_gray_image;
        int *d_histogram;
        float *d_cdf;
        float *d_input_image_data;
        float *d_output_image_data;


        cudaMalloc((void **)&d_gray_image, size * sizeof(unsigned char));
        cudaMalloc((void **)&d_histogram, HISTOGRAM_LENGTH * sizeof(int));
        cudaMalloc((void **)&d_cdf, HISTOGRAM_LENGTH * sizeof(float));
        cudaMalloc((void **)&d_input_image_data, size_channels * sizeof(float));
        cudaMalloc((void **)&d_output_image_data, size_channels * sizeof(float));

        cudaMemcpy(d_input_image_data, input_image_data, size_channels * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < iterations; i++) {

            convert_to_gray_kernel<<<(size + 255) / 256, 256>>>(d_input_image_data, d_gray_image, size);
            cudaDeviceSynchronize();


            cudaMemset(d_histogram, 0, HISTOGRAM_LENGTH * sizeof(int));
            histogram_kernel<<<(size + 255) / 256, 256>>>(d_gray_image, size, d_histogram);
            cudaDeviceSynchronize();


            compute_cdf(d_histogram, d_cdf, size);

            // Copy CDF from device to host
            cudaMemcpy(cdf, d_cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);


            float cdf_min = cdf[0];
            for (int j = 1; j < HISTOGRAM_LENGTH; j++) {
                if (cdf[j] < cdf_min) {
                    cdf_min = cdf[j];
                }
            }


            equalization_kernel<<<(size + 255) / 256, 256>>>(d_gray_image, d_output_image_data, d_cdf, cdf_min, size);
            cudaDeviceSynchronize();


            cudaMemcpy(d_input_image_data, d_output_image_data, size_channels * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(output_image_data, d_output_image_data, size_channels * sizeof(float), cudaMemcpyDeviceToHost);


        cudaFree(d_gray_image);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        cudaFree(d_input_image_data);
        cudaFree(d_output_image_data);

        return output_image;
    }
}