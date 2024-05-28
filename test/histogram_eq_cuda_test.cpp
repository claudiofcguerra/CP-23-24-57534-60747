#include <filesystem>
#include <cuda_runtime.h>
#include "gtest/gtest.h"
#include "histogram_eq.h"
#include <iostream>

using namespace cp;

#define DATASET_FOLDER "../../dataset/"
#define OUTPUT_FOLDER "../../dataset_output/"

#define TEST_ITERATIONS 10
#define BIG_INPUT_TEST_ITERATIONS 3

TEST(HistogramEq, input01_cuda)
{
    int avg = 0;
    for (int i = 0; i < TEST_ITERATIONS; i++)
    {
        std::string inputFilePath = DATASET_FOLDER "input01.ppm";
        if (!std::filesystem::exists(inputFilePath))
        {
            std::cerr << "Error: Input file " << inputFilePath << " does not exist." << std::endl;
            exit(EXIT_FAILURE);
        }

        wbImage_t inputImage = wbImport(inputFilePath.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        wbImage_t outputImage = iterative_histogram_equalization(inputImage, 4);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg += int(duration.count());
        if (i == TEST_ITERATIONS - 1) { wbExport(OUTPUT_FOLDER "output01_cuda.ppm", outputImage);
    }
    avg /= TEST_ITERATIONS;
    std::cout << "Execution time for " << TEST_ITERATIONS << " iterations in CUDA Function was: " << avg <<
              " microseconds" << std::endl;
}
}




