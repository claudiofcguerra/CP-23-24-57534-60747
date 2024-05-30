#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_eq.h"

using namespace cp;

#define DATASET_FOLDER "../../dataset/"
#define OUTPUT_FOLDER "../../dataset_output/"

#define TEST_ITERATIONS 20
#define BIG_INPUT_TEST_ITERATIONS 10


TEST(HistogramEq, input01)
{
    int avg = 0;
    std::string outputPath = std::string(OUTPUT_FOLDER) + "output01_" + TEST_NAME + ".ppm";
    for (int i = 0; i < TEST_ITERATIONS; i++)
    {
        //        std::cout << "Iteration: " << i << std::endl;
        wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
        auto start = std::chrono::high_resolution_clock::now();
        wbImage_t outputImage = iterative_histogram_equalization(inputImage, 4);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg += int(duration.count());
        if (i == TEST_ITERATIONS - 1) { wbExport(outputPath.c_str(), outputImage); }
    }
    avg /= TEST_ITERATIONS;
    std::cout << "Execution time for " << TEST_ITERATIONS << " iterations in " << TEST_NAME << " Function was: " << avg
        <<
        " microseconds" << std::endl;
}

TEST(HistogramEq, big_input)
{
    int avg = 0;
    std::string outputPath = std::string(OUTPUT_FOLDER) + "big_output_" + TEST_NAME + ".ppm";
    for (int i = 0; i < BIG_INPUT_TEST_ITERATIONS; i++)
    {
        //        std::cout << "Iteration: " << i << std::endl;
        wbImage_t inputImage = wbImport(DATASET_FOLDER "sample_5184×3456.ppm");
        auto start = std::chrono::high_resolution_clock::now();
        wbImage_t outputImage = iterative_histogram_equalization(inputImage, 4);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg += int(duration.count());
        if (i == TEST_ITERATIONS - 1) { wbExport(outputPath.c_str(), outputImage); }
    }
    avg /= BIG_INPUT_TEST_ITERATIONS;
    std::cout << "Execution time for " << BIG_INPUT_TEST_ITERATIONS << " iterations in " << TEST_NAME <<
        " Function was: "
        << avg <<
        " microseconds" << std::endl;
}

