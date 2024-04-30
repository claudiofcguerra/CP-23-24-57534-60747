#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_eq.h"

using namespace cp;

#define DATASET_FOLDER "../../dataset/"

#define TEST_ITERATIONS 50



TEST(HistogramEq, Input01_4) {


//     wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
//     auto start = std::chrono::high_resolution_clock::now();
//     wbImage_t outputImage = iterative_histogram_equalization(inputImage, 4);
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;


    int avg = 0;
    for (int i = 0; i < TEST_ITERATIONS; i++) {
//        std::cout << "Iteration: " << i << std::endl;
        wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
        auto start = std::chrono::high_resolution_clock::now();
        wbImage_t outputImage = iterative_histogram_equalization(inputImage, 4);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg += int(duration.count());
        wbExport(DATASET_FOLDER "output01_5.ppm", outputImage);
    }
     avg /= TEST_ITERATIONS;
    std::cout << "Execution time: " << avg << " microseconds" << std::endl;
}