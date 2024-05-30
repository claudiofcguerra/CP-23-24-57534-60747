#include "histogram_eq.h"
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>  // for string stream
#include <cstring>  // for string operations
#include <filesystem>  // for creating directory

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

// Function to extract base name of a file path
string extract_base_name(const char* path)
{
    string filename(path);
    size_t pos = filename.find_last_of("/\\");
    if (pos != string::npos)
    {
        return filename.substr(pos + 1);
    }
    return filename;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout << "usage: " << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }

    wbImage_t inputImage = wbImport(argv[1]);
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));

    // Measure wall-clock time
    auto start_wall = high_resolution_clock::now();
    // Measure CPU time
    auto start_cpu = clock();

    // Call the function
    wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations);

    // Measure CPU time after the function call
    auto end_cpu = clock();
    // Measure wall-clock time after the function call
    auto end_wall = high_resolution_clock::now();

    // Calculate durations
    auto duration_wall = duration_cast<milliseconds>(end_wall - start_wall).count();
    auto duration_cpu = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC; // Convert to milliseconds

    // Ensure output image is written
    wbExport(argv[3], outputImage);

    // Extract base name of the input image file
    string input_basename = extract_base_name(argv[1]);

    // Use the preprocessor macro for the CSV file suffix
#ifdef CSV_FILE_SUFFIX
        string csv_suffix = CSV_FILE_SUFFIX;
#else
    string csv_suffix = "default_suffix";
#endif

    // Construct the CSV file name and path
    string csv_folder = "results";
    string csv_filename = csv_folder + "/" + input_basename + "_" + csv_suffix + "_timing_results.csv";

    // Create the results directory if it does not exist
    if (!fs::exists(csv_folder))
    {
        fs::create_directory(csv_folder);
    }

    // Print out paths
    cout << "Output image file saved as: " << argv[3] << endl;
    cout << "Timing results CSV file saved as: " << csv_filename << endl;

    // Export timings to CSV
    ofstream csv_file;
    csv_file.open(csv_filename, ios::app); // Append mode
    csv_file << duration_wall << "," << duration_cpu << "\n";
    csv_file.close();

    return 0;
}
