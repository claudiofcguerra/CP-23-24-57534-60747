#!/bin/bash

# Set the paths for the executables
executables=(
    "cmake-build-debug/project_org"
    #"cmake-build-debug/project_omp"
    #"cmake-build-debug/project_cuda_refactor"
)

input_folder="dataset"
output_folder="dataset_output"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Number of iterations
iterations=20

# Loop over each executable
for project_executable in "${executables[@]}"; do
    # Loop over each file in the input folder
    for input_file in "$input_folder"/*.ppm; do
        # Check if the file exists and has the correct extension
        if [[ -f "$input_file" ]]; then
            # Extract the base name of the file
            base_name=$(basename "$input_file")

            # Construct the output file path
            output_file="$output_folder/$base_name"

            # Run the function at least 10 times
            for ((i = 0; i < iterations; i++)); do
                "$project_executable" "$input_file" 4 "$output_file"
            done
        else
            echo "File $input_file does not exist or has an incompatible extension."
        fi
    done
done

