cmake_minimum_required(VERSION 3.16)
project(project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)

##############################################################
# Autodetect CUDA architecture(s)
##############################################################
#source before we forget
#https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
message("-- CUDA Architecture(s): " ${CMAKE_CUDA_ARCHITECTURES})

###############
# project
###############

include_directories(include)

# Option to enable or disable vectorization
option(ENABLE_VECTORIZATION "Enable vectorization" ON)

# Set compiler flags based on the option
if (ENABLE_VECTORIZATION)
    message(STATUS "Vectorization enabled")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "GNU/CLANG")
        add_compile_options(-ftree-vectorize)
        add_compile_options(-O3)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "MSVC")
        add_compile_options(/O2)  # MSVC doesn't have a direct equivalent, but /O2 enables some vectorization
    endif ()
else ()
    message(STATUS "Vectorization disabled")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-fno-tree-vectorize)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/Od)  # Disable optimization in MSVC
    endif ()
endif ()


# Enable OpmMP if available
find_package(OpenMP REQUIRED)


# Set CUDA properties
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(project_omp
        scripts/export_timings.cpp
        src/histogram_eq_omp.cpp
)
target_compile_definitions(project_omp PRIVATE CSV_FILE_SUFFIX="OMP")

add_executable(project_org
        scripts/export_timings.cpp
        src/histogram_eq_org.cpp
)
target_compile_definitions(project_org PRIVATE CSV_FILE_SUFFIX="ORG_NOT_VECTORIZED")

add_executable(project_cuda_refactor
        scripts/export_timings.cpp
        src/histogram_eq_cuda_scratch.cu
)
target_compile_definitions(project_cuda_refactor PRIVATE CSV_FILE_SUFFIX="CUDA")

add_executable(dataset_generator
        src/dataset_generator.cpp
)

target_link_libraries(project_org PRIVATE wb)
target_link_libraries(dataset_generator PRIVATE wb)
target_link_libraries(project_cuda_refactor PRIVATE wb)
target_link_libraries(project_omp PRIVATE wb OpenMP::OpenMP_CXX)

##############################################################
# Tests
##############################################################

enable_testing()

#############
# Download and unpack googletest at configure time
#############

set(GTEST_HOME ${CMAKE_CURRENT_BINARY_DIR}/googletest)

message("-- Loading Google Test from ${GTEST_HOME}")

configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googletest.in ${GTEST_HOME}/download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines the gtest and gtest_main targets.
add_subdirectory(${GTEST_HOME}/src
        ${GTEST_HOME}/build
        EXCLUDE_FROM_ALL)

set(GTEST_BOTH_LIBRARIES gtest_main gtest)

add_subdirectory(test)