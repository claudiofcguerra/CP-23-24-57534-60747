cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "" CACHE STRING "")
if (CMAKE_CUDA_FLAGS)
    list(REMOVE_ITEM CMAKE_CUDA_FLAGS "--cudart static")
endif ()
string(APPEND CMAKE_CUDA_FLAGS "--cudart shared")
set(CMAKE_CUDA_ARCHITECTURES 52)
project(project-cp-gpu LANGUAGES CXX CUDA)

# Comment CUDA lines and uncomment this one to run on Mac.
#project(project-cp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

##############################################################
# Autodetect CUDA architecture(s)
##############################################################

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
string(REPLACE "52+PTX" "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

###############
# OPENMP
###############

find_package(OpenMP REQUIRED)
#message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
#message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")

###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)

###############
# project
###############

include_directories(include)

add_executable(project_cuda
        src/main.cpp
        src/histogram_eq_cuda.cu
)

add_executable(project_omp
        src/main.cpp
        src/histogram_eq_omp.cpp
)

add_executable(project_org
        src/main.cpp
        src/histogram_eq_org.cpp
)

target_link_libraries(project_cuda PUBLIC wb)
target_link_libraries(project_org PUBLIC wb)
target_link_libraries(project_omp PUBLIC wb OpenMP::OpenMP_CXX)

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