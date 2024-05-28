set(TEST_TARGET "histogram_eq_cuda_test")

set(TEST_SOURCES
        histogram_eq_test.cpp
        ../src/histogram_eq_cuda.cu)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb)
target_compile_definitions(${TEST_TARGET} PRIVATE TEST_NAME="CUDA")


set(TEST_TARGET "histogram_eq_omp_test")

set(TEST_SOURCES
        histogram_eq_test.cpp
        ../src/histogram_eq_omp.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb OpenMP::OpenMP_CXX)
target_compile_definitions(${TEST_TARGET} PRIVATE TEST_NAME="OMP")

set(TEST_TARGET "histogram_eq_org_test")

set(TEST_SOURCES
        histogram_eq_test.cpp
        ../src/histogram_eq_org.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb)
target_compile_definitions(${TEST_TARGET} PRIVATE TEST_NAME="ORG")