set(TEST_TARGET "template_test")

set(TEST_SOURCES
        template_test.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})


set(TEST_TARGET "histogram_eq_cuda_test")

set(TEST_SOURCES
        histogram_eq_cuda_test.cpp
        ../src/histogram_eq_cuda.cu)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb)

set(TEST_TARGET "histogram_eq_omp_test")

set(TEST_SOURCES
        histogram_eq_omp_test.cpp
        ../src/histogram_eq_omp.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb OpenMP::OpenMP_CXX)

set(TEST_TARGET "histogram_eq_org_test")

set(TEST_SOURCES
        histogram_eq_org_test.cpp
        ../src/histogram_eq_org.cpp)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES} wb)