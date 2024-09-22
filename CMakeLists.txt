cmake_minimum_required(VERSION 3.18)
project(MatrixOperations LANGUAGES CXX CUDA)

find_package(nlohmann_json CONFIG REQUIRED)
# Find GTest package using CONFIG mode
find_package(GTest CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Paths
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Source files
set(SOURCES
    ${SRC_DIR}/matrix.cpp
    ${SRC_DIR}/matrix_utils.cpp
    ${SRC_DIR}/cpu_matrix_ops.cpp
    ${SRC_DIR}/cuda_matrix_ops.cu
)

# Main executable
add_executable(matrix_ops ${SRC_DIR}/main.cpp ${SOURCES})
target_include_directories(matrix_ops PRIVATE ${INCLUDE_DIR})

# CUDA settings
set_target_properties(matrix_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable testing
enable_testing()

# Test executable
add_executable(test_matrix_ops ${TEST_DIR}/matrix_operations_test.cpp ${SOURCES})
target_include_directories(test_matrix_ops PRIVATE ${INCLUDE_DIR})
target_link_libraries(matrix_ops PRIVATE nlohmann_json::nlohmann_json)

# CUDA settings for test executable
set_target_properties(test_matrix_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link GTest libraries to test executable
target_link_libraries(test_matrix_ops PRIVATE GTest::gtest GTest::gtest_main)
target_link_libraries(test_matrix_ops PRIVATE nlohmann_json::nlohmann_json)

# Add test
add_test(NAME MatrixOperationsTest COMMAND test_matrix_ops)
