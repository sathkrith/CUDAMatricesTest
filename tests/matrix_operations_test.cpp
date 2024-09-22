#include <gtest/gtest.h>
#include "matrix.hpp"
#include "matrix_utils.hpp"
#include "cpu_matrix_ops.hpp"
#include "cuda_matrix_ops.hpp"

// Test fixture class
class MatrixOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
        int rows = 256;
        int cols = 256;
        A = MatrixUtils::randomMatrix(rows, cols, 42);
        B = MatrixUtils::randomMatrix(cols, rows, 24);
    }

    Matrix A;
    Matrix B;
};

TEST_F(MatrixOperationsTest, MultiplyTest) {
    Matrix C_cpu(A.rows, B.cols);
    Matrix C_cuda(A.rows, B.cols);

    cpuMatrixMultiply(A, B, C_cpu);
    cudaMatrixMultiply(A, B, C_cuda);

    float tolerance = 1e-4f;
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, C_cuda, tolerance));
}

TEST_F(MatrixOperationsTest, AddTest) {
    // Ensure A and B have the same dimensions
    B = MatrixUtils::randomMatrix(A.rows, A.cols, 24);

    Matrix C_cpu(A.rows, A.cols);
    Matrix C_cuda(A.rows, A.cols);

    cpuMatrixAdd(A, B, C_cpu);
    cudaMatrixAdd(A, B, C_cuda);

    float tolerance = 1e-5f;
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, C_cuda, tolerance));
}

TEST_F(MatrixOperationsTest, TransposeTest) {
    Matrix C_cpu(A.cols, A.rows);
    Matrix C_cuda(A.cols, A.rows);

    cpuMatrixTranspose(A, C_cpu);
    cudaMatrixTranspose(A, C_cuda);

    float tolerance = 1e-5f;
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, C_cuda, tolerance));
}

