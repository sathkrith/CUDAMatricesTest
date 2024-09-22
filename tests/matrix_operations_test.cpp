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
        int rows = 2;
        int cols = 2;
        A = MatrixUtils::onesMatrix(rows, cols);
        B = MatrixUtils::onesMatrix(cols, rows);
        A(0, 0) = 2.0f;
        A(0, 1) = 3.0f;
        A(1, 0) = 4.0f;
        A(1, 1) = 5.0f;
        B(0, 0) = 6.0f;
        B(0, 1) = 7.0f;
        B(1, 0) = 8.0f;
        B(1, 1) = 9.0f;
    }

    Matrix A;
    Matrix B;
};

TEST_F(MatrixOperationsTest, MultiplyTest) {
    Matrix C_cpu(A.rows, B.cols);
    Matrix C_cuda(A.rows, B.cols);
    Matrix expected = MatrixUtils::onesMatrix(2, 2);
    expected(0, 0) = 36.0f;
    expected(0, 1) = 41.0f;
    expected(1, 0) = 64.0f;
    expected(1, 1) = 73.0f;
    cpuMatrixMultiply(A, B, C_cpu);
    cudaMatrixMultiply(A, B, C_cuda);

    float tolerance = 1e-4f;
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, C_cuda, tolerance));
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, expected, tolerance));
}

TEST_F(MatrixOperationsTest, AddTest) {
    Matrix expected = MatrixUtils::onesMatrix(2, 2);
    expected(0, 0) = 8.0f;
    expected(0, 1) = 10.0f;
    expected(1, 0) = 12.0f;
    expected(1, 1) = 14.0f;
    Matrix C_cpu(A.rows, A.cols);
    Matrix C_cuda(A.rows, A.cols);

    cpuMatrixAdd(A, B, C_cpu);
    cudaMatrixAdd(A, B, C_cuda);

    float tolerance = 1e-5f;
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, C_cuda, tolerance));
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, expected, tolerance));
}

TEST_F(MatrixOperationsTest, TransposeTest) {
    Matrix C_cpu(A.cols, A.rows);
    Matrix C_cuda(A.cols, A.rows);
    Matrix expected = MatrixUtils::onesMatrix(2, 2);
    expected(0, 0) = 2.0f;
    expected(0, 1) = 4.0f;
    expected(1, 0) = 3.0f;
    expected(1, 1) = 5.0f;
    cpuMatrixTranspose(A, C_cpu);
    cudaMatrixTranspose(A, C_cuda);

    float tolerance = 1e-5f;
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, C_cuda, tolerance));
    ASSERT_TRUE(MatrixUtils::matricesAreEqual(C_cpu, expected, tolerance));
}

