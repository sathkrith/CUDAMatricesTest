#ifndef CUDA_MATRIX_OPS_HPP
#define CUDA_MATRIX_OPS_HPP

#include "matrix.hpp"

// Declare CUDA functions
void cudaMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
void cudaMatrixAdd(const Matrix& A, const Matrix& B, Matrix& C);
void cudaMatrixTranspose(const Matrix& A, Matrix& B);
void cudaMatrixVectorMultiply(const Matrix& A, const std::vector<float>& x, std::vector<float>& y)
#endif // CUDA_MATRIX_OPS_H