#ifndef CPU_MATRIX_OPS_HPP
#define CPU_MATRIX_OPS_HPP

#include "matrix.hpp"
#include <concepts>

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

void cpuMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
void cpuMatrixAdd(const Matrix& A, const Matrix& B, Matrix& C);
void cpuMatrixTranspose(const Matrix& A, Matrix& B);
void cpuMatrixVectorMultiply(const Matrix& A, const std::vector<float>& x, std::vector<float>& y);

#endif // CPU_MATRIX_OPS_H
