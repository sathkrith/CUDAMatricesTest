#include "matrix_utils.hpp"
#include <random>
#include <algorithm>
#include <cmath>

Matrix MatrixUtils::randomMatrix(int rows, int cols, unsigned int seed) {
    Matrix mat(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& elem : mat.data) {
        elem = dist(gen);
    }
    return mat;
}

Matrix MatrixUtils::zeroMatrix(int rows, int cols) {
    Matrix mat(rows, cols);
    std::fill(mat.data.begin(), mat.data.end(), 0.0f);
    return mat;
}

Matrix MatrixUtils::onesMatrix(int rows, int cols) {
    Matrix mat(rows, cols);
    std::fill(mat.data.begin(), mat.data.end(), 1.0f);
    return mat;
}

float MatrixUtils::maxDifference(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions to compare.");
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < A.data.size(); ++i) {
        float diff = std::abs(A.data[i] - B.data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

bool MatrixUtils::matricesAreEqual(const Matrix& A, const Matrix& B, float tolerance) {
    return maxDifference(A, B) <= tolerance;
}
