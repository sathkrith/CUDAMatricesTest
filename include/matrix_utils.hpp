#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include "matrix.hpp"

class MatrixUtils {
public:
    // Generates a random matrix of given dimensions
    static Matrix randomMatrix(int rows, int cols, unsigned int seed = 0);

    // Generates a matrix filled with zeros
    static Matrix zeroMatrix(int rows, int cols);

    // Generates a matrix filled with ones
    static Matrix onesMatrix(int rows, int cols);

    // Compares two matrices and returns the maximum absolute difference
    static float maxDifference(const Matrix& A, const Matrix& B);

    // Checks if two matrices are approximately equal within a tolerance
    static bool matricesAreEqual(const Matrix& A, const Matrix& B, float tolerance = 1e-5f);
};

#endif // MATRIX_UTILS_HPP
