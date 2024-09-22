#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <span>
#include <string>

// Include the JSON library
#include "nlohmann/json.hpp"

class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int rows = 0, int cols = 0);

    void randomInitialize();

    void print() const;

    // Element access
    float& operator()(int row, int col);
    float operator()(int row, int col) const;

    // Row access using std::span
    std::span<float> row(int row);
    std::span<const float> row(int row) const;

    // New methods for JSON support
    void loadFromJSON(const std::string& filename);
    void saveToJSON(const std::string& filename) const;

    // Overloaded operators for passing as arguments
    Matrix& operator=(const std::vector<std::vector<float>>& values);
};

#endif // MATRIX_H
