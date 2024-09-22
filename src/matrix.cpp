#include "matrix.hpp"
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>

using json = nlohmann::json;

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols) {}

void Matrix::randomInitialize() {
    std::mt19937 gen(0); // Fixed seed
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& elem : data) {
        elem = dist(gen);
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        auto r = row(i);
        for (const auto& val : r) {
            std::cout << std::setw(8) << val << " ";
        }
        std::cout << '\n';
    }
}

float& Matrix::operator()(int row, int col) {
    return data[row * cols + col];
}

float Matrix::operator()(int row, int col) const {
    return data[row * cols + col];
}

std::span<float> Matrix::row(int row) {
    return { &data[row * cols], static_cast<size_t>(cols) };
}

std::span<const float> Matrix::row(int row) const {
    return { &data[row * cols], static_cast<size_t>(cols) };
}

void Matrix::loadFromJSON(const std::string& filename) {
    std::ifstream file(filename);
    json j;
    file >> j;

    rows = j["rows"].get<int>();
    cols = j["cols"].get<int>();
    data.resize(rows * cols);

    auto values = j["data"].get<std::vector<std::vector<float>>>();
    for (int i = 0; i < rows; ++i) {
        for (int j_col = 0; j_col < cols; ++j_col) {
            (*this)(i, j_col) = values[i][j_col];
        }
    }
}

void Matrix::saveToJSON(const std::string& filename) const {
    json j;
    j["rows"] = rows;
    j["cols"] = cols;
    std::vector<std::vector<float>> values(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j_col = 0; j_col < cols; ++j_col) {
            values[i][j_col] = (*this)(i, j_col);
        }
    }
    j["data"] = values;

    std::ofstream file(filename);
    file << j.dump(4); // Pretty print with 4 spaces indentation
}

Matrix& Matrix::operator=(const std::vector<std::vector<float>>& values) {
    rows = values.size();
    cols = values.empty() ? 0 : values[0].size();
    data.resize(rows * cols);

    for (int i = 0; i < rows; ++i) {
        for (int j_col = 0; j_col < cols; ++j_col) {
            (*this)(i, j_col) = values[i][j_col];
        }
    }
    return *this;
}
