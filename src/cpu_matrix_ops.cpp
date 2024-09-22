#include "cpu_matrix_ops.hpp"
#include <thread>
#include <vector>
#include <barrier>
#include <latch>
#include <algorithm>

void cpuMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows;
    int n = A.cols;
    int p = B.cols;

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    auto multiply_task = [&](int thread_id) {
        int rows_per_thread = (m + num_threads - 1) / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, m);

        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(multiply_task, t);
    }
}

void cpuMatrixAdd(const Matrix& A, const Matrix& B, Matrix& C) {
    int total_elements = A.rows * A.cols;
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    auto add_task = [&](int thread_id) {
        int elems_per_thread = (total_elements + num_threads - 1) / num_threads;
        int start_idx = thread_id * elems_per_thread;
        int end_idx = std::min(start_idx + elems_per_thread, total_elements);

        for (int idx = start_idx; idx < end_idx; ++idx) {
            C.data[idx] = A.data[idx] + B.data[idx];
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(add_task, t);
    }
}

void cpuMatrixTranspose(const Matrix& A, Matrix& B) {
    int m = A.rows;
    int n = A.cols;
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    auto transpose_task = [&](int thread_id) {
        int rows_per_thread = (m + num_threads - 1) / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, m);

        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < n; ++j) {
                B(j, i) = A(i, j);
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(transpose_task, t);
    }
}

void cpuMatrixVectorMultiply(const Matrix& A, const std::vector<float>& x, std::vector<float>& y) {
    int m = A.rows;
    int n = A.cols;
    y.resize(m);

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    auto multiply_task = [&](int thread_id) {
        int rows_per_thread = (m + num_threads - 1) / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, m);

        for (int i = start_row; i < end_row; ++i) {
            float sum = std::inner_product(A.row(i).begin(), A.row(i).end(), x.begin(), 0.0f);
            y[i] = sum;
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(multiply_task, t);
    }
}
