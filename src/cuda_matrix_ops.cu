#include "matrix.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void cudaMatrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // m
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // p

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void cudaMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows;
    int n = A.cols;
    int p = B.cols;

    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * p * sizeof(float);
    size_t size_C = m * p * sizeof(float);

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, A.data.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data.data(), size_B, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(16, 16);
    dim3 dimGrid((p + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    cudaMatrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, p);

    // Copy result back to host
    cudaMemcpy(C.data.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Implement similar functions for cudaMatrixAdd and cudaMatrixTranspose
__global__ void cudaMatrixVectorMultiplyKernel(const float* A, const float* x, float* y, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * x[k];
        }
        y[row] = sum;
    }
}

void cudaMatrixVectorMultiply(const Matrix& A, const std::vector<float>& x, std::vector<float>& y) {
    int m = A.rows;
    int n = A.cols;
    size_t size_A = m * n * sizeof(float);
    size_t size_x = n * sizeof(float);
    size_t size_y = m * sizeof(float);

    float *d_A, *d_x, *d_y;

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_y, size_y);

    // Copy data to device
    cudaMemcpy(d_A, A.data.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), size_x, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;

    // Launch kernel
    cudaMatrixVectorMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_x, d_y, m, n);

    // Copy result back to host
    y.resize(m);
    cudaMemcpy(y.data(), d_y, size_y, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

// CUDA kernel for matrix addition
__global__ void matrixAddKernel(const float* A, const float* B, float* C, int rows, int cols) {
    // Calculate the global thread ID
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    // Compute the index in the 1D array
    int idx = row * cols + col;

    // Perform the addition if within bounds
    if (row < rows && col < cols) {
        C[idx] = A[idx] + B[idx];
    }
}


void cudaMatrixAdd(const Matrix& A, const Matrix& B, Matrix& C) {
    // Check if matrix dimensions match
    if (A.rows != B.rows || A.cols != B.cols) {
        std::cerr << "Error: Matrices A and B must have the same dimensions." << std::endl;
        return;
    }

    if (A.rows != C.rows || A.cols != C.cols) {
        std::cerr << "Error: Matrix C must have the same dimensions as A and B." << std::endl;
        return;
    }

    int rows = A.rows;
    int cols = A.cols;
    size_t size = rows * cols * sizeof(float);

    // Device pointers
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaError_t err;

    // Allocate device memory for A
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for A: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Allocate device memory for B
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); // Free previously allocated memory
        return;
    }

    // Allocate device memory for C
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy data from host to device for A
    err = cudaMemcpy(d_A, A.data.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from host to device for A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy data from host to device for B
    err = cudaMemcpy(d_B, B.data.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from host to device for B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Define block and grid sizes
    const int TILE_SIZE = 16;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error launching matrixAddKernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy result from device to host
    err = cudaMemcpy(C.data.data(), d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from device to host for C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void cudaMatrixTransposeKernel(const float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        B[col * m + row] = A[row * n + col];
    }
}

void cudaMatrixTranspose(const Matrix& A, Matrix& B) {
    int m = A.rows;
    int n = A.cols;
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * m * sizeof(float);

    float *d_A, *d_B;

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);

    // Copy data to device
    cudaMemcpy(d_A, A.data.data(), size_A, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(16, 16);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    cudaMatrixTransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B, m, n);

    cudaMemcpy(B.data.data(), d_B, size_B, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
}