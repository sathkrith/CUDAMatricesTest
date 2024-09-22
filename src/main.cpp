#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include "matrix.hpp"
#include "cpu_matrix_ops.hpp"
#include "cuda_matrix_ops.hpp"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --matrixA <file>   JSON file for Matrix A\n";
    std::cout << "  --matrixB <file>   JSON file for Matrix B\n";
    std::cout << "  --operation <op>   Operation to perform: multiply, add, transpose\n";
    std::cout << "  --help             Display this help message\n";
    std::cout << "By default, two 1024x1024 random matrices are generated and multiplied.\n";
}

int main(int argc, char* argv[]) {
    printUsage(argv[0]);
    std::string matrixAFile;
    std::string matrixBFile;
    std::string operation = "multiply"; // Default operation

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrixA" && i + 1 < argc) {
            matrixAFile = argv[++i];
        } else if (arg == "--matrixB" && i + 1 < argc) {
            matrixBFile = argv[++i];
        } else if (arg == "--operation" && i + 1 < argc) {
            operation = argv[++i];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cout << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Load matrices from JSON files or initialize randomly
    Matrix A, B;
    if (!matrixAFile.empty()) {
        A.loadFromJSON(matrixAFile);
    } else {
        A = Matrix(1024, 1024);
        A.randomInitialize();
    }

    if (!matrixBFile.empty()) {
        B.loadFromJSON(matrixBFile);
    } else {
        B = Matrix(1024, 1024);
        B.randomInitialize();
    }

    Matrix C_cpu, C_cuda;

    if (operation == "multiply") {
        C_cpu = Matrix(A.rows, B.cols);
        C_cuda = Matrix(A.rows, B.cols);

        // CPU Matrix Multiplication
        auto start = std::chrono::high_resolution_clock::now();
        cpuMatrixMultiply(A, B, C_cpu);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end - start;
        std::cout << "CPU Matrix Multiplication Time: " << cpu_duration.count() << " seconds.\n";

        // CUDA Matrix Multiplication
        start = std::chrono::high_resolution_clock::now();
        cudaMatrixMultiply(A, B, C_cuda);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cuda_duration = end - start;
        std::cout << "CUDA Matrix Multiplication Time: " << cuda_duration.count() << " seconds.\n";

    } else if (operation == "add") {
        if (A.rows != B.rows || A.cols != B.cols) {
            std::cerr << "Matrices must have the same dimensions for addition.\n";
            return 1;
        }
        C_cpu = Matrix(A.rows, A.cols);
        C_cuda = Matrix(A.rows, A.cols);

        // CPU Matrix Addition
        auto start = std::chrono::high_resolution_clock::now();
        cpuMatrixAdd(A, B, C_cpu);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end - start;
        std::cout << "CPU Matrix Addition Time: " << cpu_duration.count() << " seconds.\n";

        // CUDA Matrix Addition
        start = std::chrono::high_resolution_clock::now();
        cudaMatrixAdd(A, B, C_cuda);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cuda_duration = end - start;
        std::cout << "CUDA Matrix Addition Time: " << cuda_duration.count() << " seconds.\n";

    } else if (operation == "transpose") {
        C_cpu = Matrix(A.cols, A.rows);
        C_cuda = Matrix(A.cols, A.rows);

        // CPU Matrix Transpose
        auto start = std::chrono::high_resolution_clock::now();
        cpuMatrixTranspose(A, C_cpu);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end - start;
        std::cout << "CPU Matrix Transpose Time: " << cpu_duration.count() << " seconds.\n";

        // CUDA Matrix Transpose
        start = std::chrono::high_resolution_clock::now();
        cudaMatrixTranspose(A, C_cuda);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cuda_duration = end - start;
        std::cout << "CUDA Matrix Transpose Time: " << cuda_duration.count() << " seconds.\n";

    } else {
        std::cerr << "Unknown operation: " << operation << "\n";
        return 1;
    }

    // Verify results
    float max_diff = 0.0f;
    for (size_t i = 0; i < C_cpu.data.size(); ++i) {
        float diff = std::abs(C_cpu.data[i] - C_cuda.data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    std::cout << "Maximum difference between CPU and CUDA results: " << max_diff << "\n";

    // Optionally save results to JSON files
    // C_cpu.saveToJSON("result_cpu.json");
    // C_cuda.saveToJSON("result_cuda.json");

    return 0;
}
