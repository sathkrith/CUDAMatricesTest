# CUDAMatricesTest
Compares various matrix operations with their corresponding multi threaded implementations.

## Overview

This project is a comparative study of matrix operations (multiplication, addition, transposition, etc.) implemented using both:

- **C++ Multithreading** for CPU parallelism.
- **CUDA** for GPU parallelism.

The application supports:

- Reading matrices from JSON files.
- Generating random matrices.
- Command-line interface for specifying operations and inputs.
- Unit testing using Google Test framework to ensure accuracy in both CPU and GPU modes.

### General Requirements

- **C++ Compiler** with C++20 support:
  - GCC 10+ (Linux)
  - Clang 10+ (Linux, macOS)
  - MSVC 2019 or later (Windows)
- **CUDA Toolkit** compatible with your GPU and compiler.
- **CMake** version 3.14 or higher.
- **Google Test** framework for unit testing.
- **nlohmann/json** library for JSON parsing.

### Building the Project
#### Step 1: Create a Build Directory

In the root directory of the project, create a build directory and navigate into it.
```
mkdir build
cd build
```
#### Step 2: Configure the Project with CMake
Linux and macOS

```
cmake .

Windows

If you used vcpkg for dependencies:

cmd

cmake . -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -A x64
```

Replace C:/path/to/vcpkg with the actual path to your vcpkg installation.
#### Step 3: Build the Project

```
cmake --build . --config Release
```
On Linux and macOS, you can also use:
```
make
```

### Running the Application
```
./release/matrix_ops [options]
```

#### Options
Options

- --matrixA <file>: JSON file for Matrix A.
- --matrixB <file>: JSON file for Matrix B.
- --operation <op>: Operation to perform (multiply, add, transpose).
- --rowsA <value>: Number of rows for Matrix A (if generating randomly).
- --colsA <value>: Number of columns for Matrix A (if generating randomly).
- --colsB <value>: Number of columns for Matrix B (if generating randomly).
- --help: Display the help message.

#### Example

```
./release/matrix_ops --operation multiply
```

#### JSON Example
JSON:
```
{
    "rows": <number_of_rows>,
    "cols": <number_of_columns>,
    "data": [
        [row0_col0, row0_col1, ..., row0_colN],
        [row1_col0, row1_col1, ..., row1_colN],
        ...
    ]
}
example:
{
    "rows": 3,
    "cols": 2,
    "data": [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
}

```
Running the application with json
./release/matrix_ops --matrixA matrixA.json --matrixB matrixB.json --operation add

### Running Tests
```
ctest
```
