cmake_minimum_required(VERSION 3.20)
project(KernelTuner LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernels)
add_executable(matmul_bench kernels/matmul.cu)
target_compile_options(matmul_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xptxas=-v
    ${OPTIMIZATION_FLAGS}
>)

target_link_libraries(matmul_bench PRIVATE cuda)
