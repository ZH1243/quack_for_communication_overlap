cmake_minimum_required(VERSION 3.18)

project(quack_stream_gather_proxy LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(stream_gather_proxy stream_gather_proxy.cpp)
target_compile_features(stream_gather_proxy PRIVATE cxx_std_17)
target_link_libraries(stream_gather_proxy PRIVATE CUDA::cudart CUDA::cuda_driver)
target_compile_options(stream_gather_proxy PRIVATE -Wall -Wextra -Wpedantic)

