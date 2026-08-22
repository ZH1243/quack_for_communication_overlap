cmake_minimum_required(VERSION 3.18)

project(quack_stream_gather_proxy LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

function(configure_proxy_target target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

add_executable(stream_gather_proxy stream_gather_proxy.cpp)
configure_proxy_target(stream_gather_proxy)

add_library(stream_gather_proxy_thread SHARED stream_gather_proxy.cpp)
target_compile_definitions(
  stream_gather_proxy_thread PRIVATE QUACK_STREAM_GATHER_PROXY_LIBRARY=1
)
configure_proxy_target(stream_gather_proxy_thread)
