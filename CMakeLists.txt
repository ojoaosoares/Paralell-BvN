cmake_minimum_required(VERSION 3.16)
project(ParallelBvN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(bvn_bench
    src/main.cpp
    src/traffic.cpp
    src/regularizer.cpp
    src/euler_trail.cpp
    src/decomposer_seq.cpp
    src/decomposer_naive.cpp
    src/decomposer_arch.cpp
    src/postprocessor.cpp
)

target_include_directories(bvn_bench PRIVATE include)

target_compile_options(bvn_bench PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Release>:-O3 -march=native -funroll-loops>
)

target_link_libraries(bvn_bench PRIVATE OpenMP::OpenMP_CXX)
