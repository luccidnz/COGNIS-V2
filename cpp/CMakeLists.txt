cmake_minimum_required(VERSION 3.24)
project(cognis_native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

include_directories(include)

pybind11_add_module(cognis_native src/pybind_module.cpp src/fir_executor.cpp)
target_compile_options(cognis_native PRIVATE -O3 -ffast-math)
