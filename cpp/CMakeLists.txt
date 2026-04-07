cmake_minimum_required(VERSION 3.24)
project(cognis_native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Phase 1: Native build-system dependency strategy
# Require standard Python development tools
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11 via standard config first (e.g. system install or python env)
find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    message(STATUS "Found pybind11 via standard config: ${pybind11_DIR}")
else()
    # Fallback to FetchContent if local/system pybind11 is not available.
    # This is an explicitly documented fallback of last resort to reduce network brittleness.
    message(STATUS "pybind11 not found locally. Falling back to FetchContent network fetch.")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

include_directories(include)

pybind11_add_module(cognis_native src/pybind_module.cpp src/fir_executor.cpp)

option(COGNIS_EXPERIMENTAL_FAST_MATH "Enable aggressive floating point optimizations (risks determinism)" OFF)

if(COGNIS_EXPERIMENTAL_FAST_MATH)
    target_compile_options(cognis_native PRIVATE -O3 -ffast-math)
    message(STATUS "COGNIS_EXPERIMENTAL_FAST_MATH is ON. Build is not strictly deterministic.")
else()
    target_compile_options(cognis_native PRIVATE -O3)
    message(STATUS "COGNIS_EXPERIMENTAL_FAST_MATH is OFF. Using deterministic DSP compiler flags.")
endif()
