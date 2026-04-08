#!/bin/bash
set -e

echo "=================================================="
echo "COGNIS-V2 Native Validation Path"
echo "=================================================="

echo ""
echo "--- Environment Summary ---"

# 1. Python Path & Version
if command -v python3 &> /dev/null; then
    PY_BIN=$(command -v python3)
    PY_VER=$(python3 --version 2>&1)
    echo "- Python executable: $PY_BIN"
    echo "- Python version: $PY_VER"
else
    echo "FAIL: python3 not found in PATH."
    exit 1
fi

# 2. CMake Version
if command -v cmake &> /dev/null; then
    CMAKE_BIN=$(command -v cmake)
    CMAKE_VER=$(cmake --version | head -n1)
    echo "- CMake executable: $CMAKE_BIN"
    echo "- CMake version: $CMAKE_VER"
else
    echo "FAIL: cmake not found in PATH."
    echo "Please install CMake (e.g., apt install cmake)."
    exit 1
fi

# 3. Python Development Headers
# A quick way to check if Python dev headers are available via python3-config
if command -v python3-config &> /dev/null; then
    echo "- Python dev headers: FOUND (via python3-config)"
else
    echo "FAIL: python3-config not found."
    echo "Please install python development headers (e.g., apt install python3-dev)."
    exit 1
fi

# 4. pybind11
if $PY_BIN -c "import pybind11" &> /dev/null; then
    PYBIND11_DIR=$($PY_BIN -c "import pybind11; print(pybind11.get_cmake_dir())")
    echo "- pybind11: FOUND at $PYBIND11_DIR"
else
    echo "FAIL: pybind11 not found in Python environment."
    echo "Ensure your environment is set up by running: pip install -e .[dev]"
    echo "This will install the editable package, test dependencies, and native build tooling."
    exit 1
fi

echo ""
echo "--- Build Attempt ---"

echo "Attempting to build native module..."
CMAKE_ARGS="-Dpybind11_DIR=$PYBIND11_DIR"

mkdir -p cpp/build
cd cpp/build
if cmake .. $CMAKE_ARGS; then
    echo "CMake configuration: SUCCESS"
else
    echo "FAIL: CMake configuration failed."
    exit 1
fi

if cmake --build .; then
    echo "CMake build: SUCCESS"
else
    echo "FAIL: CMake build failed."
    exit 1
fi

BUILD_DIR="$PWD"
export PYTHONPATH="$BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd ../..

echo ""
echo "--- Validation Summary ---"

# Check if the module actually loads in python from the build directory
if $PY_BIN -c "import cognis_native; print(cognis_native.__file__)" &> /dev/null; then
    echo "- Native module loaded: YES"
else
    echo "FAIL: Native module built but failed to load in python."
    exit 1
fi

# explicitly mark that we are testing with native available
export COGNIS_TEST_NATIVE_AVAILABLE=1

echo "- Running pytest suite..."
if $PY_BIN -m pytest -q tests/; then
    echo "  Pytest suite: PASS"
else
    echo "  Pytest suite: FAIL"
    exit 1
fi

echo "- Running render-loop benchmark..."
if $PY_BIN -m scripts.benchmark_render_loop; then
    echo "  Render benchmark: PASS"
else
    echo "  Render benchmark: FAIL"
    exit 1
fi

echo "- Running native FIR crossover benchmark..."
if $PY_BIN -m scripts.benchmark_fir_crossover; then
    echo "  FIR benchmark: PASS"
else
    echo "  FIR benchmark: FAIL"
    exit 1
fi

echo ""
echo "=================================================="
echo "Validation complete: PASS"
echo "Native DSP module successfully built, loaded, and verified."
echo "=================================================="
