#!/bin/bash
set -e

echo "=================================================="
echo "COGNIS-V2 Native Validation Path"
echo "=================================================="

echo "[1/4] Building optional native backend..."

# Provide hints to CMake to find pybind11 in the active python environment if available
if python -c "import pybind11" &> /dev/null; then
    PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
    echo "Found pybind11 in Python env: $PYBIND11_DIR"
    CMAKE_ARGS="-Dpybind11_DIR=$PYBIND11_DIR"
else
    CMAKE_ARGS=""
fi

mkdir -p cpp/build
cd cpp/build
cmake .. $CMAKE_ARGS
make

# Ensure we don't accidentally commit .so files by carefully moving them
# to the execution context but keeping .gitignore aware.
cp cognis_native*.so ../../cognis/dsp/
cd ../..

# explicitly mark that we are testing with native available
export COGNIS_TEST_NATIVE_AVAILABLE=1

echo "[2/4] Running pytest suite..."
pytest -q tests/

echo "[3/4] Running native FIR crossover benchmark..."
python -m scripts.benchmark_fir_crossover

echo "[4/4] Validation complete."
echo "=================================================="
