#!/bin/bash
set -e

echo "=================================================="
echo "COGNIS-V2 Native Validation Path"
echo "=================================================="

echo "[1/4] Building optional native backend..."
mkdir -p cpp/build
cd cpp/build
cmake ..
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
