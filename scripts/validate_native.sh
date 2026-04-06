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
cp cognis_native*.so ../../cognis/dsp/
cd ../..

echo "[2/4] Running pytest suite..."
pytest -q tests/

echo "[3/4] Running native FIR crossover benchmark..."
python -m scripts.benchmark_fir_crossover

echo "[4/4] Validation complete."
echo "=================================================="
