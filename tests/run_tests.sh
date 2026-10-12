#!/usr/bin/env bash
# End-to-end test of the Pyro SAI pipeline on synthetic data (no cloud access needed).
# Usage: bash tests/run_tests.sh   (from the folder containing the notebooks and pyrosai.py)
set -euo pipefail
HERE=$(pwd); T=$(mktemp -d)
python tests/make_synthetic_data.py "$T/synthetic_data"
cp *.ipynb pyrosai.py "$T/"; cd "$T"
export PYROSAI_TEST=1 PYROSAI_SYNTH_ROOT="$T/synthetic_data" PYROSAI_CATALOG=pyrosai_catalog.json
for nb in 00_Data_Discovery Task1_FWI_P95_Extremes Task2_VPD_Evaporative_Demand; do
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 $nb.ipynb --output exec_$nb.ipynb
done
echo "OK - outputs in $T/test_output"
