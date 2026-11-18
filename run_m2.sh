#!/usr/bin/env bash
# Runs every Milestone 2 experiment. Finished subjects/folds are skipped, so the script can be restarted.
# Usage: bash run_m2.sh [python] [data_npz]
PY=${1:-python}
DATA=${2:-../data_npz}
mkdir -p results_m2/logs
run() { "$PY" -W ignore src/improve.py --data-npz "$DATA" "$@" 2>&1 | tee -a results_m2/logs/run_m2.log; }

for seed in 20200220 1 2; do
  # within-subject: baseline + one change at a time
  run --protocol within --model shallow --seed $seed
  run --protocol within --model eegnet --seed $seed
  run --protocol within --model shallow --ea --seed $seed
  run --protocol within --model shallow --augment --seed $seed
done
for seed in 20200220 1 2; do
  # leave-one-subject-out
  run --protocol loso --model shallow --seed $seed
  run --protocol loso --model shallow --ea --seed $seed
  run --protocol loso --model eegnet --ea --seed $seed
  run --protocol loso --model shallow --ea --augment --seed $seed
done
for seed in 20200220 1 2; do
  run --protocol within --model conformer --seed $seed
done
# EEGConformer LOSO is ~1.5 h per seed on an RTX 5050 laptop GPU: one seed
run --protocol loso --model conformer --ea --seed 20200220
