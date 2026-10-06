#!/bin/bash
# LoRA-DPO grid on GPU for seeds 0 and 1: arm A0 once, then for each seed A1 plus
# A2 and A3 x {strong,weak} x {confident,boundary} (nine trained runs per seed).
# Completed runs are skipped, so the script is safe to re-launch after an interruption.
# Further seeds: bash src2/run_dpo_more_seeds.sh 2 3 4
#
# Usage:  bash src2/run_dpo_grid.sh
# Needs results2/dpo_pool.json (python src2/dpo_prep.py) and one NVIDIA GPU with about
# 16 GB. Run one job on the GPU at a time. Set PYTHON to choose the interpreter and
# HF_HOME to choose the model cache, for example
#   PYTHON=.venv/bin/python HF_HOME=$HOME/hf bash src2/run_dpo_grid.sh
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
BASE=Qwen/Qwen2.5-1.5B-Instruct
mkdir -p results2/dpo_gpu
LOG=results2/dpo_gpu/grid.log
$PY src2/dpo_gpu_study.py --base $BASE --arm A0 >> $LOG 2>&1
for seed in 0 1; do
  $PY src2/dpo_gpu_study.py --base $BASE --arm A1 --seed $seed >> $LOG 2>&1
  for det in strong weak; do
    for sel in confident boundary; do
      for arm in A2 A3; do
        $PY src2/dpo_gpu_study.py --base $BASE --arm $arm --det $det --sel $sel --seed $seed >> $LOG 2>&1
      done
    done
  done
done
echo GRID_DONE >> $LOG
