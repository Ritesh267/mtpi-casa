#!/bin/bash
# Further seeds for the LoRA-DPO grid on GPU (seed-major, so every completed seed covers all
# nine configurations). Completed runs are skipped; safe to re-launch after an interruption.
#   bash src2/run_dpo_more_seeds.sh 2 3 4
cd "$(dirname "$0")/.." || exit 1
# Set HF_HOME to choose the model cache and PYTHON to choose the interpreter.
PY=${PYTHON:-python}
BASE=Qwen/Qwen2.5-1.5B-Instruct
LOG=results2/dpo_gpu/grid_more.log
for seed in "$@"; do
  $PY src2/dpo_gpu_study.py --base $BASE --arm A1 --seed $seed >> $LOG 2>&1
  for det in strong weak; do
    for sel in confident boundary; do
      for arm in A2 A3; do
        $PY src2/dpo_gpu_study.py --base $BASE --arm $arm --det $det --sel $sel --seed $seed >> $LOG 2>&1
      done
    done
  done
  echo "SEED_DONE $seed $(date -u)" >> $LOG
done
echo GRID_MORE_DONE >> $LOG
