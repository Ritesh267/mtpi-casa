# Reproducing the results, table by table

Run every command from the repository root. Times are taken from the shipped logs unless a line says otherwise. Two machines were used:

- **Original run** (`results_v1_original/`): two CPU cores, 7 GB of memory, no GPU.
- **Later runs** (`results/`, `results2/`, `results3/`): four CPU cores, 15 GB of memory, one Tesla T4 with 15 GB.

## Before you start

1. Install the packages in `requirements.txt` and fetch the data as described in `data/README.md`. Sections 1 to 3 below need `data/mtpi_bench.jsonl`. The RAG scripts also need the two SQuAD files.
2. The scripts overwrite files in the result folders without asking, and later scripts load those files. Work in a copy, or check `git status` afterwards.
3. The saved models are not shipped. Run section 1 first: `src/eval_casa.py` writes `results/stack.pkl` and `results/casa_meta_linear.pkl`, and `src/adv_train.py` writes `results/casa_at.pkl`. Almost every later script loads them.
4. A rebuild gives numbers close to the re-run in `results/`. The stateless, pooling, accumulator-only and linear CASA rows should match to three decimals. The gradient-boosted rows will not match the printed tables exactly (see the README, "Result folders").
5. `python tools/check_report.py` prints the fields behind the tables from the shipped files. `python tools/check_report.py --run results` prints the core tables from `results/`, which is where your own run writes.

## 1. Core pipeline (CPU)

Run in this order.

| Report item | Command | Result file | Time |
|---|---|---|---|
| Tables 3.1, 3.2 | `python src/build_bench.py` | `data/mtpi_bench.jsonl` | seconds |
| Table 6.1 | `python src/grid.py` | `results/grid.json`, `results/grid.csv` | about 20 min, of which 938 s for the encoder |
| Tables 6.2 to 6.6 | `python src/eval_casa.py` | `results/casa_main.json`, `results/casa_main.csv` | about 10 min from nothing: 214 s, 40 s and 305 s for the three caches |
| Tables 6.9, 6.10 | `python src/adv_train.py` | `results/adv_train.json`, `results/casa_at.pkl` | 724 s for the training features (977 s in the re-run); the rest was not timed |
| Table 6.8 | `python src/asym_control.py` | `results/asym_control.json` | 645 s for the training features; the rest was not timed |
| Tables 6.7, 6.9, 6.10 | `python src/recalib.py` | `results/adaptive_final.json` | not timed |
| Table 6.11 | `python src/run_dpo.py` | `results/dpo.json` | about 21 min: 160 s, then 343 s, 368 s and 361 s for the three arms |
| Figures 3.1, 4.1, 6.1 to 6.8 | `python src/figures.py` | `figures/fig_bench.png` and nine more | seconds |

Notes:

- Table 6.7 is the `stateless` and `CASA-base` columns of `adaptive_final.json`. Table 6.9 is all four model columns. The first row of Table 6.10 is the `fpr` entry of the same file, and its other rows are `fpr_fragmented_benign` in `adv_train.json`.
- The last row of Table 6.1 is `turn_auc` in `casa_main.json`.
- Table 6.12 is a summary of the other tables. No script writes it.
- `src/grid.py` is not needed for CASA. `src/eval_casa.py` fits its own stacked detector.
- `src/eval_casa.py` reuses `results/stack.pkl`, `results/turn_scores.json` and `results/window_scores.json` when they exist. To rebuild from nothing, delete all three together.
- `src/save_detector.py` is optional. Nothing reads its output, and it needs files that `src/grid.py` writes.
- `src/eval_adaptive.py` is an earlier adaptive evaluation. No printed table comes from it.
- The scripts print their tables. The shipped logs were made by redirecting that output, for example `python src/eval_casa.py | tee results/casa.log`.
- `src/figures.py` draws from `results/`, so its output will differ from the shipped images, which were drawn from the original run.

## 2. Extended experiments

These load the models written in section 1.

| Report item | Command | Result file | Hardware and time |
|---|---|---|---|
| Section 6.12, bootstrap | `python src2/bootstrap_casa.py` | `results2/bootstrap_casa.json` | CPU, 12 s |
| Section 6.12, partitions | `for s in 1 2 3 4; do python src2/multiseed_casa.py --seed $s; done` | `results2/multiseed_casa/seed1.json` to `seed4.json` | CPU, 496 to 650 s per seed |
| Table 6.13, decisions | `python scripts/precompute_defence.py` | `results2/defence_decisions.json` | CPU, 539 s |
| Table 6.13, models | see the command below | `results2/asr_live.json` | GPU, 327 to 1,321 s per model |
| Table 6.14 | three hardening runs, then the merge: see below | `results2/casa_fix.json` | CPU, about 2 hours per seed |
| Table 6.15 | `python src2/composed_attacks.py` | `results2/composed_eval.json` | CPU, about 2 min (not logged) |
| Table 6.16 | `python src2/xai_audit.py` | `results2/xai_audit.json` | CPU, 4 s |
| Section 6.11, checks | `python src3/xai_robustness.py` | `results3/xai_robustness.json` | CPU, seconds |
| Section 6.4, pacing mechanism | `python src3/pace_mechanism.py` | `results3/pace_mechanism.json` | CPU, about 7 min (not logged) |
| Table 6.17 | the GPU grid: see below | `results2/dpo_gpu/`, `results2/summary.json` | GPU, 212 s for arm A0, 654 to 714 s per trained run |

Served models (Table 6.13). The second command is the one that was run:

```bash
python scripts/precompute_defence.py
python scripts/measure_asr_hf.py \
  --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct \
           Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct microsoft/Phi-3.5-mini-instruct \
  --defences none stateless casa_base casa_at --max-new 96 --out results2/asr_live.json
```

A model already present in the output file is skipped. Qwen2.5-3B and Phi-3.5-mini are loaded in 4-bit, the others in half precision.

Hardening (Table 6.14). The three seeds are independent and can run at the same time:

```bash
python src2/casa_hardening.py --seeds 0 --out results2/casa_fix_seed0.json
python src2/casa_hardening.py --seeds 1 --out results2/casa_fix_seed1.json
python src2/casa_hardening.py --seeds 2 --out results2/casa_fix_seed2.json
python src2/merge_hardening.py
```

Alignment on Qwen2.5-1.5B-Instruct (Table 6.17):

```bash
python src2/dpo_prep.py                 # CPU: writes results2/dpo_pool.json
bash src2/run_dpo_grid.sh               # GPU: arm A0, then nine trained runs for each of seeds 0 and 1
bash src2/run_dpo_more_seeds.sh 2       # GPU: nine more runs for every further seed named
python src2/assemble_summary.py         # collects everything into results2/summary.json
```

- Arm A0 must run first. It writes the refusal thresholds that every other run loads.
- A run whose result file exists is skipped. Delete the file to repeat a run.
- Run one job on the GPU at a time. The logs show out-of-memory failures when two shared it.
- The base model is loaded in float16 and is not quantised. Each trained run takes 75 optimiser steps on 600 pairs.
- `python src2/check_cache.py` checks that the cached scorer of the study gives the same scores as the uncached one. It needs the GPU and `results2/dpo_pool.json`.
- `src2/assemble_summary.py` reads `results2/dpo_pool.json` for its `detector_agreement` field. Without that file the field is left out, so run `src2/dpo_prep.py` before you rebuild the summary.

## 3. Live RAG and online decisions

| Report item | Command | Result file | Hardware and time |
|---|---|---|---|
| Table 6.18 | `python src3/rag_build.py` | `results3/rag_retrieval.json` | GPU if present, else CPU; not logged |
| Table 6.19 | `python src3/rag_detect.py --retriever bm25` and `--retriever dense` | `results3/rag_detection_bm25.json`, `results3/rag_detection_dense.json` | CPU, about 6 min each with a warm score cache; a first run takes longer (not logged) |
| Table 6.19, paired intervals | `python src3/rag_paired.py` | `results3/rag_paired.json` | CPU, under a minute |
| Table 6.20, models | `python src3/rag_asr.py --models MODEL ...` | `results3/rag/gen_*.json` | GPU, 1,195 s to 10,688 s per model |
| Table 6.20, table | `python src3/rag_asr_table.py` | `results3/rag_asr.json` | CPU, seconds |
| Table 6.21 | `python src3/online_eval.py` | `results3/online_eval.json` | CPU, 3,292 s |

Notes:

- `src3/rag_build.py` also writes the session files `results3/rag/sessions_*.json` and the embedding cache `results3/rag/wiki_emb.npy`. They contain benchmark text and are not shipped. Every other RAG script needs them, except `src3/rag_paired.py` and `src3/rag_asr_table.py`.
- `src3/rag_detect.py` also writes `results3/rag_decisions_bm25.json`, `results3/rag_decisions_dense.json`, `results3/rag_session_stats_bm25.json` and `results3/rag_session_stats_dense.json`, which the two scripts above read.
- The served-model run used the dense retriever, five chunks per turn and at most 64 new tokens in the final reply, which are the defaults. The times per model in `results3/logs/rag_asr.log` are 1,195 s (Qwen2.5-0.5B), 2,541 s (Qwen2.5-1.5B), 3,249 s (SmolLM2-1.7B) and 10,688 s (Qwen2.5-3B in 4-bit). A model whose four output files exist is skipped.
- `src3/rag_asr.py` stores outcomes only: whether the code is in the reply, whether the answer is correct, a refusal marker and context statistics.

## 4. Figures

| Report figures | Command | Reads | Writes |
|---|---|---|---|
| 3.1, 4.1, 6.1 to 6.8 | `python src/figures.py` | `results/`, the benchmark | `figures/` |
| 6.9 to 6.13 | `python src2/make_figures.py` | `results2/summary.json`, `results3/xai_robustness.json` | `figures2/` |
| 6.14 to 6.16 | `python src3/make_figures3.py` | `results3/rag_retrieval.json`, `results3/rag_detection_bm25.json`, `results3/rag_detection_dense.json`, `results3/rag_asr.json` | `figures3/` |

Which image is which figure:

| Figure | File |
|---|---|
| 3.1 | `figures/fig_bench.png` |
| 4.1 | `figures/fig_arch.png` |
| 6.1 | `figures/fig_grid.png` |
| 6.2 | `figures/fig_ceiling.png` |
| 6.3 | `figures/fig_opcurve.png` |
| 6.4 | `figures/fig_breakdown.png` |
| 6.5 | `figures/fig_ablation.png` |
| 6.6 | `figures/fig_augcontrol.png` |
| 6.7 | `figures/fig_adaptive.png` |
| 6.8 | `figures/fig_dpo.png` |
| 6.9 | `figures2/figS1_asr_by_model.png` |
| 6.10 | `figures2/figS4_casa_fix.png` |
| 6.11 | `figures2/figS5_composed.png` |
| 6.12 | `figures2/figS3_xai.png` |
| 6.13 | `figures2/figS2_dpo_scale.png` |
| 6.14 | `figures3/figR1_rag_delivery.png` |
| 6.15 | `figures3/figR2_rag_detection.png` |
| 6.16 | `figures3/figR3_rag_asr.png` |

`figures2/figS1b_dsr_asr.png` and `figures2/figS6_bootstrap.png` are drawn by the same script and are not used in the report.

## 5. Which field is which table cell

| Report | File | Field |
|---|---|---|
| Table 6.1 | `grid.json` | the row with matching `rep` and `clf`: `dim`, `auc`, `ap`, `recall`, `fpr`, `f1` |
| Table 6.2 | `casa_main.json` | `results[0]` to `results[6]` |
| Table 6.3 | `casa_main.json` | `ceiling` |
| Table 6.4 | `casa_main.json` | `operating_curve` |
| Table 6.5 | `casa_main.json` | `breakdown` |
| Table 6.6 | `casa_main.json` | `results[7]` to `results[11]`, with `delta_auc` |
| Tables 6.7, 6.9 | `adaptive_final.json` | `rows` |
| Table 6.8 | `asym_control.json` | `detection`, `fpr` |
| Table 6.10 | `adaptive_final.json`, `adv_train.json` | `fpr`; `fpr_fragmented_benign` |
| Table 6.11 | `dpo.json` | `rows` |
| Table 6.13 | `results2/asr_live.json` | `models`, then `defences`: `asr`, `block_rate`, `benign_block_rate` |
| Table 6.14 | `results2/casa_fix.json` | `summary`: `mean`, `lo`, `hi` |
| Table 6.15 | `results2/composed_eval.json` | one entry per set |
| Table 6.16 | `results2/xai_audit.json` | `features` |
| Table 6.17 | `results2/summary.json` | `dpo_gpu` |
| Table 6.18 | `results3/rag_retrieval.json` | `planted`, `gold_recall` |
| Table 6.19 | `results3/rag_detection_bm25.json`, `results3/rag_detection_dense.json` | `by_mode`, `clean` |
| Table 6.20 | `results3/rag_asr.json` | `models`, `blocked` |
| Table 6.21 | `results3/online_eval.json` | `attack`, `benign`, `append` |

For Tables 6.1 to 6.11 read the files in `results_v1_original/`. The same file names in `results/` hold the re-run.

## 6. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'data'` | the script was started outside the repository root | `cd` to the root first |
| `FileNotFoundError: data/mtpi_bench.jsonl` | the benchmark has not been built | see `data/README.md` |
| `FileNotFoundError` for a CSV under `jailbreak_llms/` | the corpus is not cloned, or is elsewhere | clone it, or pass `--corpus` |
| `FileNotFoundError: results/stack.pkl` | the saved detector is not shipped | run `python src/eval_casa.py` |
| `ModuleNotFoundError: No module named 'casa'` when you load a pickle in your own code | the pickles contain classes defined in `src/` | put `import sys; sys.path.insert(0, "src")` before `joblib.load` |
| a pickle does not load, or numbers shift without a message | another scikit-learn version; warnings are switched off | install the pinned version, or rebuild the pickles |
| "loaded cached turn scores" after you deleted `stack.pkl` | the score caches are not tied to the detector | delete the three files named in section 1 together |
| `src2/dpo_gpu_study.py` stops on a missing threshold file | arm A0 has not been run | run `--arm A0` first |
| `src2/dpo_gpu_study.py` prints "exists, skipping" | the result file of that run exists | delete it to repeat the run |
| `torch.OutOfMemoryError` on the GPU | two jobs share one GPU | run one at a time |
| `scripts/measure_asr_hf.py` stops with "decisions incomplete" | the decisions file is missing or partial | run `python scripts/precompute_defence.py` to the end |
| `scripts/measure_asr_hf.py` prints "cached" and does nothing | that model is already in the output file | use another `--out` |
| a GPU script fails on a machine without an NVIDIA GPU | the device name `cuda` is written into the code | these scripts need such a GPU |
