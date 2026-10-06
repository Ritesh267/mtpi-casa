# MTPI-Bench and CASA

Code and result files for the report *Conversation-Level Defence Against Multi-Turn Prompt Injection in Locally Deployed Large Language Models: Sequential Evidence Accumulation, Provenance Separation and Detector-Supervised Preference Alignment*, by Ritesh Mukherjee (October 2026).

**Status.** This is research code that accompanies an extended research report. The report extends the author's MSc dissertation and is not the dissertation submitted for the degree. The code, the experiments and the report were produced with assistance from a research-mentoring service and from generative-AI tools, and the report's statement on authorship says so. Nothing here has been peer reviewed. It is a record of experiments, not a product and not a filter that is ready to deploy. Please read "Known limitations" before you rely on any number.

## What the project is

A prompt injection is text that makes a language model follow an instruction that its user or operator did not intend. Most filters check one message at a time. An attacker can then split the instruction over several turns of a conversation, or put it into a document that the model retrieves. This project studies a defence that decides per conversation instead of per message, for models that run on local hardware.

The parts:

- **MTPI-Bench.** A benchmark of 2,302 conversations and 11,756 turns, built from a public collection of jailbreak prompts and ordinary prompts. Each prompt is cut into 1 to 8 turns and delivered directly or inside a document envelope. Benign prompts are cut in the same way.
- **Turn-level detector.** A comparison of six text representations and five classifiers, and a stacked detector that gives every turn a probability.
- **CASA** (Conversation-Aware Sequential Aggregation). A small classifier over 31 features of the whole sequence of turn scores: an accumulator of evidence, trajectory features, separate treatment of retrieved content, and re-scoring of joined turns.
- **Adaptive attacker and adversarial training.** Six scripted rewriting strategies, and CASA-AT, the same classifier retrained on rewritten attack and benign conversations.
- **Internal layer.** LoRA adapters trained with Direct Preference Optimisation (DPO) on preference pairs that the detector labels, first on a small policy trained from scratch and then on Qwen2.5-1.5B-Instruct.
- **Later experiments.** Attack success against five served open-weight models, two repairs for the attacks that survive, template-composed attacks, an attribution audit with robustness checks, a live retrieval-augmented generation (RAG) pipeline over Wikipedia paragraphs, and a comparison of deciding once per conversation with deciding after every turn.

## Results at a glance

Each figure below is read from a shipped result file, and `python tools/check_report.py` prints them. "Per-turn filter" means the stateless baseline that blocks when any single turn scores above its threshold.

- **Detection per turn is hard.** The stacked detector has a test AUC of 0.878. The best single configuration reaches 91.0% recall only at a 36.8% false-positive rate (`results_v1_original/casa_main.json`, `results_v1_original/grid.json`).
- **Deciding per conversation helps on the benchmark.** At a 5% false-positive budget on the main partition, recall rises from 48.1% to 57.0% and AUC from 0.867 to 0.908 against the per-turn filter (`results_v1_original/casa_main.json`).
- **The gain is modest and varies.** The AUC gain is 0.041 with a 95% bootstrap interval of 0.021 to 0.061. CASA recovers 75.1% of the AUC that fragmentation removes on the main partition, and 52.2% to 66.3% on four further partitions (`results2/bootstrap_casa.json`, `results2/multiseed_casa/`).
- **An attacker who knows the defence breaks it.** With the decision taken at the end of the conversation, inserting benign turns between payload turns (pacing) lowers CASA's detection to 8.4%, and a combined strategy lowers it to 0.0% (`results_v1_original/adaptive_final.json`).
- **Adversarial training repairs part of that.** CASA-AT detects 49.8% under pacing and 43.5% under fragmentation. The per-turn filter detects 48.1% and 57.4% of the same attacks, so the conversation layer ends level with it or behind it. Dilution (3.0%) and the combined strategy (8.4%) stay undetected (same file).
- **Augmenting only the attack class is a trap.** It gives 85.7% detection under the combined strategy and blocks 78.1% of benign conversations transformed in the same way (`results_v1_original/asym_control.json`).
- **Deciding after every turn changes the picture.** At thresholds recalibrated for that rule, CASA detects 54.4% of unmodified attacks and the per-turn filter 48.1%. Under pacing CASA detects 25.3% and the per-turn filter 48.1% (`results3/online_eval.json`).
- **Served models.** The share of attacks that are delivered and not refused is 86% with no defence, 46% with the per-turn filter and 29% with CASA, as means over five models. The defences are not at matched benign block rates in that study: 4.5% against 10.7% (`results2/asr_live.json`).
- **Repairs for the surviving attacks.** Detection of dilution rises from 5.3% to 19.8%, and of the combined strategy from 0.4% to 9.4%, over three training seeds. Most of these attacks stay undetected (`results2/casa_fix.json`).
- **Live RAG.** Chunking and ranking limit what an attacker delivers. Under BM25 the complete payload reaches the model in 41.8% of sessions for a query-targeted document, and in 90.3% when the attacker knows the chunk size. For the query-targeted attacker, scoring each retrieved chunk separately raises the per-turn filter from 35.4% to 51.5%; the conversation layer on top of that gives 54.9% (`results3/rag_retrieval.json`, `results3/rag_detection_bm25.json`).
- **Alignment from detector labels.** On the small CPU policy, detector-generated pairs give a separation of +44.3 points against +45.5 for curated pairs, with one run per arm (`results_v1_original/dpo.json`). For Qwen2.5-1.5B-Instruct see the `dpo_gpu` block of `results2/summary.json`.

## Repository layout

```text
README.md, LICENSE, CITATION.cff, requirements.txt, .gitignore
docs/REPRODUCE.md        table by table: script, command, result file, hardware, time
tools/check_report.py    prints the result fields behind the tables (standard library only)
data/README.md           how to fetch the public data; nothing else is shipped in data/

src/                     core pipeline (CPU)
  build_bench.py           builds MTPI-Bench from the public corpus
  data.py                  loading, partition by source prompt
  reps.py, encoder.py      six representations; transformer encoder trained from scratch
  ensemble.py, grid.py     stacked turn detector; turn-level grid
  casa.py, eval_casa.py    CASA features and accumulator; conversation-level comparison
  adaptive.py              six attacker strategies and the payload-preservation screen
  adv_train.py, recalib.py symmetric adversarial training; recalibrated thresholds
  asym_control.py          the attack-only augmentation control
  lora_dpo.py, run_dpo.py  LoRA and DPO written out; the small three-arm study
  eval_adaptive.py         an earlier adaptive evaluation, behind no printed table
  save_detector.py         optional; its output is not used
  figures.py               Figures 3.1, 4.1 and 6.1 to 6.8
src2/                    extended experiments
  casa_hardening.py, merge_hardening.py   repairs for the two surviving attacks
  composed_attacks.py                     template-composed attacks
  xai_audit.py                            attribution in the linear meta-classifier
  dpo_prep.py, dpo_gpu_study.py           LoRA-DPO on Qwen2.5-1.5B-Instruct (GPU)
  run_dpo_grid.sh, run_dpo_more_seeds.sh  the grid of GPU runs
  check_cache.py                          checks the cached scorer of the GPU study (GPU)
  bootstrap_casa.py, multiseed_casa.py    bootstrap intervals; further partitions
  assemble_summary.py, make_figures.py    results2/summary.json; Figures 6.9 to 6.13
src3/                    live RAG, online decisions, robustness checks
  rag_lib.py, rag_build.py                knowledge base, retrievers, sessions, planted documents
  rag_detect.py, rag_paired.py            defences in live sessions; paired comparisons
  rag_asr.py, rag_asr_table.py            served models in the live pipeline (GPU); the table of outcomes
  online_eval.py                          decisions at the end and after every turn
  xai_robustness.py, pace_mechanism.py    checks on the audit; the pacing mechanism
  make_figures3.py                        Figures 6.14 to 6.16
scripts/                 served-model and GPU scripts
  precompute_defence.py, measure_asr_hf.py   attack success against served models
  measure_asr_ollama.py                      alternative protocol, behind no reported number
  train_dpo_gpu.py                           untested alternative set-up, behind no reported number

results_v1_original/     the run behind the printed Tables 6.1 to 6.11
results/                 a later re-run of the same pipeline
results2/                extended experiments (Tables 6.13 to 6.17, intervals)
results3/                live RAG, online decisions, robustness checks (Tables 6.18 to 6.21)
figures/, figures2/, figures3/   the images used in the report
```

Two words in the docstrings need a gloss. "The thesis" means the report named above. "Supervisor comment 4" and similar are the numbered review comments on an earlier draft that the scripts in `src2/` and `scripts/` were written to answer; the numbers mean nothing outside that review.

## Installation

Python 3.13 was used. The quick start below needs no installation at all for `tools/check_report.py`, and only numpy, scipy and matplotlib for the other commands.

```bash
git clone https://github.com/YOUR-ACCOUNT/mtpi-casa.git
cd mtpi-casa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that were installed when the later experiments ran. The core pipeline in `src/` runs on a CPU. The scripts that load pretrained models need an NVIDIA GPU, because they name the device `cuda` directly: `src2/dpo_gpu_study.py`, `src2/check_cache.py`, `scripts/measure_asr_hf.py`, `scripts/train_dpo_gpu.py` and `src3/rag_asr.py`. `src3/rag_build.py` uses a GPU for the dense retriever if one is present and a CPU otherwise. The experiments here ran on one Tesla T4 with 15 GB of memory. Run every command from the repository root: the scripts use relative paths such as `data/mtpi_bench.jsonl`.

## Getting the data

Nothing in this repository contains benchmark text. `data/README.md` has the details and checksums. In short:

```bash
# 1. the public prompt collection, then the benchmark (a few seconds)
git clone https://github.com/verazuo/jailbreak_llms
python src/build_bench.py

# 2. SQuAD v1.1, only for the live RAG study
mkdir -p data/squad
curl -L -o data/squad/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
curl -L -o data/squad/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

Models are downloaded from the Hugging Face hub on first use. None of them needed an access request when this was written. Set `HF_HOME` to choose the cache folder; the cache on the experiment machine takes about 20 GB.

| Model | Used by |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | served-model studies |
| `Qwen/Qwen2.5-1.5B-Instruct` | served-model studies, alignment study |
| `Qwen/Qwen2.5-3B-Instruct` | served-model studies, loaded in 4-bit |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | served-model studies |
| `microsoft/Phi-3.5-mini-instruct` | served-model study of Table 6.13, loaded in 4-bit |
| `sentence-transformers/all-MiniLM-L6-v2` | dense retriever of the RAG study |

## Quick start with only the shipped files

These commands need no benchmark, no model and no GPU.

```bash
python tools/check_report.py                  # the fields behind the tables, original run
python tools/check_report.py --run results    # the same from the later re-run
python src3/rag_paired.py                     # paired tests from the per-session statistics
python src3/rag_asr_table.py                  # rebuilds results3/rag_asr.json
python src2/merge_hardening.py                # rebuilds results2/casa_fix.json from the per-seed files
python src2/make_figures.py                   # redraws figures2/
python src3/make_figures3.py                  # redraws figures3/
```

The last five overwrite shipped files with recomputed ones. The JSON files should come out unchanged, which `git status` will confirm. The images can differ in their bytes on another matplotlib version without differing to the eye. Everything else needs the benchmark, and most of it needs the saved detector: see `docs/REPRODUCE.md`.

## Result folders

- `results_v1_original/` is the original run of the core pipeline. Tables 6.1 to 6.11 of the report are printed from it.
- `results/` is a later re-run of `src/eval_casa.py`, `src/adv_train.py` and `src/recalib.py`, made with the same code and seeds to rebuild the saved detector and meta-classifiers. Its new files are `casa_main.json`, `casa_main.csv`, `adv_train.json`, `adaptive_final.json` and the three logs whose names end in `_rerun`. The other files are copies of the original ones. The extended experiments load the models of this re-run.
- `results2/` holds the extended experiments: served models, hardening, composed attacks, the attribution audit, the alignment study on Qwen2.5-1.5B-Instruct, bootstrap intervals and further partitions. `results2/summary.json` collects them.
- `results3/` holds the live RAG study, the offline and online evaluation, and the robustness checks on the audit.

The two runs of the core pipeline give identical figures for the stateless, pooling and accumulator-only rows. The linear meta-classifier differs in the fourth decimal of AUC (0.9079 against 0.9077), and the adversarially trained linear model differs by one conversation under four of the attacker strategies. The gradient-boosted models differ by more: full CASA has recall 0.515 in the re-run against 0.561 in the original run, and the ablation changes are +0.0017, +0.0007, -0.0029, -0.0033 and -0.0124 against +0.0054, -0.0033, -0.0052, -0.0046 and -0.0154. The ablation differences are therefore within run-to-run variation. The cause of the difference between the runs was not established; the library versions of the original run were not recorded.

The operating thresholds of the adversarially trained models are the `thresholds` entry of `adaptive_final.json`, not the thresholds stored with the pickled models.

Log files are shipped as written, with one change: absolute paths of the experiment machine inside error traces were replaced by relative paths when the package was assembled.

## Seeds

All seeds are fixed, but they are not all the same.

| Seed | Used for |
|---|---|
| 42 | corpus sampling, partition, model fitting, cross-validation, augmentation of training conversations, bootstrap resampling, permutation in the audit |
| 7 | conversation construction in `src/build_bench.py`; batch sampling in `src/run_dpo.py`; the composed set; question assignment in the RAG sessions (7, 8 and 9 for the three partitions) |
| 43 | augmentation of validation conversations in `src/adv_train.py` |
| 99 | test-time attack rewrites behind Tables 6.7 to 6.9, 6.14 and 6.21 |
| 5 | the transformed-benign controls |
| 0 and 1 | response assignment and random pair selection in `src/run_dpo.py` |
| 11 | the fragmented set of `src2/composed_attacks.py`; appended turns in `src3/online_eval.py` |
| 1 to 4 | further partitions (`src2/multiseed_casa.py`) |
| 0 to 2 | training seeds of the hardening experiment (`src2/casa_hardening.py`) |
| run argument | `src2/dpo_gpu_study.py --seed`; the seeds that were run are listed in `results2/summary.json` |

The served-model scripts decode greedily and draw nothing at random.

## What is not shipped, and how to rebuild it

| Not shipped | Why | Rebuilt by |
|---|---|---|
| `data/mtpi_bench.jsonl` | jailbreak text from a public corpus | `python src/build_bench.py` |
| the source corpus, SQuAD | public, not ours to redistribute | see "Getting the data" |
| `results/stack.pkl` (188 MB), `results/casa_meta.pkl`, `results/casa_meta_linear.pkl` | pickled models | `python src/eval_casa.py` |
| `results/casa_at.pkl` | pickled models | `python src/adv_train.py`; then `python src/recalib.py` for the thresholds |
| `results/turn_scores.json`, `results/window_scores.json` | score caches | `python src/eval_casa.py` |
| `results2/dpo_pool.json` | holds every source prompt | `python src2/dpo_prep.py` |
| `results3/rag/sessions_*.json`, `results3/rag/wiki_emb.npy` | benchmark text; embedding cache | `python src3/rag_build.py` |
| `results3/rag/score_cache_*.joblib`, `results3/rag/casa_rag_*.pkl` | score caches; pickled models | `python src3/rag_detect.py --retriever bm25` and `--retriever dense` |
| model outputs | never stored: the scripts keep verdicts and rates only | |
| LoRA adapter weights | `src2/dpo_gpu_study.py` stores evaluation results only | re-run the study |

A rebuild will not reproduce the gradient-boosted rows of the printed tables exactly, for the reason given under "Result folders". Expect the stateless, pooling, accumulator-only and linear CASA rows to match to three decimals. Pickles are tied to the scikit-learn version that wrote them, and the scripts silence warnings, so use the pinned version.

## Known limitations

These come from an audit of the report against this code, made before release. Items marked "kept" would change the benchmark or every downstream number if fixed, so they are documented, not changed.

Benchmark

- Length is matched between the classes; platform of origin is not. 1,030 of the 1,151 benign prompts come from prompt-sharing websites, against 383 of the 1,151 attacks. Within one platform the detectors are weaker than on the mixed test set: AUC 0.833 for the per-turn filter and 0.875 for CASA on the 271 website conversations, and 0.724 and 0.757 on the 188 Discord and Reddit conversations.
- Lead-in turns and retrieval filler are drawn from the whole benign pool before the partition exists, and near-duplicate prompts are not removed. 86 of the 2,292 test turns equal a training turn, and 41 of the 461 test prompts share their first 60 words with a training prompt. Connector phrases are added to directly delivered turns only. Kept.
- Labels come from which source file a prompt was in. They were not reviewed by hand.
- The retrieval channel of the benchmark is a constructed document envelope. Only `src3/` uses a working retrieval pipeline.

Detector and conversation layer

- The meta-classifier is fitted on training conversations whose turn scores are in-sample for the turn detector (turn-level AUC 0.988 on training turns, 0.878 on test turns). Kept.
- The accumulator's alarm value is not tuned: the selection criterion does not depend on it. The selected leak (0.80) and user prior (0.45) lie at the edge of the search grid. Kept.
- Two of the 31 features, `high_rate` and `frac_high`, are the same formula. Kept.
- The ablation is one run. Its "no trajectory" row also removes the standard deviation of the turn scores.
- All results rest on one test partition of 461 conversations, 237 of them attacks. One point of recall is about 2.4 conversations. The further partitions are re-splits of the same corpus and share 17% to 22% of their test prompts.
- `src/eval_casa.py` reuses cached scores without checking which detector produced them. Delete `results/stack.pkl`, `results/turn_scores.json` and `results/window_scores.json` together.

Adaptive evaluation and adversarial training

- Tables 6.2 to 6.10 decide once, at the end of the conversation. Under that rule, appending one benign turn after a complete attack lowers CASA's detection from 57.0% to 2.1%. A filter that decides after every turn is not affected. Read the offline figures as rankings of complete conversations, not as what a deployed filter would block.
- The filler text that the attacker inserts is taken from the benign training prompts, which are in-sample for the turn detector.
- The transformed-benign control re-creates its random generator for every conversation, so every benign conversation receives the same filler sequence (`src/adv_train.py`, `src/asym_control.py`, `src2/casa_hardening.py`). `src3/online_eval.py` uses one generator for the whole control.
- `results/casa_at.pkl` stores thresholds set on augmented validation data (0.9693 and 0.99943). The reported operating point uses the recalibrated ones (0.9869 and 0.99980). With the stored thresholds 16 and 14 of 224 benign test conversations are flagged, against 11 and 10.
- Table 6.10 mixes the two: its transformed-benign rows were computed before recalibration and its clean row after.
- `src/eval_adaptive.py` and the two `adaptive.csv` files are an earlier seed-42 run against an earlier model. No printed table comes from them.
- In `src2/casa_hardening.py` the composed augmentation rewrites lead-in turns and inserted filler after its first step and drops the retrieved channel of some conversations, and its baseline is not the CASA-AT of `src/adv_train.py`. The training seeds change the augmentation and the classifier only; the test attacks are the same at every seed, so the intervals show training randomness and not sampling variation.
- The composed set of `src2/composed_attacks.py` is built from templates: one of four preambles replaces the lead-in turns, and one of six back-reference phrases is put in front of each later turn, whose original text stays. Benign conversations are not rebuilt, so there is no benign control.

Served models

- The defences in Table 6.13 decide after every turn at thresholds calibrated for one decision per conversation, so they are not at matched benign block rates (4.5%, 10.7% and 11.2%). Attack success has not been measured again at the recalibrated thresholds of `results3/online_eval.json`.
- "Success" there means that no refusal marker was found in the first 400 characters of the reply. It does not show that the injected instruction was obeyed. Only aggregate rates were stored.
- Qwen2.5-3B and Phi-3.5-mini were loaded in 4-bit, and their stored parameter counts are the packed counts.
- The detection layer and the aligned model were never run together. The served models carry no adapters.
- `scripts/measure_asr_ollama.py` uses a different protocol and replays attacks from all partitions. `scripts/train_dpo_gpu.py` selects pairs without a held-out partition and has no evaluation script. Neither produced a reported number.

Alignment

- In `src/run_dpo.py` the policy sees at most the first 86 tokens of a prompt, its vocabulary is built from all three partitions, the three arms start from different random adapter initialisations, and each arm was run once. The 94.4% agreement between detector and ground truth is measured on training prompts; on test prompts it is 83.5%.
- In the study on Qwen2.5-1.5B-Instruct, "refused" means that the refusal margin exceeds the base model's median margin on validation prompts, because the plain rule saturates. The detector-selected arms use the same 600 pairs at every seed, so a seed changes only the training randomness. Few seeds were run, and the intervals are wide. The language-modelling loss on benign prompts rises after alignment (the `nll_mean` fields of `results2/summary.json`).
- A repeated CPU study was started and stopped by a file error before it wrote a result. Its script and log are not shipped and nothing from it is reported.

Attribution audit

- The audit describes one fitted model. The accumulator features carry 52.9% of the attribution in the deployed model, which is fitted on unscaled features, and 14.9% after standardising. Removing them costs 0.0009 AUC. The two largest features correlate at 0.968 (`results3/xai_robustness.json`). An earlier version of `src2/xai_audit.py` wrote a `note` and a `top_group_targeted_by` field that stated a conclusion these checks do not support; a result file written before the correction may still carry them, and they should be ignored.

Live RAG

- One knowledge base of encyclopaedic text, two retrievers, no re-ranking and no query rewriting. The attacker knows the user's questions in advance. Planted documents are written by fixed rules and are not optimised against the retriever. The planted instruction asks for a harmless code, which measures obedience and not harm. The served models are small, and not every model of Table 6.13 completed this protocol on the available GPU.
- The turn detector was trained on jailbreak-style payloads. A plainly worded planted instruction is flagged in 0.4% to 2.5% of sessions, depending on the defence.

Reproducibility

- The scripts overwrite files in the result folders without asking, and later scripts load those files. Work in a copy if you want to keep the shipped results.

## Safety note

This is defensive research. The benchmark is built from jailbreak prompts that were already public in a research corpus (Shen et al., 2024). The multi-turn and retrieval variants rearrange that text; no new attack technique is proposed. The benchmark is distributed as a build script, and no prompt text, session file or model output is included here. Keep it that way if you fork the repository: `.gitignore` excludes the files that contain prompt text.

The instruction planted in the RAG study is harmless. It asks the model to include a short code such as `ZX-1234` in its reply, so that obedience can be checked without producing harmful content. No attack was run against any hosted or third-party system. All models were downloaded and run on the experiment machine.

## Licence

The code and the result files in this repository are released under the MIT licence (see `LICENSE`). The source corpus, SQuAD and the models keep their own licences: the prompt collection of Shen et al. is under MIT, SQuAD v1.1 under CC BY-SA 4.0, and each model under the licence on its hub page.

## Citation

`CITATION.cff` holds the citation data. In plain form:

Mukherjee, R. (2026) *Conversation-Level Defence Against Multi-Turn Prompt Injection in Locally Deployed Large Language Models: Sequential Evidence Accumulation, Provenance Separation and Detector-Supervised Preference Alignment*. Extended research report. Not peer reviewed.

If you use the benchmark, please also cite the source of its prompts: Shen, X., Chen, Z., Backes, M., Shen, Y. and Zhang, Y. (2024) '"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models', ACM CCS 2024, doi 10.1145/3658644.3670388. The RAG study uses SQuAD: Rajpurkar, P., Zhang, J., Lopyrev, K. and Liang, P. (2016), EMNLP 2016, doi 10.18653/v1/D16-1264.
