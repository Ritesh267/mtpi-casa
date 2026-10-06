# data/

This folder is empty on purpose. The benchmark contains jailbreak prompts from a public research corpus, and the retrieval study uses a public question-answering dataset. Neither is redistributed here. Both are fetched or rebuilt with the commands below, run from the repository root.

## 1. MTPI-Bench (needed by almost every script)

```bash
git clone https://github.com/verazuo/jailbreak_llms
python src/build_bench.py            # writes data/mtpi_bench.jsonl
```

`build_bench.py` reads two files of the clone:

- `jailbreak_llms/data/prompts/jailbreak_prompts_2023_12_25.csv`
- `jailbreak_llms/data/prompts/regular_prompts_2023_12_25.csv`

If the clone is somewhere else, pass its path: `python src/build_bench.py --corpus /path/to/jailbreak_llms`.

What to expect:

| Item | Value |
|---|---|
| Conversations | 2,302 (1,151 attack, 1,151 benign) |
| Turns | 11,756 |
| File size | 13,463,717 bytes |
| SHA-256 of `data/mtpi_bench.jsonl` | `117af77633925475929eb834e634b02248300b59b1eb574178f1adce3985dd3a` |
| Run time | a few seconds |

The build is deterministic. On 1 October 2026 it reproduced the file used for the report byte for byte from the two CSV files of the public repository. If your checksum differs, the corpus or your pandas and numpy versions differ from the ones in `requirements.txt`, and the numbers you obtain downstream will not match the shipped result files.

The corpus was released with Shen et al. (2024), "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models, ACM CCS 2024. Its repository carries an MIT licence. Please cite that paper if you use the benchmark.

## 2. SQuAD v1.1 (needed only by the live retrieval study in `src3/rag_*.py`)

```bash
mkdir -p data/squad
curl -L -o data/squad/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
curl -L -o data/squad/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

| File | Size in bytes | SHA-256 |
|---|---|---|
| `data/squad/train-v1.1.json` | 30,288,272 | `3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955` |
| `data/squad/dev-v1.1.json` | 4,854,279 | `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9` |

The two files give 20,958 distinct Wikipedia paragraphs and 98,169 questions. The pipeline uses the 97,750 questions that have four words or more. SQuAD is distributed under CC BY-SA 4.0 (Rajpurkar et al., 2016).

## 3. Models

No model file belongs in this folder. The scripts that use pretrained models download them from the Hugging Face hub on first use. The README lists the model names. Set `HF_HOME` to choose where the cache goes.

## Do not commit what you build here

`.gitignore` excludes everything in `data/` except this file, and it excludes the clone `jailbreak_llms/`. Keep it that way: the benchmark file and the session files derived from it contain jailbreak text.
