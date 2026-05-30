# Running the Pipeline

This repository uses `click` to provide a CLI for the training pipeline. All commands are run via `main.py`. We recommend using `uv` to handle dependencies.

## Installation

```bash
uv sync
```

## Running the stages

The pipeline is split into several stages, each of which uses a configuration file from the `configs/` directory.

### 1. Tokenizer Training and FOCUS
Trains a new tokenizer on a representative sample of data and performs lexical initialization of the model's vocabulary using the FOCUS method.

```bash
uv run python main.py tokenizer --config configs/tokenizer.yaml
```

*To test quickly, use `--dummy-data` flag:*
```bash
uv run python main.py tokenizer --config configs/tokenizer.yaml --dummy-data
```

### 2. Pruning
Prunes unused tokens from the model's vocabulary based on a calibration dataset to optimize the size and efficiency.

```bash
uv run python main.py prune --config configs/pruning.yaml
```

### 3. Data Preparation
Runs filtering (spam, SEO, etc.) and MinHash LSH deduplication on the pre-training dataset.

```bash
uv run python main.py data-prep --config configs/data_prep.yaml
```

### 4. Continual Pre-Training (CPT)
Continues the pre-training phase of the model using the prepared dataset.

```bash
uv run python main.py cpt --config configs/cpt.yaml
```

### 5. Supervised Fine-Tuning (General SFT)
Performs instruction tuning on the model using dialog datasets formatted in ChatML.

```bash
uv run python main.py sft --config configs/sft.yaml
```

### 6. Alignment (DPO/GRPO)
Aligns the model with human preferences using methods like DPO or GRPO (RLVR) with custom reward functions.

```bash
uv run python main.py alignment --config configs/alignment.yaml
```

### 7. Final Task SFT
Performs task-specific fine-tuning, optionally using PEFT/LoRA.

```bash
uv run python main.py task-sft --config configs/task_sft.yaml
```

## Running Tests

To run the full test suite and check coverage (requires `pytest` and `pytest-cov`):

```bash
uv run pytest --cov=src --cov-fail-under=90 --cov-report=term-missing
```
