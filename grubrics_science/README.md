# GRubrics Science

Train (with RL) a model called GRubrics whose ONLY job is to GENERATE scoring rubrics for open-ended science research questions.

## Overview

GRubrics is trained using functional alignment: a generated rubric is "good" if, when used by a fixed Judge to score multiple candidate answers for the same question, it produces scores/rankings similar to what the GOLDEN rubric produces (using the SAME fixed Judge).

## Project Structure

```
grubrics_science/
  configs/          # Configuration files (YAML)
  llm/              # LLM client abstractions and prompts
  tasks/            # Dataset loaders and task definitions
  judge/            # Fixed Judge wrapper (evaluates answers with rubrics)
  rewards/          # Reward computation (alignment metrics, length penalty)
  rl/               # RL training loop and model wrapper
  utils/            # Utilities (IO, logging, seeding)
  data/             # Cache outputs
```

## Usage

### Precompute Mode

Generate and cache answers + gold scores:

```bash
python -m grubrics_science.rl.train_grpo --mode=precompute
```

### Train Mode

Train the GRubrics model:

```bash
python -m grubrics_science.rl.train_grpo --mode=train
```

## Configuration

Edit `configs/default.yaml` to adjust:
- K: number of answers per question (default: 8)
- k_train: subset size for training (default: 4)
- M: number of rubrics generated per episode (default: 6)
- lambda_len: length penalty coefficient (default: 0.01)

## Dataset

Uses FrontierScience-Research dataset at `data/frontierscience-research/test.jsonl`.

Each record has:
- `problem`: the full question prompt
- `answer`: the GOLDEN rubric (target/reference)

## Installation

```bash
pip install -r grubrics_science/requirements.txt
```

## Quick Start

### 1. Precompute answers and gold scores

```bash
python -m grubrics_science.rl.train_grpo --mode=precompute
```

This will:
- Load questions from the dataset
- Generate K=8 diverse answers per question using Answer Policy
- Score each answer with the golden rubric using Judge
- Cache results in `grubrics_science/data/cache/precompute_cache.jsonl`

### 2. Train GRubrics

```bash
python -m grubrics_science.rl.train_grpo --mode=train
```

This will:
- Load cached answers and gold scores
- Generate M=6 rubrics per question using GRubrics model
- Evaluate rubrics with Judge (batched)
- Compute rewards based on alignment with gold scores
- Update GRubrics model with REINFORCE

## Smoke Test

Run a quick smoke test:

```bash
python -m grubrics_science.test_smoke
```

## Configuration

Edit `grubrics_science/configs/default.yaml` to adjust:
- `K`: number of answers per question (default: 8)
- `k_train`: subset size for training (default: 4)
- `M`: number of rubrics generated per episode (default: 6)
- `lambda_len`: length penalty coefficient (default: 0.01)
- `alignment_metric`: "spearman", "pairwise", or "pearson" (default: "spearman")

## Architecture

- **Answer Policy**: Fixed Azure OpenAI model that generates diverse answers
- **Judge**: Fixed Azure OpenAI model that evaluates answers with rubrics (batched)
- **GRubrics**: Trainable Qwen model that generates rubrics (trained with RL)

## Notes

- The Judge and Answer Policy are FIXED and never trained
- Only GRubrics is trained using functional alignment
- Reward is computed from multiple answers per question (not single pair)
- Judge evaluation is batched: M rubrics evaluated in one call per answer

