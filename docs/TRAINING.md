# Training Guide

This repository currently supports training two kinds of models:

1. A reinforcement-learning policy that interacts with `VirtualTB-v0`
2. A supervised-learning baseline trained from `virtualTB/SupervisedLearning/dataset.txt`

The environment simulator itself is **not** fully trainable from this repository. The three environment weights under `virtualTB/data/` are pre-trained assets, and the original simulator-training pipeline is not included here.

## Python version

Use Python `3.11` only for now.

## Environment setup

```bash
uv sync --python 3.11 --extra dev
```

## CPU usage

For CPU-only training:

```bash
uv run virtualtb-train-sl --epochs 50 --batch-size 256 --device cpu
uv run virtualtb-train-rl --episodes 200 --eval-interval 20 --device cpu
```

## GPU usage

The code supports `--device cuda` for policy/baseline models, but whether CUDA is actually available depends on the PyTorch build in your rented machine.

Recommended GPU machine requirements:

- Python `3.11`
- NVIDIA GPU with a CUDA-compatible driver
- A CUDA-enabled PyTorch 2.x install matching the machine image

On a rented GPU box, verify this first:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `True`, you can run:

```bash
uv run virtualtb-train-sl --device cuda --save-checkpoints
uv run virtualtb-train-rl --device cuda --save-checkpoints
```

## Outputs

### Supervised learning

Default output directory:

```text
artifacts/sl/
```

Files:

- `metrics.csv`
- `model_final.pt`
- optional epoch checkpoints if `--save-checkpoints` is enabled

### Reinforcement learning

Default output directory:

```text
artifacts/rl/
```

Files:

- `metrics.csv`
- optional actor/critic checkpoints if `--save-checkpoints` is enabled

## What cannot be trained here

This repository does **not** include:

- the raw data and code used to train the simulator weights
- a public pipeline for retraining `generator_model.pt`
- a public pipeline for retraining `action_model.pt`
- a public pipeline for retraining `leave_model.pt`

So if you want to retrain the simulator itself, you will need to build a new data pipeline and new simulator-training code outside the current repository.
