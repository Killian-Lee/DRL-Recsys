# VirtualTaobao

VirtualTaobao is a virtual online-retail environment trained from real Taobao interaction data and exposed as a `Gymnasium` environment for recommendation and reinforcement-learning research.

This repository now targets a modern workflow:

- Python `3.11` as the main development version
- Python `3.12` as a secondary supported version
- `Gymnasium` as the primary environment API
- `NumPy 2.x` and `PyTorch 2.x`
- `uv sync` and `uv run` as the recommended install and execution path

The environment package exposes `VirtualTB-v0`, plus two legacy research examples:

- `virtualTB/ReinforcementLearning/main.py`: DDPG baseline
- `virtualTB/SupervisedLearning/main.py`: supervised-learning baseline

Both examples are now runnable as modern CLI entry points:

- `virtualtb-train-rl`
- `virtualtb-train-sl`

## Project layout

```text
virtualTB/
├─ __init__.py
├─ smoke.py
├─ utils.py
├─ data/
│  ├─ action_model.pt
│  ├─ generator_model.pt
│  └─ leave_model.pt
├─ envs/
│  └─ virtualTB.py
├─ model/
│  ├─ ActionModel.py
│  ├─ LeaveModel.py
│  └─ UserModel.py
├─ ReinforcementLearning/
│  ├─ ddpg.py
│  └─ main.py
└─ SupervisedLearning/
   ├─ dataset.txt
   └─ main.py
```

## Install with uv

```bash
uv sync --python 3.11
```

For development tools:

```bash
uv sync --python 3.11 --extra dev
```

## Minimal smoke run

```bash
uv run python -m virtualTB.smoke
```

Or:

```bash
uv run virtualtb-smoke
```

## Minimal Gymnasium example

```python
import gymnasium as gym
import virtualTB

env = gym.make("VirtualTB-v0")
obs, info = env.reset(seed=0)

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Environment contract

- Observation shape: `(91,)`
- Action shape: `(27,)`
- Observation dtype: `float32`
- Action dtype: `float32`
- `reset()` returns `(obs, info)`
- `step()` returns `(obs, reward, terminated, truncated, info)`

The observation is composed of:

- `88` static user features
- `2` dynamic feedback features from the previous interaction
- `1` page index feature

The action is a `27`-dimensional continuous vector representing item-attribute weights.

## Run tests

```bash
uv run pytest -q
```

## Train models

Supervised baseline:

```bash
uv run virtualtb-train-sl --epochs 50 --batch-size 256 --device cpu
```

RL baseline:

```bash
uv run virtualtb-train-rl --episodes 200 --eval-interval 20 --device cpu
```

For the full training notes, including GPU-machine requirements, see [docs/TRAINING.md](docs/TRAINING.md).

## Run module examples

```bash
uv run python -m virtualTB.ReinforcementLearning.main
uv run python -m virtualTB.SupervisedLearning.main
```

## Migration notes

This repository originally depended on legacy `gym` and old NumPy behavior. During modernization:

- the package entry point was migrated to `Gymnasium`
- the environment API was updated to the modern reset/step contract
- package metadata was moved to `pyproject.toml`
- the recommended workflow became `uv sync` / `uv run`

Legacy validation before modernization showed:

- import and `reset()` worked with `gym==0.26.2`
- `step()` failed under NumPy 2.x due to `numpy.bool8` usage in legacy `gym`
- only installing `gymnasium` was not enough because the original package hard-imported `gym`

## Reference

Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, and An-Xiang Zeng.
[Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning](https://arxiv.org/abs/1805.10000).
AAAI 2019.
