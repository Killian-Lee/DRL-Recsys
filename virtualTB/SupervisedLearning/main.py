from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import virtualTB  # noqa: F401


def init_weight(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(88 + 3, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 27),
            nn.Tanh(),
        )
        self.model.apply(init_weight)

    def forward(self, x):
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised baseline on VirtualTB data.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("dataset.txt"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl"))
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python 3.11 environment.")
    return torch.device(device_arg)


def load_dataset(dataset_path: Path):
    features, labels, clicks = [], [], []
    with io.open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            features_l, labels_l, clicks_l = line.split("\t")
            features.append([float(x) for x in features_l.split(",")])
            labels.append([float(x) for x in labels_l.split(",")])
            clicks.append(int(clicks_l))
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(clicks, dtype=torch.float32),
    )


def evaluate(model: nn.Module, env: gym.Env, device: torch.device, episodes: int, seed: int) -> float:
    total_clicks = 0.0
    total_page = 0
    model.eval()
    with torch.no_grad():
        for i in range(episodes):
            features, _ = env.reset(seed=seed if i == 0 else None)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                predictions = model(
                    torch.tensor(features, dtype=torch.float32, device=device)
                ).cpu().numpy()
                features, clicks, terminated, truncated, _ = env.step(predictions)
                total_clicks += clicks
                total_page += 1
    model.train()
    return total_clicks / max(total_page, 1) / 10.0


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    clicks: torch.Tensor,
    env: gym.Env,
    args: argparse.Namespace,
    device: torch.device,
):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "metrics.csv"
    num_samples = len(clicks)

    with history_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["epoch", "loss", "ctr"])

        for epoch in range(args.epochs):
            indices = np.random.permutation(num_samples)
            total_loss = 0.0
            batch_num = (num_samples + args.batch_size - 1) // args.batch_size

            for batch_idx in range(batch_num):
                batch_indices = indices[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                m_features = features[batch_indices].to(device)
                m_labels = labels[batch_indices].to(device)
                m_clicks = clicks[batch_indices].to(device)

                y_labels = model(m_features)
                loss = torch.mean(m_clicks * ((y_labels - m_labels) ** 2).sum(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item()) / batch_num

            ctr = evaluate(model, env, device, args.eval_episodes, args.seed)
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss:.4f} ctr={ctr:.4f}")
            writer.writerow([epoch + 1, f"{total_loss:.6f}", f"{ctr:.6f}"])

            if args.save_checkpoints:
                checkpoint_path = args.output_dir / f"model_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), checkpoint_path)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, labels, clicks = load_dataset(args.dataset)
    env = gym.make("VirtualTB-v0")
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, features, labels, clicks, env, args, device)

    final_ctr = evaluate(model, env, device, args.eval_episodes, args.seed)
    final_model_path = args.output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final CTR: {final_ctr:.4f}")
    print(f"Saved final model to {final_model_path}")
    env.close()


if __name__ == "__main__":
    main()
