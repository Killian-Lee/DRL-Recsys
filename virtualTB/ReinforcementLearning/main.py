from __future__ import annotations

import argparse
import random
from collections import namedtuple
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import virtualTB  # noqa: F401

from virtualTB.ReinforcementLearning.ddpg import DDPG

Transition = namedtuple("Transition", ("state", "action", "mask", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension, dtype=np.float32) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension, dtype=np.float32) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPG policy on VirtualTB-v0.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-episode", type=int, default=5)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"))
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python 3.11 environment.")
    return device_arg


def evaluate(agent: DDPG, env: gym.Env, episodes: int) -> tuple[float, float]:
    total_reward = 0.0
    total_steps = 0
    for _ in range(episodes):
        state, _ = env.reset()
        while True:
            state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            action = agent.select_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            total_reward += reward
            total_steps += 1
            state = next_state
            if terminated or truncated:
                break
    avg_reward = total_reward / episodes
    avg_ctr = total_reward / max(total_steps, 1) / 10.0
    return avg_reward, avg_ctr


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    env = gym.make("VirtualTB-v0")
    eval_env = gym.make("VirtualTB-v0")
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed)

    agent = DDPG(
        gamma=args.gamma,
        tau=args.tau,
        hidden_size=args.hidden_size,
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device=device,
    )

    memory = ReplayMemory(args.replay_size)
    ounoise = OUNoise(env.action_space.shape[0])

    history_path = args.output_dir / "metrics.csv"
    with history_path.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write("episode,train_reward,eval_reward,eval_ctr,total_steps\n")

        total_numsteps = 0
        for i_episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed if i_episode == 0 else None)
            state = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            episode_reward = 0.0
            ounoise.reset()

            while True:
                action = agent.select_action(state, ounoise)
                next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
                done = terminated or truncated
                total_numsteps += 1
                episode_reward += reward

                action_tensor = action.detach().clone().to(dtype=torch.float32)
                mask = torch.tensor([not done], dtype=torch.float32)
                next_state_tensor = torch.from_numpy(
                    np.asarray(next_state, dtype=np.float32)
                ).unsqueeze(0)
                reward_tensor = torch.tensor([reward], dtype=torch.float32)

                memory.push(state, action_tensor, mask, next_state_tensor, reward_tensor)
                state = next_state_tensor

                if done:
                    break

            if len(memory) >= args.batch_size:
                for _ in range(args.updates_per_episode):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))
                    agent.update_parameters(batch)

            eval_reward = float("nan")
            eval_ctr = float("nan")
            if (i_episode + 1) % args.eval_interval == 0 or i_episode == 0:
                eval_reward, eval_ctr = evaluate(agent, eval_env, args.eval_episodes)
                print(
                    f"Episode {i_episode + 1}/{args.episodes} "
                    f"train_reward={episode_reward:.2f} eval_reward={eval_reward:.2f} "
                    f"eval_ctr={eval_ctr:.4f} steps={total_numsteps} device={device}"
                )
                if args.save_checkpoints:
                    agent.save_model(
                        env_name="VirtualTB-v0",
                        suffix=f"episode_{i_episode + 1}",
                        actor_path=args.output_dir / f"actor_episode_{i_episode + 1}.pt",
                        critic_path=args.output_dir / f"critic_episode_{i_episode + 1}.pt",
                    )

            metrics_file.write(
                f"{i_episode + 1},{episode_reward:.6f},{eval_reward:.6f},{eval_ctr:.6f},{total_numsteps}\n"
            )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
