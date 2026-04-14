from __future__ import annotations

import gymnasium as gym
import numpy as np

import virtualTB  # noqa: F401


def main() -> None:
    env = gym.make("VirtualTB-v0")
    observation, info = env.reset(seed=0)
    print("reset", observation.shape, sorted(info.keys()))

    episode_return = 0.0
    for step in range(10):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        print(
            f"step={step} reward={reward:.1f} terminated={terminated} "
            f"truncated={truncated} page={info['page_index']}"
        )
        if terminated or truncated:
            break

    print(f"episode_return={episode_return:.1f}")
    env.close()


if __name__ == "__main__":
    main()
