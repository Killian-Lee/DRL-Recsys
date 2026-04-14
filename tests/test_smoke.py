import gymnasium as gym
import numpy as np

import virtualTB  # noqa: F401


def test_import_and_make_env():
    env = gym.make("VirtualTB-v0")
    assert env.observation_space.shape == (91,)
    assert env.action_space.shape == (27,)
    env.close()


def test_reset_and_step_contract():
    env = gym.make("VirtualTB-v0")
    observation, info = env.reset(seed=123)
    assert observation.shape == (91,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert "CTR" in info

    next_observation, reward, terminated, truncated, step_info = env.step(
        env.action_space.sample()
    )
    assert next_observation.shape == (91,)
    assert next_observation.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.observation_space.contains(next_observation)
    assert "page_index" in step_info
    env.close()


def test_terminal_step_does_not_autoreset():
    env = gym.make("VirtualTB-v0")
    observation, _ = env.reset(seed=7)
    assert observation[-1] == 0.0

    env.unwrapped.leave_page = 1
    terminal_observation, _, terminated, truncated, _ = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert terminal_observation[-1] == 1.0

    reset_observation, _ = env.reset(seed=8)
    assert reset_observation[-1] == 0.0
    env.close()
