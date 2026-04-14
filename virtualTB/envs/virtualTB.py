from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from virtualTB.model.ActionModel import ActionModel
from virtualTB.model.LeaveModel import LeaveModel
from virtualTB.model.UserModel import UserModel
from virtualTB.utils import FLOAT


class VirtualTB(gym.Env[ObsType, ActType]):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in {None, "human"}:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        self.render_mode = render_mode
        self.n_user_feature = 88
        self.n_item_feature = 27
        self.max_c = 100

        self.obs_low = np.concatenate(
            (
                np.zeros(self.n_user_feature, dtype=np.float32),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        self.obs_high = np.concatenate(
            (
                np.ones(self.n_user_feature, dtype=np.float32),
                np.array([10.0, 9.0, float(self.max_c)], dtype=np.float32),
            )
        )
        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_item_feature,),
            dtype=np.float32,
        )

        self.user_model = UserModel().load()
        self.user_action_model = ActionModel().load()
        self.user_leave_model = LeaveModel().load()

        self.total_a = 0.0
        self.total_c = 0
        self.leave_page = 0
        self.episode_done = False
        self.cur_user = np.zeros(self.n_user_feature, dtype=np.float32)
        self.lst_action = np.zeros(2, dtype=np.float32)
        self.rend_action = self.lst_action.copy()

    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.cur_user,
                self.lst_action,
                np.array([self.total_c], dtype=np.float32),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)

    def _make_info(self) -> dict[str, float | int]:
        ctr = float(self.total_a / max(self.total_c, 1) / 10.0)
        return {
            "CTR": ctr,
            "total_clicks": float(self.total_a),
            "page_index": int(self.total_c),
            "leave_page": int(self.leave_page),
        }

    def _user_generator(self) -> tuple[np.ndarray, int]:
        user = self.user_model.generate().squeeze(0).detach().cpu().numpy().astype(np.float32)
        leave_page = int(self.user_leave_model.predict(FLOAT(user).unsqueeze(0)).item())
        return user, leave_page

    def seed(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
            np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            import torch

            torch.manual_seed(seed)
        return [seed]

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        super().reset(seed=seed)
        self.seed(seed)
        self.total_a = 0.0
        self.total_c = 0
        self.cur_user, self.leave_page = self._user_generator()
        self.lst_action = np.zeros(2, dtype=np.float32)
        self.rend_action = self.lst_action.copy()
        self.episode_done = False
        return self.state, self._make_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | int]]:
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape {action.shape} does not match expected {self.action_space.shape}."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        prediction = self.user_action_model.predict(
            FLOAT(self.cur_user).unsqueeze(0),
            FLOAT([[self.total_c]]),
            FLOAT(action).unsqueeze(0),
        )
        self.lst_action = prediction.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        self.rend_action = self.lst_action.copy()

        reward = float(self.lst_action[0])
        self.total_a += reward
        self.total_c += 1

        terminated = bool(self.total_c >= self.leave_page)
        truncated = bool(self.total_c >= self.max_c and not terminated)
        self.episode_done = terminated or truncated

        return self.state, reward, terminated, truncated, self._make_info()

    def render(self):
        if self.render_mode not in {None, "human"}:
            return None

        click_count, action_index = np.clip(self.rend_action, a_min=0.0, a_max=None)
        print("Current State:")
        print("\t", self.state)
        print("User's action:")
        print(
            "\tclick:%2d, done:%s, index:%2d"
            % (int(click_count), "True" if self.episode_done else "False", int(action_index))
        )
        print("Total clicks:", int(self.total_a))
        return None

    def close(self):
        return None
