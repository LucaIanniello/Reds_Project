import os

import gym
import numpy as np
from gym import spaces

from .core import Env


class MetaWorldState(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, state_key: str = "proprio"):
        super().__init__(env)
        self.state_key = state_key

        state_space = spaces.Box(
            low=self.state(self.observation_space[self.state_key].low),
            high=self.state(self.observation_space[self.state_key].high),
        )

        self.observation_space = spaces.Dict(
            {
                key: state_space if key == self.state_key else space
                for key, space in self.observation_space.spaces.items()
            }
        )

    @staticmethod
    def state(state):
        return np.concatenate((state[..., :4], state[..., 18:22]), axis=-1)

    def observation(self, observation):
        """Removes object information from the state.
        Args:
            observation: The observation to remove object information from
        Returns:
            The updated observation
        """
        observation.update(
            {
                self.state_key: self.state(observation[self.state_key]),
            }
        )
        return observation


class MetaWorld(Env):
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera="corner2",
    ):
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat

        self._prev_obs = None
        self._camera = camera
        self._max_episode_steps = self._env.max_path_length

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "state": self._env.observation_space,
        }
        return spaces

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        action = self._env.action_space
        return action

    def step(self, action):
        assert np.isfinite(action).all(), action
        try:
            reward = 0.0
            success = 0.0
            for _ in range(self._action_repeat):
                state, rew, done, info = self._env.step(action)
                success += float(info["success"])
                reward += rew or 0.0
            success = min(success, 1.0)
            assert success in [0.0, 1.0]
            done = success
            obs = {
                "image": self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera),
                "state": state,
            }
            self._prev_obs = obs
        except ValueError:
            done = True
            obs = self._prev_obs
            reward = 0.0
            info = {"success": False, "unscaled_reward": 0.0}
        return obs, reward, done, info

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        # return state
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera),
            "state": state,
            "success": False,
        }
        return obs
