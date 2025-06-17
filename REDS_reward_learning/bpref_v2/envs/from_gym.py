import gym
import numpy as np

from .core import Env, wrap_env


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [k for k, v in env.obs_space.items() if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class FromGym(Env):
    def __init__(self, env, seed=42, **kwargs):
        if isinstance(env, str):
            self._env = gym.make(env, **kwargs)
        else:
            assert not kwargs, kwargs
            self._env = env

        self._env = wrap_env(self._env)
        self._env.seed(seed)
        self._env.action_space.seed(seed)
        self._env.observation_space.seed(seed)
        self._shape = self._env.render().shape

    @property
    def observation_space(self):
        spaces = {
            "state": self._env.observation_space,
            "image": gym.spaces.Box(0, 255, self._shape, np.uint8),
        }
        return spaces

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        state, rew, done, info = self._env.step(action)
        obs = {"image": self._env.render("rgb_array"), "state": state}
        return obs, rew, done, info

    def render(self):
        image = self._env.render("rgb_array")
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
