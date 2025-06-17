from .wrappers import *


class Env:
    def __len__(self):
        return 0  # Return positive integer for batched envs.

    def __bool__(self):
        return True  # Env is always truthy, despite length zero.

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"len={len(self)}, "
            f"obs_space={self.obs_space}, "
            f"act_space={self.act_space})"
        )

    @property
    def observation_space(self):
        # The observation space must contain the keys is_first, is_last, and
        # is_terminal. Commonly, it also contains the keys reward and image. By
        # convention, keys starting with log_ are not consumed by the agent.
        raise NotImplementedError("Returns: dict of spaces")

    @property
    def action_space(self):
        # The observation space must contain the keys action and reset. This
        # restriction may be lifted in the future.
        raise NotImplementedError("Returns: dict of spaces")

    def step(self, action):
        raise NotImplementedError("Returns: dict")

    def render(self):
        raise NotImplementedError("Returns: array")

    def close(self):
        pass


def wrap_env(env):
    env = EpisodeMonitor(env)
    env = SinglePrecision(env)
    return env
