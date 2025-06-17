import copy
from collections import deque
from typing import Callable, Union

import furniture_bench
import gym
import numpy as np

from bpref_v2.data.instruct import get_furniturebench_instruct


class ContextWindow(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(
        self,
        env: gym.Env,
        image_keys: Union[list, str],
        image_size: int,
        window_size: int,
        skip_frame: int,
        reward_model: Callable = None,
        tokenizer_fn: Callable = None,
        postprocess_fn: Callable = None,
        rtg: Union[int, None] = None,
    ):
        super().__init__(env)
        self.image_keys = image_keys
        self.image_size = image_size
        self.window_size = window_size
        self.skip_frame = skip_frame
        self.reward_model = reward_model
        self.tokenizer_fn = tokenizer_fn
        self.postprocess_fn = postprocess_fn

        self._use_rtg = (rtg != None and self.reward_model is not None)
        if self._use_rtg:
            self.rtg = self.postprocess_fn(rtg)
        self._reset_batch()
        self._batchify = lambda x: np.asarray(x)[None, ...]

    def _reset_batch(self):
        self._i = 0
        self._skills = 0
        target_keys = self.image_keys + ["action", "rtg", "timestep", "attn_mask", "instruct", "text_padding_mask"]
        dummy_batch = {key: deque([], maxlen=self.window_size) for key in target_keys}
        for ik in self.image_keys:
            dummy_batch[ik].extend([np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.window_size)
        dummy_batch["action"].extend([np.zeros((self.env.act_space.shape[0],), dtype=np.float32)] * self.window_size)
        dummy_batch["timestep"].extend([np.asarray(0).astype(np.int32)] * self.window_size)
        dummy_batch["attn_mask"].extend([np.asarray(0).astype(np.int32)] * self.window_size)
        dummy_batch["instruct"].extend([np.zeros((77,), dtype=np.int32)] * self.window_size)
        dummy_batch["text_padding_mask"].extend([np.zeros((77,), dtype=np.float32)] * self.window_size)
        if self._use_rtg:
            dummy_batch["rtg"].extend([np.asarray([self.rtg]).astype(np.int32)] * self.window_size)
        else:
            dummy_batch["instruct"].extend([np.zeros((77,), dtype=np.int32)] * self.window_size)
            dummy_batch["text_padding_mask"].extend([np.zeros((77,), dtype=np.float32)] * self.window_size)
        self.total_batch = {frame: copy.deepcopy(dummy_batch) for frame in range(self.skip_frame)}

    def _get_current_instruct(self):
        _, skill_complete = self.env._env.get_assembly_action()
        self._skills += skill_complete
        if self._use_rtg:
            current_instruct, text_padding_mask = self.tokenizer_fn(
                get_furniturebench_instruct(self.env._env.furniture_name, int(self._skills), output_type="all")
            )
            # shape: current_instruct: (20, 77) / text_padding_mask: (1, 77)
        else:
            current_instruct, text_padding_mask = self.tokenizer_fn(
                get_furniturebench_instruct(self.env._env.furniture_name, int(self._skills), output_type="one")
            )
           # shape: current_instruct: (1, 77) / text_padding_mask: (1, 77)
        return current_instruct.squeeze(), text_padding_mask.squeeze()

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        self._i += 1
        key = self._i % self.skip_frame
        batch = self.total_batch[key]
 
        if self._use_rtg:
            multimodal_reward = self.reward_model.get_reward(
                {
                    "image": {key: self._batchify(batch[key])for key in self.image_keys},
                    "action": self._batchify(batch["action"]),
                    "timestep": self._batchify(batch["timestep"]),
                    "attn_mask": self._batchify(batch["attn_mask"]),
                    "instruct": self._batchify(batch["instruct"]),
                }
            )
            multimodal_reward = np.asarray(multimodal_reward)
            # TODO: How to post-process this value.
            batch["rtg"].append(batch["rtg"][-1] - self.postprocess_fn(multimodal_reward))

        for ik in self.image_keys:
            batch[ik].append(obs[ik].squeeze())
        batch["action"].append(action.squeeze())
        batch["timestep"].append(np.asarray(self._i).astype(np.int32))
        batch["attn_mask"].append(np.asarray(1).astype(np.int32))

        current_instruct, current_text_padding_mask = self._get_current_instruct()
        if not self._use_rtg:
            batch["instruct"].append(current_instruct)
            batch["text_padding_mask"].append(current_text_padding_mask)
        else:
            batch["insturct"] = current_instruct
            batch["text_padding_mask"] = current_text_padding_mask

        batch = {key: self._batchify(val) for key, val in batch.items()}

        output_batch = {
            "image": {key: batch[key] for key in self.image_keys},
            "action": batch["action"],
            "timestep": batch["timestep"],
            "attn_mask": batch["attn_mask"],
            "rtg": batch["rtg"],
            "text_padding_mask": batch["text_padding_mask"],
            "instruct": batch["instruct"],
        }
        return output_batch, reward, done, info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._reset_batch()

        key = self._i % self.skip_frame
        batch = self.total_batch[key]
        for ik in self.image_keys:
            batch[ik].append(obs[ik].squeeze())
        current_instruct, current_text_padding_mask = self._get_current_instruct()
        if not self._use_rtg:
            batch["instruct"].append(current_instruct)
            batch["text_padding_mask"].append(current_text_padding_mask)
        else:
            batch["insturct"] = current_instruct
            batch["text_padding_mask"] = current_text_padding_mask

        batch = {key: self._batchify(val) for key, val in batch.items()}

        output_batch = {
            "image": {key: batch[key] for key in self.image_keys},
            "action": batch["action"],
            "timestep": batch["timestep"],
            "attn_mask": batch["attn_mask"],
            "rtg": batch["rtg"],
            "text_padding_mask": batch["text_padding_mask"],
            "instruct": batch["instruct"],
        }
        return output_batch


if __name__ == "__main__":
    from ..furniturebench import FurnitureBench

    class DummyRewardModel(object):
        def __init__(self):
            self._i = 0

        def get_reward(self, _):
            return 10

    env = FurnitureBench("FurnitureSimLegacy-v0", "one_leg")
    env = ContextWindow(
        env,
        image_keys=["color_image1", "color_image2"],
        image_size=224,
        window_size=4,
        skip_frame=16,
        rtg=5000,
        reward_model=DummyRewardModel(),
    )
    init = env.reset()
    timestep = 0
    for _ in range(100):
        timestep += 1
        res, rew, done, info = env.step(env.action_space.sample())
        print(f"[INFO]: {timestep} rtg {res['rtg']}")
        if done:
            break
