from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from bpref_v2.utils.reward_model_loader import load_reward_model

tree_map = jax.tree_util.tree_map


class InvalidSequenceError(Exception):
    def __init__(self, message):
        super().__init__(message)


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


class DrsRewardModel:
    PRIVATE_LIKELIHOOD_KEY = "log_immutable_density"
    PUBLIC_LIKELIHOOD_KEY = "density"

    def __init__(
        self,
        task: str = "",
        model_name: str = "DRS",
        rm_path: str = "",
        camera_keys: str = "image",
        reward_scale: Optional[Union[Dict[str, Tuple], Tuple]] = None,
        window_size: int = 4,
        skip_frame: int = 1,
        reward_model_device: int = 0,
        encoding_minibatch_size: int = 32,
        use_task_reward: bool = False,
        use_scale: bool = False,
        clip_value: float = 1.0,
    ):
        self.domain, self.task = task.split("_", 1)
        self.reward_model = load_reward_model(rm_type=model_name, task_name=self.task, ckpt_path=Path(rm_path))
        self.device = jax.devices()[reward_model_device]

        self.model_name = model_name
        self.camera_keys = [camera_keys] if isinstance(camera_keys, str) else camera_keys
        self.reward_scale = reward_scale
        self.skip_frame = skip_frame
        self.window_size = window_size
        self.seq_len = self.window_size * self.skip_frame
        self.seq_len_steps = self.seq_len
        self.encoding_minibatch_size = encoding_minibatch_size
        self.use_task_reward = use_task_reward
        self.use_scale = use_scale
        self.clip = clip_value

        print(
            f"finished loading {self.__class__.__name__}:"
            f"\n\tseq_len: {self.seq_len}"
            f"\n\ttask: {self.task}"
            f"\n\tmodel: {self.model_name}"
            f"\n\tcamera_keys: {self.camera_keys}"
            f"\n\tseq_len_steps: {self.seq_len_steps}"
            f"\n\tskip_frame: {self.skip_frame}"
            f"\n\tuse_task_reward? {self.use_task_reward}"
            f"\n\tuse_scale? {self.use_scale}"
        )

        self._call_step = 0
        self._reset_step = 1000

    def __call__(self, seq, **kwargs):
        self._call_step += 1
        if self._call_step % self._reset_step == 0:
            jax.clear_caches()
        return self.process_seq(self.compute_reward(seq, **kwargs), **kwargs)

    def _reward_scaler(self, reward):
        if self.use_scale and self.reward_scale:
            if isinstance(self.reward_scale, dict) and (self.task not in self.reward_scale):
                return reward
            rs = self.reward_scale[self.task] if isinstance(self.reward_scale, dict) else self.reward_scale
            reward = np.array(np.clip((reward - rs[0]) / (rs[1] - rs[0]), 0.0, 1.0)) * self.clip
            return reward
        else:
            return reward

    def compute_reward(self, seq: List[Dict[str, Any]]):
        if len(seq) < self.seq_len_steps:
            raise InvalidSequenceError(
                f"Input sequence must be at least {self.seq_len_steps} steps long. Seq len is {len(seq)}"
            )

        # Where in sequence to start computing likelihoods. Don't perform redundant likelihood computations.
        start_idx = 0
        for i in range(self.seq_len_steps - 1, len(seq)):
            if not self.is_step_processed(seq[i]):
                start_idx = i
                break
        start_idx = int(max(start_idx - self.seq_len_steps + 1, 0))
        T = len(seq) - start_idx
        camera_keys, skip_frame = self.camera_keys, self.skip_frame

        # Construct image sequence.
        image_batch = {}
        for ik in camera_keys:
            image_batch[ik] = jnp.stack([seq[idx][ik] for idx in range(start_idx, len(seq))])
        stage_indices = jnp.asarray([seq[idx]["reward"] for idx in range(start_idx, len(seq))])

        # Compute batch of encodings and embeddings for likelihood computation.
        idxs = list(range(T - self.seq_len + 1))
        batch_images = {
            ik: np.asarray([image_batch[ik][idx : (idx + self.seq_len)] for idx in idxs]) for ik in camera_keys
        }

        rewards = []
        for i in trange(
            0, len(idxs), self.encoding_minibatch_size, leave=False, ncols=0, desc="reward compute per batch"
        ):
            _range = list(range(i, min(i + self.encoding_minibatch_size, len(idxs))))
            mb_images = {ik: val[_range] for ik, val in batch_images.items()}
            batch = {
                "image": {ik: val[:, skip_frame - 1 :: skip_frame] for ik, val in mb_images.items()},
            }
            output = self.reward_model.get_reward(batch_to_jax(batch), stage_idx=stage_indices[np.asarray(_range)])
            rewards.append(output)
        rewards = jnp.concatenate(rewards, axis=0).squeeze()
        if len(rewards.shape) <= 1:
            rewards = self._reward_scaler(rewards)
        if self.use_task_reward:
            rewards = rewards + jnp.array([seq[i]["reward"] for i in idxs]) * 10
        assert len(rewards) == (T - self.seq_len_steps + 1), f"{len(rewards)} != {T - self.seq_len_steps + 1}"
        for i, rew in enumerate(rewards):
            idx = start_idx + self.seq_len_steps - 1 + i
            assert not self.is_step_processed(seq[idx]), f"seq[idx]: {seq[idx]}"
            seq[idx][DrsRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew

        if seq[0]["is_first"]:
            first_images = {ik: image_batch[ik][:1] for ik in camera_keys}
            # make set of indices for the first sequence
            tmp = jnp.concatenate([jnp.zeros((self.seq_len_steps - 1,)), jnp.arange(self.seq_len_steps)], axis=0)
            indices = np.lib.stride_tricks.sliding_window_view(tmp, self.seq_len_steps, axis=0)[:-1].astype(np.int32)
            first_images = {
                ik: np.stack([batch_images[ik][0][_indices] for _indices in indices], axis=0) for ik in camera_keys
            }
            first_batch = {
                "image": {ik: val[:, skip_frame - 1 :: skip_frame] for ik, val in first_images.items()},
            }

            first_output = self.reward_model.get_reward(
                batch_to_jax(first_batch), stage_idx=stage_indices[: self.seq_len_steps - 1]
            )
            first_rewards = first_output.squeeze()
            if len(first_rewards.shape) <= 1:
                first_rewards = self._reward_scaler(first_rewards)
            if self.use_task_reward:
                first_rewards = (
                    first_rewards + jnp.array([seq[i]["reward"] for i in range(self.seq_len_steps - 1)]) * 10
                )
            assert len(first_rewards) == self.seq_len_steps - 1, f"{len(first_rewards)} != {self.seq_len_steps - 1}"
            for i, rew in enumerate(first_rewards):
                assert not self.is_step_processed(seq[i]), f"Step {i} already processed"
                seq[i][DrsRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew
        return seq

    def is_step_processed(self, step):
        return DrsRewardModel.PRIVATE_LIKELIHOOD_KEY in step.keys()

    def is_seq_processed(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                return False
        return True

    def process_seq(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                continue
            step[DrsRewardModel.PUBLIC_LIKELIHOOD_KEY] = step[DrsRewardModel.PRIVATE_LIKELIHOOD_KEY]
        return seq[self.seq_len_steps - 1 :]
