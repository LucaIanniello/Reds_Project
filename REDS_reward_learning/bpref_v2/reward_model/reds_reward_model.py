from pathlib import Path

import clip
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from bpref_v2.data.instruct import (
    TASK_TO_PHASE,
    get_furniturebench_instruct,
    get_metaworld_instruct,
    get_rlbench_instruct,
)
from bpref_v2.utils.reward_model_loader import load_reward_model

tree_map = jax.tree_util.tree_map


class InvalidSequenceError(Exception):
    def __init__(self, message):
        super().__init__(message)


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


class REDSRewardModel:
    PRIVATE_LIKELIHOOD_KEY = "log_immutable_density"
    PUBLIC_LIKELIHOOD_KEY = "density"

    def __init__(
        self,
        task: str = "",
        model_name: str = "REDS",
        rm_path: str = "",
        camera_keys: str = "image",
        reward_scale=None,
        window_size: int = 4,
        skip_frame: int = 1,
        reward_model_device: int = 0,
        encoding_minibatch_size: int = 32,
        use_task_reward: bool = False,
        use_scale: bool = False,
        clip_value: float = 1.0,
        scale_value: float = 0.1,
        debug: bool = False,
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
        self.scale = scale_value
        self.debug = debug

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

        if self.domain == "metaworld" or self.domain == "factorworld":
            instruct_fn = get_metaworld_instruct
        elif self.domain == "rlbench":
            instruct_fn = get_rlbench_instruct
        elif self.domain == "furniture":
            instruct_fn = get_furniturebench_instruct
        self.insts = {
            phase: clip.tokenize(instruct_fn(self.task, phase, output_type="all")).detach().cpu().numpy()
            for phase in range(TASK_TO_PHASE[self.task])
        }
        self.phases = np.asarray([phase for phase in range(TASK_TO_PHASE[self.task])])
        self._call_step = 0
        self._reset_step = 100

    def __call__(self, seq, **kwargs):
        self._call_step += 1
        if self._call_step % self._reset_step == 0:
            jax.clear_caches()
        return self.process_seq(self.compute_reward(seq, **kwargs), **kwargs)

    def _reward_scaler(self, reward):
        return reward * self.scale
        # if self.use_scale and self.reward_scale:
        #     if isinstance(self.reward_scale, dict) and (self.task not in self.reward_scale):
        #         return reward
        #     rs = self.reward_scale[self.task] if isinstance(self.reward_scale, dict) else self.reward_scale
        #     reward = np.array(np.clip((reward - rs[0]) / (rs[1] - rs[0]), 0.0, 1.0)) * self.clip
        #     return reward
        # else:
        #     return reward

    def compute_reward(self, seq):
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
        camera_keys, window_size, skip_frame = self.camera_keys, self.window_size, self.skip_frame

        # Construct image sequence.
        image_batch = {}
        for ik in camera_keys:
            image_batch[ik] = jnp.stack([seq[idx][ik] for idx in range(start_idx, len(seq))])

        # Compute batch of encodings and embeddings for likelihood computation.
        idxs = list(range(T - self.seq_len + 1))
        batch_images = {
            ik: np.asarray([image_batch[ik][idx : (idx + self.seq_len)] for idx in idxs]) for ik in camera_keys
        }

        rewards = []
        if self.debug:
            target_text_indices = []
            cont_rewards = []
        # liv_rewards = []
        # epic_rewards = []
        for i in trange(
            0, len(idxs), self.encoding_minibatch_size, leave=False, ncols=0, desc="reward compute per batch"
        ):
            _range = list(range(i, min(i + self.encoding_minibatch_size, len(idxs))))
            mb_images = {ik: val[_range] for ik, val in batch_images.items()}
            batch = {
                "instruct": {key: np.stack([val for _ in _range], axis=0) for key, val in self.insts.items()},
                # "phase": np.stack([self.phases for _ in _range], axis=-1),
                "image": {ik: val[:, skip_frame - 1 :: skip_frame] for ik, val in mb_images.items()},
                "timestep": None,
                "attn_mask": np.ones((len(_range), window_size), dtype=np.float32),
            }
            output = self.reward_model.get_reward(batch_to_jax(batch))
            rewards.append(output["rewards"])
            if self.debug:
                target_text_indices.append(output["target_text_indices"])
                cont_rewards.append(output["cont_rewards"])
            # liv_rewards.append(output["liv_rewards"])
            # epic_rewards.append(output["epic_rewards"])
        rewards = jnp.concatenate(rewards, axis=0).squeeze()
        if self.debug:
            cont_rewards = jnp.concatenate(cont_rewards, axis=1)
            target_text_indices = jnp.concatenate(target_text_indices, axis=0)
        # liv_rewards = jnp.concatenate(liv_rewards, axis=0)
        # epic_rewards = jnp.concatenate(epic_rewards, axis=0)
        if len(rewards.shape) <= 1:
            rewards = self._reward_scaler(rewards)
        if self.use_task_reward:
            rewards = rewards + jnp.array([seq[i]["reward"] for i in idxs]) * 10

        if rewards.shape == ():
            # Make it jnp array.
            rewards = jnp.array([rewards])

        assert len(rewards) == (T - self.seq_len_steps + 1), f"{len(rewards)} != {T - self.seq_len_steps + 1}"
        for i, rew in enumerate(rewards):
            idx = start_idx + self.seq_len_steps - 1 + i
            assert not self.is_step_processed(seq[idx]), f"seq[idx]: {seq[idx]}"
            seq[idx][REDSRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew
            if self.debug:
                seq[idx]["cont_reward"] = cont_rewards[..., i]
                seq[idx]["target_text_indices"] = target_text_indices[i]
            # seq[idx]["epic_reward"] = epic_rewards[i]
            # seq[idx]["liv_reward"] = liv_rewards[i]

        if seq[0]["is_first"]:
            first_images = {ik: image_batch[ik][:1] for ik in camera_keys}
            # make set of indices for the first sequence
            tmp = jnp.concatenate([jnp.zeros((self.seq_len_steps - 1,)), jnp.arange(self.seq_len_steps)], axis=0)
            indices = np.lib.stride_tricks.sliding_window_view(tmp, self.seq_len_steps, axis=0)[:-1].astype(np.int32)
            first_images = {
                ik: np.stack([batch_images[ik][0][_indices] for _indices in indices], axis=0) for ik in camera_keys
            }
            first_attn_masks = np.flipud(np.triu(np.ones((self.seq_len_steps - 1, self.seq_len), dtype=np.float32), 1))
            first_batch = {
                "instruct": {
                    key: np.stack([val for _ in range(self.seq_len_steps - 1)], axis=0)
                    for key, val in self.insts.items()
                },
                # "phase": np.stack([self.phases for _ in range(self.seq_len_steps - 1)], axis=-1),
                "image": {ik: val[:, skip_frame - 1 :: skip_frame] for ik, val in first_images.items()},
                "timestep": None,
                "attn_mask": first_attn_masks[:, skip_frame - 1 :: skip_frame],
            }

            first_output = self.reward_model.get_reward(batch_to_jax(first_batch))
            first_rewards = first_output["rewards"].squeeze()
            if self.debug:
                first_target_text_indices = first_output["target_text_indices"]
                first_cont_rewards = first_output["cont_rewards"]

            # first_liv_rewards, first_epic_rewards, first_cont_rewards, first_cont_reward_raw = (
            #     first_output["liv_rewards"],
            #     first_output["epic_rewards"],
            #     np.argmax(first_output["cont_rewards"], axis=0),
            #     first_output["cont_rewards"],
            # )
            if len(first_rewards.shape) <= 1:
                first_rewards = self._reward_scaler(first_rewards)
            if self.use_task_reward:
                first_rewards = (
                    first_rewards + jnp.array([seq[i]["reward"] for i in range(self.seq_len_steps - 1)]) * 10
                )
            assert len(first_rewards) == self.seq_len_steps - 1, f"{len(first_rewards)} != {self.seq_len_steps - 1}"
            for i, rew in enumerate(first_rewards):
                assert not self.is_step_processed(seq[i]), f"Step {i} already processed"
                seq[i][REDSRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew
                if self.debug:
                    seq[i]["cont_reward"] = first_cont_rewards[..., i]
                    seq[i]["target_text_indices"] = first_target_text_indices[i]

                # seq[i]["epic_reward"] = first_epic_rewards[i]
                # seq[i]["cont_reward"] = first_cont_rewards[i]
                # seq[i]["cont_reward_raw"] = first_cont_reward_raw[..., i]
                # seq[i]["liv_reward"] = first_liv_rewards[i]

        return seq

    def is_step_processed(self, step):
        return REDSRewardModel.PRIVATE_LIKELIHOOD_KEY in step.keys()

    def is_seq_processed(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                return False
        return True

    def process_seq(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                continue
            step[REDSRewardModel.PUBLIC_LIKELIHOOD_KEY] = step[REDSRewardModel.PRIVATE_LIKELIHOOD_KEY]
        # return seq
        return seq[self.seq_len_steps - 1 :]
