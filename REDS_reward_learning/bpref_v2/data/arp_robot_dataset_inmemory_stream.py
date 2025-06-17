import glob
import os
import random
import traceback
from collections import Counter, deque
from io import BytesIO
from typing import Sequence

import clip
import numpy as np
import torch
from ml_collections import ConfigDict
from tqdm import trange

from .instruct import PHASE_TO_REWARD, get_metaworld_instruct, get_rlbench_instruct


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0]


def get_indices_with_same_values(arr):
    result = {key: [] for key in set(arr)}
    for i in range(len(arr)):
        val = arr[i]
        result[val].append(i)
    return result


def shape_expert_rewards(skills, eps=0.2):
    new_skills = np.asarray(skills).copy()
    transitions = []
    for i in range(len(skills) - 1):
        if skills[i + 1] != skills[i]:
            transitions.append(i + 1)
    for idx, elem in enumerate(transitions):
        new_skills[elem - 1] += eps
        # target_range = range(elem[idx - 1], elem)
        if idx == 0:
            target_range = range(0, elem)
        else:
            target_range = range(int(transitions[idx - 1]), elem)
        start, end = new_skills[target_range[0]], new_skills[target_range[-1]]
        for i in target_range:
            new_skills[i] = (end - start) / len(target_range) * (i - target_range[0]) + start

    return new_skills


def get_failure_skills_and_phases(skill, phase, task_name="one_leg", failure_phase=-1, threshold=20):
    failure_skills, failure_phases = phase.copy(), phase.copy().astype(np.float32)
    cursor = 0
    for i in range(len(failure_skills)):
        failure_skills[i] = max(cursor, failure_skills[i])
        if cursor < failure_skills[i]:
            cursor = failure_skills[i]
    apply_failure = False
    for idx in range(1, len(phase) - 1):
        if phase[idx] - phase[idx - 1] < 0 and phase[idx + 1] - phase[idx] <= 0:
            apply_failure = True
            break
    if apply_failure:
        pass
        # failure_phases[idx:] = 999
        failure_skills[idx:] = 100 + failure_skills[idx:]

    return failure_skills, failure_phases


class ARPRobotDataset(torch.utils.data.IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.data_dir = ""
        config.max_episode_steps = 100
        config.task_name = "one_leg"

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_size = 64

        config.benchmark = "metaworld"
        config.image_keys = "image"
        config.image_main_key = "image"
        config.state_key = ""
        config.action_dim = 4
        config.clip_action = 0.999

        config.skip_frame = 1
        config.window_size = 2

        config.offset = 0
        config.num_demos = 100
        config.target_skill = -1
        config.shaping_eps = 0.0

        config.fetch_every = 2000
        config.num_workers = 8
        config.max_size = 5000

        # Reward Learning option
        config.output_type = "raw"
        config.pvr_type = "LIV"
        config.use_sparse = False
        config.random_sample_init = 4
        config.random_sample_end = 20
        config.pearson_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, update, start_offset_ratio=None, split="train", demo_type="success", h5_file=None):
        self.config = self.get_default_config(update)
        self.split = split

        self.demo_type = "" if demo_type == "all" else f"{demo_type}"
        self._total_episode_fns = self._fetch_episodes()
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0
        self._samples_since_last_fetch = self.config.fetch_every

        self.random_start_offset = 0

        self.suffix = "" if self.config.output_type == "raw" else f"_{self.config.pvr_type}"
        if self.suffix == "":
            self.tokenizer = self.build_tokenizer()

        self.mode = "global"
        self.p2r = {tn: PHASE_TO_REWARD[tn] for tn in self.config.task_name.split("|")}

        self.instruct_fn = self._get_instruct_fn()

    def _get_instruct_fn(self):
        if self.config.benchmark == "metaworld":
            return get_metaworld_instruct
        elif self.config.benchmark == "rlbench":
            return get_rlbench_instruct

    def set_mode(self, mode):
        self.mode = mode

    def _fetch_episodes(self):
        data_dir, task_name = self.config.data_dir.split("|"), self.config.task_name.split("|")
        episodes = []
        num_demos_per_task = self.config.num_demos
        for dd, tn in zip(data_dir, task_name):
            _dataset_path = dd
            _episodes = sorted(glob.glob(os.path.join(_dataset_path, self.split, f"*_{self.demo_type}.npz")))[
                :num_demos_per_task
            ]
            _episodes_pairs = [(idx, tn, _ep) for idx, _ep in enumerate(_episodes)]
            episodes.extend(_episodes_pairs)
            print(f"[INFO] {tn} | {self.split}_{self.demo_type} | {len(_episodes_pairs)} episodes are loaded.")
        return episodes

    def _load_episode(self, task_name, episode_path):
        image_keys = self.config.image_keys.split("|")
        target_keys = (
            ["action", "timestep", "skill", "phase", "attn_mask"] + image_keys + [f"next_{ik}" for ik in image_keys]
        )
        episode_data = {key: [] for key in target_keys}
        done_flags, task_names = [], []

        episode = np.load(episode_path, allow_pickle=True)
        episode = {key: np.asarray(val) for key, val in episode.items()}

        phases = episode["skills"].astype(np.int32)

        if self.demo_type == "success":
            for i in [-3, -2]:
                if phases[i] != phases[-1]:
                    phases[i] += 1

        episode_length = episode["rewards"].shape[0]
        episode_data["action"] = episode["actions"].astype(np.float32)
        episode_data["skill"] = episode["skills"].astype(np.int32)
        episode_data["phase"] = phases.astype(np.int32)

        for i in range(episode_length):
            episode_data["timestep"].append(np.asarray(i).astype(np.int32))
            episode_data["attn_mask"].append(np.asarray(1).astype(np.int32))

            done_flags.append(bool(i == episode_length - 1))
            task_names.append(np.frombuffer(str(task_name).encode("utf-8"), dtype=np.uint8))

            for image_key in image_keys:
                image = episode["observations"][i][image_key].astype(np.uint8)
                episode_data[image_key].append(image)
                episode_data[f"next_{image_key}"].append(min(i + self.config.skip_frame, episode_length - 1))

        episode_data = {key: np.asarray(val) for key, val in episode_data.items()}
        episode_data.update({"terminals": np.asarray(done_flags), "task_name": np.asarray(task_names)})
        episode_data["failure_phase"] = int(episode.get("failure_phase", -1))

        if self.demo_type == "failure":
            failure_skills, failure_phases = get_failure_skills_and_phases(
                episode_data["skill"],
                episode_data["phase"],
                task_name=task_name,
                failure_phase=episode_data["failure_phase"],
            )
            episode_data["phase"] = failure_phases
            episode_data["skill"] = failure_skills

        rewards = np.asarray([self.p2r[task_name][phase] for phase in episode_data["phase"]], dtype=np.float32)
        if self.demo_type == "success":
            shaped_rewards = np.asarray(shape_expert_rewards(rewards, self.config.shaping_eps))
        else:
            shaped_rewards = rewards
        episode_data["reward"] = shaped_rewards

        return episode_data

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def build_tokenizer(self):
        tokenizer_max_length = 77
        tokenizer = clip.tokenize

        def tokenizer_fn(instruct):
            tokenized_instruct = np.asarray(tokenizer(instruct)).astype(np.int32)
            padding_mask = np.ones(tokenizer_max_length).astype(np.float32)
            return tokenized_instruct, padding_mask

        return tokenizer_fn

    def _compute_window_indices(self, demo_offset):
        window_size, frame = self.config.window_size, self.config.skip_frame
        demo_start = 0
        stack_start, stack_end = (
            max(demo_start + demo_offset - window_size * frame, demo_start),
            demo_start + demo_offset,
        )
        batch_indices = [stack_end]
        e = stack_end
        for _ in range(self.config.window_size - 1):
            e -= frame
            if e >= stack_start:
                batch_indices.insert(0, e)
            else:
                break
        batch_indices = np.asarray(batch_indices)
        return batch_indices

    def _get_tokenized_instructions(self, decoded_task_name: bytes, skills: Sequence):
        _instruct = []
        for idx in range(len(skills)):
            _instruct.append(self.instruct_fn(decoded_task_name, np.asarray(skills[idx])))
        instruct, text_padding_mask = self.tokenizer(_instruct)
        return instruct, text_padding_mask

    def _sample_episode(self):
        _, _, eps_fn = random.choice(self._episode_fns).split("|")
        return self._episodes[eps_fn]

    def _compute_skill_stats(self, skills):
        skill_dict = {key: val / len(skills) for key, val in dict(Counter(skills)).items()}
        skill_dict[np.max(list(skill_dict.keys())) + 1] = 1.0
        return skill_dict

    def _store_episode(self, eps_idx, tn, eps_fn):
        try:
            episode = self._load_episode(tn, eps_fn)
        except Exception as e:
            print(f"[ERROR] {e}")
            raise
            return None, False
        ep_len = episode_len(episode)
        while ep_len + self._size > self.config.max_size:
            early_eps_idx, early_eps_tn, early_eps_fn = self._episode_fns.pop(0).split("|")
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
        self._episode_fns.append(f"{eps_idx}|{tn}|{eps_fn}")
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += ep_len

        return ep_len, True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self.config.fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except AttributeError:
            worker_id = 0
        shuffled_episode_fns = self._total_episode_fns.copy()
        np.random.shuffle(shuffled_episode_fns)
        fetched_size = 0
        for idx in (
            pbar := trange(
                len(shuffled_episode_fns),
                desc=f"[{self.demo_type.upper()} {self.split.upper()} WORKER {worker_id}] fetch data",
                ncols=0,
                leave=False,
            )
        ):
            # print(f"sample episode from {len(self._episode_fns)} episodes.")
            eps_idx, tn, eps_fn = shuffled_episode_fns[idx]
            if eps_idx % self.config.num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                continue
            if fetched_size > self.config.max_size:
                break
            eps_len, flag = self._store_episode(eps_idx, tn, eps_fn)
            if not flag:
                break
            pbar.set_postfix({"fetched_size": fetched_size})
            fetched_size += eps_len

    def _get_image_stack(self, episodes, image_keys, indices):
        image_stack = {ik: deque([], maxlen=self.config.window_size) for ik in image_keys}
        for ik in image_keys:
            images = episodes[ik]
            for _ in range(self.config.window_size):
                image_stack[ik].append(images[indices[0]])
            target_images = images[indices]
            if target_images.shape[1] == 3:
                target_images = target_images.transpose(0, 2, 3, 1)
            image_stack[ik].extend(target_images)
        return {key: np.asarray(val) for key, val in image_stack.items()}

    def __sample_global(self):
        episode = self._sample_episode()
        index = np.random.randint(0, len(episode["terminals"]))
        task_name = BytesIO(episode["task_name"][index]).read().decode("utf-8")
        task_name = self._decode_task_name(episode["task_name"][index])

        batch = {"image": {}}
        image_keys = self.config.image_keys.split("|")
        batch_indices = self._compute_window_indices(index)
        batch["image"] = self._get_image_stack(episode, image_keys, batch_indices)
        batch.update(self._stack_episode_data(episode, ["action", "timestep", "phase", "attn_mask"], batch_indices))

        batch["skill"] = batch["phase"]
        batch["action"] = np.clip(batch["action"], -self.config.clip_action, self.config.clip_action)
        batch["instruct"] = self._get_instructions(task_name, batch["phase"], episode, index)
        batch["reward"] = self._compute_reward(episode, batch["phase"], task_name)

        batch.update(self._get_random_next_state(episode, index, image_keys))
        batch.update(self._get_pearson_data(task_name, image_keys))

        if len(self.config.task_name.split("|")) > 1:
            task_hash = abs(hash(task_name))
            batch["skill"] += task_hash
            batch["phase"] += task_hash

        batch["ep_phase"] = np.max(episode["phase"])

        return batch

    def _decode_task_name(self, task_name):
        return BytesIO(task_name).read().decode("utf-8")

    def _get_random_next_state(self, episode, index, image_keys):
        random_next_index = min(
            index + np.random.randint(1, self.config.random_sample_end), len(episode["terminals"]) - 1
        )
        random_next_batch_indices = self._compute_window_indices(random_next_index)
        random_next_image = self._get_image_stack(episode, image_keys, random_next_batch_indices)
        random_next_phase = self._stack_episode_data(episode, ["phase"], random_next_batch_indices)["phase"]
        return {
            "random_next_image": random_next_image,
            "random_next_phase": random_next_phase,
            "random_next_instruct": self._get_tokenized_instructions(
                self._decode_task_name(episode["task_name"][index]), random_next_phase
            )[0],
        }

    def _compute_reward(self, episode, phases, task_name):
        if self.config.use_sparse:
            rew = episode["terminals"][phases].astype(np.float32)
        rew = np.asarray([self.p2r[task_name][phase] for phase in phases], dtype=np.float32)
        return rew

    def _stack_episode_data(self, episode, keys, batch_indices):
        stacked_data = {}
        for key in keys:
            initial_value = np.zeros((self.config.action_dim,)) if key == "action" else np.asarray(0)
            initial_dtype = np.float32 if key == "action" else np.int32

            stack = deque(
                [initial_value.astype(initial_dtype)] * self.config.window_size, maxlen=self.config.window_size
            )
            stack.extend(episode[key][batch_indices])
            stacked_data[key] = np.asarray(stack)
        return stacked_data

    def _get_pearson_data(self, task_name, image_keys):
        pearson_data = {"pearson_image": {}, "pearson_reward": [], "pearson_phase": []}
        for _ in range(self.config.pearson_size):
            pearson_episode = self._sample_episode()
            pearson_index = np.random.randint(0, len(pearson_episode["terminals"]))
            pearson_batch_indices = self._compute_window_indices(pearson_index)

            pearson_images = self._get_image_stack(pearson_episode, image_keys, pearson_batch_indices)
            for key in image_keys:
                if key not in pearson_data["pearson_image"]:
                    pearson_data["pearson_image"][key] = []
                pearson_data["pearson_image"][key].append(pearson_images[key])

            pearson_data["pearson_reward"].append(self.p2r[task_name][pearson_episode["phase"][pearson_index]])
            pearson_data["pearson_phase"].append(pearson_episode["phase"][pearson_index])

        pearson_data["pearson_image"] = {key: np.asarray(val) for key, val in pearson_data["pearson_image"].items()}
        pearson_data["pearson_reward"] = np.asarray(pearson_data["pearson_reward"], dtype=np.float32)
        pearson_data["pearson_phase"] = np.asarray(pearson_data["pearson_phase"], dtype=np.int32)
        pearson_data["pearson_instruct"], _ = self._get_tokenized_instructions(task_name, pearson_data["pearson_phase"])

        return pearson_data

    def _get_instructions(self, task_name, phase, episode, index):
        if self.suffix == "":
            return self._get_tokenized_instructions(task_name, phase)[0]
        return episode[f"instruct{self.suffix}"][index]

    def _sample(self):
        try:
            self._try_fetch()
        except StopIteration:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        if self.mode == "local":
            return self.__sample_local()
        else:
            return self.__sample_global()

    def __iter__(self):
        while True:
            yield self._sample()

    @property
    def num_actions(self):
        return self.config.action_dim

    @property
    def obs_shape(self):
        res = {"image": {}}
        for key in self.config.image_keys.split("|"):
            res["image"][key] = (self.config.image_size, self.config.image_size, 3)
        res["rtg"] = (1,)
        if self.config.state_key != "":
            res["state"] = self.config.state_dim
        return res


if __name__ == "__main__":
    config = ARPRobotDataset.get_default_config()
    base_path = "/home/rlbench_data/pick_up_cup"
    config.benchmark = "rlbench"
    config.task_name = "pick_up_cup"
    config.data_dir = base_path
    config.window_size = 4
    config.skip_frame = 1
    config.num_demos = 1
    config.image_keys = "image_front|image_wrist"
    config.use_sparse = False
    config.output_type = "raw"
    # config.rtg_key = "clip"

    split = "train"
    ds = ARPRobotDataset(update=config, demo_type="success")
    ds.mode = "global"
    from tqdm import tqdm

    # for i in trange(len(ds)):
    for idx, batch in tqdm(enumerate(ds), total=100):
        if idx == 80:
            print("early break.")
            break
        # batch = ds[i]
        for key, val in batch.items():
            if isinstance(val, dict):
                for ik in val:
                    print(f"[INFO] {key}|{ik}: {val[ik].shape} | {val[ik].dtype}")
            elif isinstance(val, np.ndarray):
                print(f"[INFO] {key}: {val.shape} | {val.dtype}")
            else:
                print(f"[INFO] {key}: {val} | {type(val)}")
        # print(f"[INFO] skill: {batch['skill']}")
        # print(f"[INFO] reward: {batch['reward']}")
        print("")
        # # print(f"[INFO] rtg: {batch['rtg']}")
        # # print(f"[INFO] rtg of dataset: {ds.rtg}")
        # # print(f"[INFO] rtg_scale of dataset: {ds.rtg_scale}")
        from PIL import Image

        image_keys = config.image_keys.split("|")
        for ik in image_keys:
            for ws in range(config.window_size):
                img = Image.fromarray(batch["image"][ik][ws])
                img.save(f"{ik}_ws{ws}.jpeg")
                # img = Image.fromarray(batch["next_image"][ik])
                # img.save(f"{ik}_next_ws{ws}.jpeg")
                # if ds.mode == "local":
                #     img = Image.fromarray(batch["initial_image"][ik][ws])
                #     img.save(f"{ik}_initial_ws{ws}.jpeg")
                #     img = Image.fromarray(batch["goal_image"][ik][ws])
                #     img.save(f"{ik}_goal_ws{ws}.jpeg")

        # if 0 in batch["skill"]:
        #     print("early break.")
        #     break
