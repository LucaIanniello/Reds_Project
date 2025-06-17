import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from PIL import Image
from tqdm import tqdm, trange

from bpref_v2.reward_model.disc_reward_model import DISCRewardModel
from bpref_v2.reward_model.drs_reward_model import DrsRewardModel
from bpref_v2.reward_model.reds_reward_model import REDSRewardModel

sys.path.append("/home/workspace/viper_rl")
from viper_rl.videogpt.reward_models.videogpt_reward_model import VideoGPTRewardModel

FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "Furniture_one_leg", "Name of task name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("output_dir", "/dev/null/", "result dir.")
flags.DEFINE_enum("reward_type", "REDS", ["REDS", "DrS", "VIPER", "ORIL"], "Type of reward model.")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("image_keys", "image", "image keys.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")
flags.DEFINE_integer("num_samples", 8, "number of canonical samples to compute.")
flags.DEFINE_float("discount", 0.99, "discount factor for computing EPIC dist.")
flags.DEFINE_string("file_tp", "npz", "file type of demonstration data.")


device = torch.device("cuda")


def traj_to_seq(trajectories, task_name):
    seq = []
    rewards = np.cumsum(trajectories["skills"]) if "furniture" in task_name else trajectories["skills"]
    for i in range(min(len(trajectories["observations"]), len(rewards))):
        elem = {}
        elem.update(trajectories["observations"][i])
        elem["reward"] = rewards[i]
        elem["is_first"] = False
        seq.append(elem)

    return seq


def load_dataset(data_path, task_name, seq_len, file_tp="npz"):
    base_dataset = []
    pearson_dataset = []
    eps = sorted(list(data_path.glob(f"*.{file_tp}")))
    for ep in tqdm(eps, desc="load files", ncols=0, leave=False):
        if file_tp == "npz":
            _ep = np.load(ep, allow_pickle=True)
        elif file_tp == "pkl":
            _ep = pickle.load(ep.open("rb"))
        _ep = {key: val for key, val in _ep.items()}
        for i in range(len(_ep["observations"])):
            for key in _ep["observations"][i]:
                if "image" in key:
                    image = _ep["observations"][i][key]
                    if image.shape[0] == 3:
                        _ep["observations"][i][key] = np.moveaxis(image, 0, -1)
                if (
                    key == "color_image2"
                    and "furniture" in task_name
                    and FLAGS.reward_type == "VIPER"
                    and "real" in str(ep)
                ):
                    _ep["observations"][i]["color_image2"] = np.array(
                        Image.fromarray(_ep["observations"][i]["color_image2"]).resize(
                            (128, 128), Image.Resampling.NEAREST
                        )
                    )
        _seq = traj_to_seq(_ep, task_name)
        pearson_dataset.extend(_seq)
        T = len(_seq)
        idxs = list(range(T - seq_len))
        batch_samples = [_seq[idx : (idx + seq_len + 1)] for idx in idxs]
        base_dataset.extend(batch_samples)
    return base_dataset, pearson_dataset


def compute_pearson_distance(rewa: np.ndarray, rewb: np.ndarray, dist: np.ndarray | None = None) -> float:
    """Computes pseudometric derived from the Pearson correlation coefficient.
    It is invariant to positive affine transformations like the Pearson correlation coefficient.
    Args:
        rewa: A reward array.
        rewb: A reward array.
        dist: Optionally, a probability distribution of the same shape as rewa and rewb.
    Returns:
        Computes the Pearson correlation coefficient rho, optionally weighted by dist.
        Returns the square root of 1 minus rho.
    """

    # def _check_dist(dist: np.ndarray) -> None:
    #     assert np.allclose(np.sum(dist), 1)
    #     assert np.all(dist >= 0)

    def _center(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = np.average(x, weights=weights)
        return x - mean

    dist = np.ones_like(rewa) / np.prod(np.asarray(rewa.shape))
    # _check_dist(dist)
    assert rewa.shape == dist.shape
    rewa, rewb = rewa.squeeze(), rewb.squeeze()
    assert rewa.shape == rewb.shape, f"{rewa.shape} != {rewb.shape}"

    dist = dist.flatten()
    rewa = _center(rewa.flatten(), dist)
    rewb = _center(rewb.flatten(), dist)

    vara = np.average(np.square(rewa), weights=dist)
    varb = np.average(np.square(rewb), weights=dist)
    cov = np.average(rewa * rewb, weights=dist)
    corr = cov / (np.sqrt(vara) * np.sqrt(varb) + 1e-10)
    corr = np.where(corr > 1.0, 1.0, corr)
    return np.sqrt(0.5 * (1 - corr))


def remove_keys(seq, key):
    for elem in seq:
        if key in elem:
            del elem[key]
    return seq


def compute_epic(reward_model, base_data, pearson_data, num_samples=64, discount=0.99):
    # assert len(base_data) == len(pearson_data)
    cannon_pseudo_rewards = []
    cannon_gt_rewards = []
    for i in trange(len(base_data), desc="compute canonical samples"):
        images = remove_keys(base_data[i][-reward_model.seq_len :].copy(), reward_model.PRIVATE_LIKELIHOOD_KEY)
        base_reward = reward_model(images)[0][reward_model.PUBLIC_LIKELIHOOD_KEY]
        gt_reward = np.asarray(base_data[i][-1]["reward"])[None]
        pearson_samples = np.random.choice(pearson_data, num_samples)
        _cannon_pseudo_rewards, _cannon_gt_rewards, _cannon_pseudo_next_rewards = [], [], []
        for j in trange(num_samples, leave=False, ncols=0, desc="compute pseudo rewards"):
            new_seq = remove_keys(
                copy.deepcopy(
                    (base_data[i][: reward_model.seq_len] + pearson_samples[j : j + 1].tolist())[
                        -reward_model.seq_len :
                    ]
                ),
                reward_model.PRIVATE_LIKELIHOOD_KEY,
            )
            new_next_seq = remove_keys(
                copy.deepcopy(
                    (base_data[i][-reward_model.seq_len :] + pearson_samples[j : j + 1].tolist())[
                        -reward_model.seq_len :
                    ]
                ),
                reward_model.PRIVATE_LIKELIHOOD_KEY,
            )

            _cannon_gt_rewards.append(pearson_samples[j]["reward"])
            _cannon_pseudo_rewards.append(reward_model(new_seq)[0][reward_model.PUBLIC_LIKELIHOOD_KEY])
            _cannon_pseudo_next_rewards.append(reward_model(new_next_seq)[0][reward_model.PUBLIC_LIKELIHOOD_KEY])

        cannon_pseudo_rewards.append(
            base_reward + discount * np.mean(_cannon_pseudo_rewards) - np.mean(_cannon_pseudo_next_rewards)
        )
        cannon_gt_rewards.append(gt_reward + (discount - 1) * np.mean(_cannon_gt_rewards))

    epic_dist = compute_pearson_distance(np.asarray(cannon_gt_rewards), np.asarray(cannon_pseudo_rewards))
    return {
        "epic_dist": epic_dist,
        "cannon_gt_rewards": np.asarray(cannon_gt_rewards),
        "cannon_pseudo_rewards": np.asarray(cannon_pseudo_rewards),
    }


def main(_):
    demo_dir = FLAGS.demo_dir

    # load reward model.
    if FLAGS.reward_type == "REDS":
        reward_model = REDSRewardModel(
            task=FLAGS.task_name,
            model_name="REDS",
            rm_path=FLAGS.ckpt_path,
            camera_keys=FLAGS.image_keys.split("|"),
            reward_scale=None,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            reward_model_device=0,
            encoding_minibatch_size=1,
            use_task_reward=False,
            use_scale=False,
        )
    elif FLAGS.reward_type == "DrS":
        reward_model = DrsRewardModel(
            task=FLAGS.task_name,
            model_name="DRS",
            rm_path=FLAGS.ckpt_path,
            camera_keys=FLAGS.image_keys.split("|"),
            reward_scale=None,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            reward_model_device=0,
            encoding_minibatch_size=1,
            use_task_reward=False,
            use_scale=False,
        )
    elif FLAGS.reward_type == "VIPER":
        suite, task = FLAGS.task_name.split("_", 1)
        reward_model = VideoGPTRewardModel(
            task=FLAGS.task_name,
            vqgan_path=os.path.join(FLAGS.ckpt_path, suite, f"{suite}_vqgan"),
            videogpt_path=os.path.join(FLAGS.ckpt_path, suite, f"{suite}_videogpt_l4_s4"),
            camera_key=FLAGS.image_keys.split("|")[0],
            reward_scale=None,
            minibatch_size=1,
            encoding_minibatch_size=1,
        )
    elif FLAGS.reward_type == "ORIL":
        reward_model = DISCRewardModel(
            task=FLAGS.task_name,
            model_name="DISC",
            rm_path=FLAGS.ckpt_path,
            camera_keys=FLAGS.image_keys.split("|"),
            reward_scale=None,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            reward_model_device=0,
            encoding_minibatch_size=1,
            use_task_reward=False,
            use_scale=False,
        )

    dir_path = Path(demo_dir)
    base_dataset, pearson_dataset = load_dataset(dir_path, FLAGS.task_name, reward_model.seq_len, FLAGS.file_tp)
    epic_stats = compute_epic(
        reward_model, base_dataset, pearson_dataset, num_samples=FLAGS.num_samples, discount=FLAGS.discount
    )
    os.makedirs(Path(FLAGS.output_dir) / FLAGS.reward_type.lower(), exist_ok=True)
    with open(Path(FLAGS.output_dir) / FLAGS.reward_type.lower() / f"{FLAGS.task_name}", "wb") as f:
        pickle.dump(epic_stats, f)


if __name__ == "__main__":
    app.run(main)
