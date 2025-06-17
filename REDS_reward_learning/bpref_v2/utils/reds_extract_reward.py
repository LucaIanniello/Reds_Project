import datetime
import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from PIL import Image

from bpref_v2.reward_model.disc_reward_model import DISCRewardModel
from bpref_v2.reward_model.drs_reward_model import DrsRewardModel
from bpref_v2.reward_model.reds_reward_model import REDSRewardModel

sys.path.append("/home/workspace/viper_rl")
from viper_rl.videogpt.reward_models.videogpt_reward_model import VideoGPTRewardModel

FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "Furniture_one_leg", "Name of task name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_enum("reward_type", "REDS", ["REDS", "DrS", "VIPER", "ORIL"], "Type of reward model.")
flags.DEFINE_integer("num_success_demos", -1, "Number of demos to convert")
flags.DEFINE_integer("num_failure_demos", -1, "Number of demos to convert")
flags.DEFINE_integer("batch_size", 512, "Batch size for encoding images")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("demo_type", "success", "type of demonstrations.")
flags.DEFINE_string("image_keys", "image", "image keys.")
flags.DEFINE_string("rm_type", "REDS", "reward model type.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")
flags.DEFINE_string("file_type", "pkl", "file type")
flags.DEFINE_string("reward_suffix", "", "reward suffix")


device = torch.device("cuda")


def save_episode(episode, fn: Path, file_type="pkl"):
    if file_type == "pkl":
        pickle.dump(episode, fn.open("wb"))

    elif file_type == "npz":
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **episode)
            bs.seek(0)
            with fn.open("wb") as f:
                f.write(bs.read())


def load_episode(episode, file_type="pkl"):
    if file_type == "pkl":
        with open(episode, "rb") as f:
            ep = pickle.load(f)
    elif file_type == "npz":
        ep = np.load(episode, allow_pickle=True)
    return {key: ep[key] for key in ep}


def main(_):
    demo_dir = FLAGS.demo_dir
    FLAGS.image_keys = "color_image2|color_image1"

    # load reward model.
    if FLAGS.reward_type == "REDS":
        reward_model = REDSRewardModel(
            task=FLAGS.task_name,
            model_name=FLAGS.rm_type,
            rm_path=FLAGS.ckpt_path,
            camera_keys=FLAGS.image_keys.split("|"),
            reward_scale=None,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            reward_model_device=0,
            encoding_minibatch_size=FLAGS.batch_size,
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
            encoding_minibatch_size=FLAGS.batch_size,
            use_task_reward=False,
            use_scale=False,
        )
    elif FLAGS.reward_type == "VIPER":
        domain, task = FLAGS.task_name.split("_", 1)
        reward_model = VideoGPTRewardModel(
            task=FLAGS.task_name,
            vqgan_path=os.path.join(FLAGS.ckpt_path, f"{domain}_vqgan"),
            videogpt_path=os.path.join(FLAGS.ckpt_path, f"{domain}_videogpt_l4_s4"),
            camera_key=FLAGS.image_keys.split("|")[0],
            reward_scale=None,
            minibatch_size=FLAGS.batch_size,
            encoding_minibatch_size=FLAGS.batch_size,
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
            encoding_minibatch_size=FLAGS.batch_size,
            use_task_reward=False,
            use_scale=False,
        )

    dir_path = Path(demo_dir)

    demo_type = [f"_{elem}" for elem in FLAGS.demo_type.split("|")]
    files = []
    for _demo_type in demo_type:
        print(f"Loading {_demo_type[1:]} demos...")
        demo_files = sorted(list(dir_path.glob(f"*{_demo_type}.{FLAGS.file_type}")))
        len_demos = (
            getattr(FLAGS, f"num{_demo_type}_demos")
            if getattr(FLAGS, f"num{_demo_type}_demos") > 0
            else len(demo_files)
        )
        files.extend([(idx, _demo_type[1:], path) for idx, path in enumerate(demo_files[:len_demos])])

    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    def replay_chunk_to_seq(trajectories):
        seq = []
        if FLAGS.reward_type == "DrS":
            semi_sparse_rewards = np.cumsum(trajectories["skills"], axis=0)

        for i in range(FLAGS.window_size * FLAGS.skip_frame - 1):
            elem = {}
            elem["is_first"] = i == 0
            for key in ["observations", "rewards"]:
                if key == "observations":
                    for _key, _val in trajectories[key][i].items():
                        if _key == "color_image2" and FLAGS.reward_type == "VIPER":
                            image = _val
                            image = np.array(Image.fromarray(image).resize((64, 64), Image.Resampling.NEAREST))
                            elem[_key] = image
                        else:
                            elem[_key] = _val
                elif key == "rewards":
                    try:
                        elem["reward"] = trajectories[key][i].squeeze()
                    except:
                        elem["reward"] = trajectories[key][0]
                elif isinstance(trajectories[key], np.ndarray):
                    elem[key] = trajectories[key][0]
            seq.append(elem)

        for i in range(len(trajectories["observations"])):
            elem = {}
            elem["is_first"] = i == -1
            for key in ["observations", "rewards"]:
                if key == "observations":
                    for _key, _val in trajectories[key][i].items():
                        if _key == "color_image2" and FLAGS.reward_type == "VIPER":
                            image = _val
                            image = np.array(Image.fromarray(image).resize((64, 64), Image.Resampling.NEAREST))
                            elem[_key] = image
                        else:
                            elem[_key] = _val
                elif key == "rewards":
                    try:
                        if FLAGS.reward_type == "DrS":
                            elem["reward"] = semi_sparse_rewards[i].squeeze()
                        else:
                            elem["reward"] = trajectories[key][i].squeeze()
                    except:
                        if FLAGS.reward_type == "DrS":
                            elem["reward"] = semi_sparse_rewards[i]
                        else:
                            elem["reward"] = trajectories[key][i]
                elif isinstance(trajectories[key], np.ndarray):
                    elem[key] = trajectories[key][i]
            seq.append(elem)

        return seq

    for idx, demo_type, file_path in files:
        print(f"Loading [{demo_type} {idx+1} | total {len_files}] {file_path}...")
        x = load_episode(file_path, file_type=FLAGS.file_type)
        # first check whether it is already labeled.
        # if x.get("reds_rewards_ckpt_path", None) is not None and x["reds_rewards_ckpt_path"] == FLAGS.ckpt_path:
        #     print(f"Already labeled with {x['reds_rewards_ckpt_path']} in timestamp {x['timestamp']}")
        #     del x
        #     continue

        if x["observations"][0]["color_image1"].shape[0] == 3:
            for i in range(len(x["observations"])):
                x["observations"][i]["color_image1"] = x["observations"][i]["color_image1"].transpose((1, 2, 0))
                x["observations"][i]["color_image2"] = x["observations"][i]["color_image2"].transpose((1, 2, 0))

        _seq = replay_chunk_to_seq(x)
        # print(f"color_image2: {[_seq[i]['color_image2'].shape for i in range(len(_seq))]}")
        seq = reward_model(_seq)
        rewards = np.asarray([elem[reward_model.PUBLIC_LIKELIHOOD_KEY] for elem in seq])
        assert len(x["observations"]) == len(rewards), f"{len(x['observations'])} != {len(rewards)}"

        reward_key = FLAGS.reward_type.lower() + "_rewards" + FLAGS.reward_suffix
        x[f"{reward_key}"] = rewards
        x["timestamp"] = datetime.datetime.now().timestamp()
        x[f"{reward_key}_ckpt_path"] = FLAGS.ckpt_path

        if x["observations"][0]["color_image1"].shape[0] != 3:
            for i in range(len(x["observations"])):
                x["observations"][i]["color_image1"] = x["observations"][i]["color_image1"].transpose((2, 0, 1))
                x["observations"][i]["color_image2"] = x["observations"][i]["color_image2"].transpose((2, 0, 1))

        save_episode(x, file_path, file_type=FLAGS.file_type)
        print(f"Re-saved at {file_path}")


if __name__ == "__main__":
    app.run(main)
