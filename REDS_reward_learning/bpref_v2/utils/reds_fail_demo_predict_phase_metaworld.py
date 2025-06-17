import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from PIL import Image

from bpref_v2.data.instruct import TASK_TO_PHASE
from bpref_v2.reward_model.reds_reward_model import REDSRewardModel

sys.path.append("/home/workspace/viper_rl")

FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "metaworld_door-open", "Name of task name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_enum("reward_type", "REDS", ["REDS"], "Type of reward model.")
flags.DEFINE_integer("num_success_demos", 100, "Number of success demonstrations.")
flags.DEFINE_integer("batch_size", 32, "Batch size for encoding images")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("image_keys", "image", "image keys.")
flags.DEFINE_string("rm_type", "REDS", "reward model type.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")
flags.DEFINE_string("file_type", "npz", "file type")
flags.DEFINE_float("quantile", 0.75, "quantile")

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
    FLAGS.image_keys = "image"

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
            debug=True,
        )
    else:
        raise ValueError(f"Invalid reward type: {FLAGS.reward_type}")

    dir_path = Path(demo_dir)

    success_demos = sorted(list(dir_path.glob(f"*_success.{FLAGS.file_type}")))[: FLAGS.num_success_demos]
    failure_demos = sorted(list(dir_path.glob(f"*_failure.{FLAGS.file_type}")))

    len_succ_files = len(success_demos)
    len_fail_files = len(failure_demos)

    if len_succ_files == 0 or len_fail_files == 0:
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
                        if _key == "image" and FLAGS.reward_type == "VIPER":
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
                        if _key == "image" and FLAGS.reward_type == "VIPER":
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

    cont_logit_per_skill = {i: [] for i in range(TASK_TO_PHASE[FLAGS.task_name.split("_", 1)[-1]])}
    for idx, file_path in enumerate(success_demos):
        print(f"Computing stat per phase: [success {idx+1} | total {len_succ_files}] {file_path}...")
        x = load_episode(file_path, file_type=FLAGS.file_type)
        _seq = replay_chunk_to_seq(x)
        seq = reward_model(_seq)
        cont_rewards = np.asarray([elem["cont_reward"] for elem in seq])
        skills = x["skills"].astype(np.uint8)
        for i in range(len(cont_rewards)):
            cont_logit_per_skill[skills[i]].append(cont_rewards[i, skills[i]])

    cont_logit_per_skill = {
        key: np.quantile(val, FLAGS.quantile) if len(val) else 0.0 for key, val in cont_logit_per_skill.items()
    }
    print(f"cont_logit_per_skill from {len(success_demos)} success demonstrations: {cont_logit_per_skill}")

    for idx, file_path in enumerate(failure_demos):
        print(f"Loading [failure {idx+1} | total {len_fail_files}] {file_path}...")
        x = load_episode(file_path, file_type=FLAGS.file_type)
        if x["observations"][0]["image"].shape[0] == 3:
            for i in range(len(x["observations"])):
                x["observations"][i]["image"] = x["observations"][i]["image"].transpose((1, 2, 0))

        _seq = replay_chunk_to_seq(x)
        seq = reward_model(_seq)

        cont_rewards = np.asarray([elem["cont_reward"] for elem in seq])
        assert len(x["observations"]) == len(cont_rewards), f"{len(x['observations'])} != {len(cont_rewards)}"

        # predict failure phase
        cursor = 0
        predicted_phases = []
        for i in range(len(cont_rewards)):
            phase_changed = 0
            next_cursor = min(cursor + 1, cont_rewards.shape[1] - 1)
            if (
                cont_rewards[i, next_cursor] > cont_rewards[i, cursor]
                and cont_rewards[i, next_cursor] > cont_logit_per_skill[next_cursor]
            ):
                phase_changed = 1
                cursor = next_cursor
            predicted_phases.append(phase_changed)

        x["skills"] = predicted_phases

        save_episode(x, file_path, file_type=FLAGS.file_type)
        print(f"Re-saved at {file_path}")


if __name__ == "__main__":
    app.run(main)
