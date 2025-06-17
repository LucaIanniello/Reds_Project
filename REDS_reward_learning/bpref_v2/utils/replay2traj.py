import datetime
import io
import os
import uuid
from pathlib import Path

import numpy as np
from tqdm import trange
from rich.console import Console
from absl import app, flags

import shutil


console = Console()
FLAGS = flags.FLAGS

# Environment Setting.
flags.DEFINE_string("data_path", "", "Path to the data directory.")
flags.DEFINE_string("task_name", "door-open", "Task name.")
flags.DEFINE_integer("seed", 0, "Target seed.")
flags.DEFINE_string("replay_path", "", "Path to the replay.")
flags.DEFINE_string("experiment_key", "", "Experiment key for training online agents.")
flags.DEFINE_integer("chunk_size", 1024, "chunk_size for replay file.")
flags.DEFINE_string("image_keys", "image", "image keys for saving.")
flags.DEFINE_integer("num_demos", 10, "number of demonstrations to collect.")


def save_episode(directory, episode, tp):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = len(episode[list(episode.keys())[0]])
    filename = os.path.join(directory, f"{timestamp}_{length}_{identifier[:4]}_{tp}.npz")
    filename = Path(filename).expanduser()

    data = {key: np.stack(val) for key, val in episode.items()}

    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename


def replay_ep2data(ep, image_keys=["image"]):
    length = ep["action"].shape[0]
    data_keys = ["observations", "actions", "terminals", "rewards", "discounts", "skills"]
    data = {key: [] for key in data_keys}
    for i in range(length):
        elem = {}
        for ok in image_keys:
            elem[ok] = ep[ok][i]
        data["observations"].append(elem)
    data["actions"] = ep["action"]
    data["terminals"] = ep["is_terminal"]
    data["rewards"] = data["skills"] = ep["skill"].squeeze()
    data["discounts"] = ep["is_last"]

    data = {key: np.asarray(val) for key, val in data.items()}

    return data


def replay_chunk2data(chunk_path, output_path, image_keys=["image"]):
    chunk = np.load(chunk_path)
    chunk = {key: chunk[key] for key in chunk}

    ep_indices = list(np.nonzero(chunk["is_last"])[0])
    ep_indices.insert(0, 0)
    ep_indices.append(len(chunk["is_last"]) - 1)

    def get_last_idx(last_indices, index):
        for i in range(len(last_indices)):
            if last_indices[i] > index:
                return last_indices[i]
        return index

    # for idx in range(min(len(ep_first_indices), len(ep_last_indices))):
    for idx in range(1, len(ep_indices) - 1):
        last_idx = get_last_idx(ep_indices, ep_indices[idx])
        ep_range = range(ep_indices[idx], last_idx + 1)
        if not len(ep_range) > 1:
            continue
        ep = {key: val[ep_range] for key, val in chunk.items()}
        data = replay_ep2data(ep, image_keys=image_keys)
        save_episode(output_path, data, "success" if np.sum(ep["success"]) > 0. else "failure")


def replay2data(replay_path, output_path, image_keys=["image"], chunk_size=1024):
    eps = sorted(list(Path(replay_path).glob("*.npz")))
    for idx in trange(len(eps), desc="convert trajectory from replays"):
        if str(chunk_size) not in eps[idx].stem:
            continue
        replay_chunk2data(eps[idx], output_path, image_keys=image_keys)

def main(_):
    # for task_name in ["door-open", "drawer-open", "lever-pull", "peg-insert-side", "push", "faucet-close", "disassemble", "sweep-into"]:
    output_path = Path(FLAGS.data_path) / f"{FLAGS.task_name}" / "failures"
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    replay_path = Path(FLAGS.replay_path) / f"{FLAGS.task_name}_{FLAGS.experiment_key}_seed{FLAGS.seed}" / "replay"
    print(f"replay_path: {replay_path}")
    replay2data(replay_path, output_path, image_keys=FLAGS.image_keys.split("|"), chunk_size=FLAGS.chunk_size)

    failure_replays = sorted(list(Path(output_path).glob("*failure.npz")))
    print(f"number of failures: {len(failure_replays)}")
    try:
        interleave = int(len(failure_replays) / FLAGS.num_demos)
        train_files = failure_replays[::interleave][: FLAGS.num_demos]
        print(f"added train files: {len(train_files)}")
        eval_files = failure_replays[-max(int(FLAGS.num_demos // 10), 1) :]

        target_train_path = Path(FLAGS.data_path) / f"{FLAGS.task_name}" / "train"
        target_eval_path = Path(FLAGS.data_path) / f"{FLAGS.task_name}" / "val"
        os.makedirs(target_train_path, exist_ok=True)
        os.makedirs(target_eval_path, exist_ok=True)
        for file in train_files:
            shutil.copy(str(file), target_train_path / file.name)
        for file in eval_files:
            shutil.copy(str(file), target_eval_path / file.name)
        shutil.rmtree(output_path)

    except Exception as e:
        print(f"Error: {e}, delete temporary demonstrations for sanity.")
        shutil.rmtree(output_path)


if __name__ == "__main__":
    app.run(main)
