import argparse
import os
import pathlib
import pickle
from collections import defaultdict, deque

import h5py
import numpy as np
from tqdm import tqdm, trange


def check_dtype(val):
    if isinstance(val, int):
        return np.int32
    if isinstance(val, list):
        return check_dtype(val[0])
    if isinstance(val, float):
        return np.float32
    if isinstance(val, np.ndarray):
        return val.dtype


def ds_to_hdf5(
    data_dir,
    task_name,
    num_demos,
    demo_chunk_size,
    h5_file,
    split="train",
    image_keys="color_image2|color_image1",
    offset=0,
    window_size=4,
    skip_frame=16,
    capacity=1e6,
):
    data_dir, task_name = data_dir.split("|"), task_name.split("|")
    episodes = []
    num_demos_per_task = num_demos // len(task_name)
    for dd, tn in zip(data_dir, task_name):
        _dataset_path = pathlib.Path(dd)
        _episodes = sorted((_dataset_path / split).glob(r"[0-9]*.pkl"))[:num_demos_per_task]
        _episodes_pairs = [(tn, _ep) for _ep in _episodes]
        episodes.extend(_episodes_pairs)

    image_keys = image_keys.split("|")
    target_keys = ["action", "timestep", "skill", "attn_mask"] + [image_key for image_key in image_keys]
    liv_keys = [f"{elem}_{image_key}" for elem in ["initial", "next", "goal"] for image_key in image_keys] + ["r"]
    demo_keys = ["demo_number", "demo_offset", "demo_idx"]
    offset, total_timestep, chunk_timestep, demo_number = offset, 0, 0, 0

    with (pbar := tqdm(desc="load data", ncols=0, total=len(episodes))):
        for ep_idx, (tn, ep) in enumerate(episodes, start=1):
            # do it in batch style: update h5_file every (demo_chunk) demonstrations.
            if ep_idx % demo_chunk_size == 1:
                ret = {key: [] for key in target_keys}
                done_bool_, task_name_ = [], []
                demo_data = {key: [] for key in demo_keys}
                liv_data = {key: [] for key in liv_keys}

            with open(ep, "rb") as f:
                ep = pickle.load(f)
            ep = {key: np.asarray(val) for key, val in ep.items()}
            N = ep["actions"].shape[0]
            demo_data["demo_idx"].extend([[total_timestep, total_timestep + N]] * N)
            cumsum_skills = np.cumsum(ep["skills"])
            subgoals = list(np.nonzero(ep["skills"])[0]) + [len(ep["skills"]) - 1]

            total_stack = defaultdict(dict)
            for frame in range(skip_frame):
                total_stack[frame] = {key: deque([], maxlen=window_size) for key in target_keys}
                for j in range(window_size):
                    total_stack[frame]["action"].append(np.zeros((ep["actions"].shape[1],), dtype=np.float32))
                    total_stack[frame]["timestep"].append(np.asarray(0).astype(np.int32))
                    total_stack[frame]["skill"].append(np.asarray(0).astype(np.int32))
                    total_stack[frame]["attn_mask"].append(np.asarray(0).astype(np.int32))
                    for ik in image_keys:
                        total_stack[frame][ik].append(np.zeros(ep["observations"][0][ik].shape[-3:], dtype=np.uint8))

            for i in trange(offset, N, desc=f"load ep {ep_idx}", ncols=0, leave=False):
                mod = i % skip_frame
                stack = total_stack[mod]

                action = ep["actions"][i].astype(np.float32)
                timestep = np.asarray(i).astype(np.int32)
                attn_mask = np.asarray(1).astype(np.int32)
                skill = cumsum_skills[i].astype(np.int32)

                stack["action"].append(action)
                stack["timestep"].append(timestep)
                stack["attn_mask"].append(attn_mask)
                stack["skill"].append(skill)

                demo_data["demo_offset"].append(i)
                demo_data["demo_number"].append(demo_number)

                done_bool = bool(ep["rewards"][i])
                done_bool_.append(done_bool)
                liv_data["r"].append(int(done_bool) - 1)
                encoded_tn = np.frombuffer(str(tn).encode("utf-8"), dtype=np.uint8)
                task_name_.append(encoded_tn)

                for ik in image_keys:
                    image = ep["observations"][i][ik].astype(np.uint8)
                    stack[ik].append(image)

                    liv_data[f"next_{ik}"].append(min(i + skip_frame, N - 1))
                    initial_idx = 0 if cumsum_skills[i] == 0 else subgoals[cumsum_skills[i] - 1]
                    liv_data[f"initial_{ik}"].append(initial_idx)
                    liv_data[f"goal_{ik}"].append(subgoals[cumsum_skills[i]])

                for key in target_keys:
                    ret[key].append(np.stack(stack[key]))

            total_timestep += N
            demo_number += 1

            if ep_idx % demo_chunk_size == 0 or ep_idx == num_demos:
                ret = {key: np.asarray(val, dtype=check_dtype(val)) for key, val in ret.items()}
                ret.update({key: np.asarray(val, dtype=check_dtype(val)) for key, val in liv_data.items()})
                ret.update({key: np.asarray(val, dtype=check_dtype(val)) for key, val in demo_data.items()})
                ret.update({"terminals": np.asarray(done_bool_), "task_name": np.asarray(task_name_)})

                batch_size = 128
                for key, val in tqdm(
                    ret.items(),
                    total=len(ret),
                    desc=f"write demo into h5_file [{ep_idx - demo_chunk_size + 1} - {ep_idx} / {num_demos}]",
                    leave=False,
                    ncols=0,
                ):
                    if h5_file.get(key) is None:
                        h5_file.create_dataset(
                            key,
                            (capacity, *val[0].shape),
                            maxshape=(capacity, *val[0].shape),
                            chunks=(32, *val[0].shape),
                            compression="lzf",
                            dtype=check_dtype(val),
                        )
                    len_chunk_demos = ret["action"].shape[0]
                    for start in trange(
                        0, len_chunk_demos, batch_size, desc=f"write data for {key}", ncols=0, leave=False
                    ):
                        end = min(start + batch_size, len_chunk_demos)
                        h5_file[key][chunk_timestep + start : chunk_timestep + end] = val[start:end]
                chunk_timestep += len_chunk_demos
                pbar.update(demo_chunk_size)

    for key in h5_file.keys():
        v_shape = h5_file[key].shape[1:]
        h5_file[key].resize((total_timestep, *v_shape))

    h5_file.close()


def main():
    # Include argument parser
    parser = argparse.ArgumentParser(description="Convert npz files to hdf5.")
    parser.add_argument("--task-name", type=str, default="one_leg")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--input-dir", type=str, required=True, help="path to input files")
    parser.add_argument("--output-dir", type=str, required=True, help="path for saving output files")
    parser.add_argument("--num-demos", type=int, help="number of demonstrations.")
    parser.add_argument("--skip-frame", type=int, help="skip frame for making context.")
    parser.add_argument("--window-size", type=int, help="window size for making context.")
    parser.add_argument(
        "--capacity", default=3e5, type=float, help="expected capacity of current data: e.g.) max timestep * num-demos"
    )
    parser.add_argument(
        "--demo-chunk-size", type=int, default=8, help="how many demonstrations to write per one update."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = pathlib.Path(args.output_dir).expanduser()
    out_dir = out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.remove(out_dir / f"data_w{args.window_size}_s{args.skip_frame}.hdf5")
        print("delete past hdf5 file.")
    except OSError as e:
        print(f"error occurred. : {e}")
        pass
    shard_file = h5py.File(out_dir / f"data_w{args.window_size}_s{args.skip_frame}.hdf5", "a")
    ds_to_hdf5(
        data_dir=args.input_dir,
        task_name=args.task_name,
        num_demos=args.num_demos,
        demo_chunk_size=args.demo_chunk_size,
        h5_file=shard_file,
        split=args.split,
        window_size=args.window_size,
        skip_frame=args.skip_frame,
        capacity=args.capacity,
    )


if __name__ == "__main__":
    main()
