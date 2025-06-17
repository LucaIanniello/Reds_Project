import argparse
import pathlib
from collections import deque
from functools import partial

import clip
import h5py
import numpy as np
from tqdm import trange

from bpref_v2.data.instruct import get_furniturebench_instruct
from bpref_v2.utils.jax_utils import batch_to_jax
from bpref_v2.utils.utils import set_random_seed

from .reward_model_loader import load_reward_model

observation_dim = (224, 224, 3)
action_dim = 8

###############################################################################################################
######################################## Base funciton ########################################################
###############################################################################################################


def load_feature_fn(model_type, model, feature_type):
    if model_type == "CLIP":
        fn = extract_visual_text_feature
    elif model_type == "LIV":
        fn = extract_visual_text_feature

    return partial(fn, model=model)


def extract_feature(h5_file, args, feature_fn, feature_dim=512):
    image_keys = args.image_keys.split("|")
    N = h5_file["action"].shape[0]
    try:
        for ik in image_keys:
            del h5_file[f"{ik}_{args.model_type}"]
        del h5_file[f"instruct_{args.model_type}"]
    except Exception as e:
        print(f"Error occurred: {e}")

    for ik in image_keys:
        h5_file.create_dataset(f"{ik}_{args.model_type}", (N, args.window_size, feature_dim), dtype=np.float32)
    h5_file.create_dataset(f"instruct_{args.model_type}", (N, args.window_size, feature_dim), dtype=np.float32)

    demo_indicator = np.unique(h5_file["demo_idx"])
    assert (
        len(demo_indicator) - 1 >= args.num_demos
    ), f"[ERROR] This h5 file contains {len(demo_indicator) - 1} demonstrations, which is lower than {args.num_demos}"
    for i in trange(min(len(demo_indicator) - 1, args.num_demos), desc="extract feature", ncols=0):
        demo_range = range(demo_indicator[i], demo_indicator[i + 1])
        images, actions, skills = (
            {ik: h5_file[ik][demo_range].transpose(0, 1, 3, 4, 2) for ik in image_keys},
            h5_file["action"][demo_range],
            h5_file["skill"][demo_range],
        )
        features = feature_fn(images=images, actions=actions, skills=skills, args=args)
        for ik in image_keys:
            h5_file[f"{ik}_{args.model_type}"][demo_range] = features[ik]
        h5_file[f"instruct_{args.model_type}"][demo_range] = features["instruct"]


###############################################################################################################
############################# Reward computation function per algorithm #######################################
###############################################################################################################


def extract_visual_text_feature(model, **kwargs):
    images, skills, args = kwargs["images"], kwargs["skills"], kwargs["args"]
    task_name, image_keys = (
        args.task_name,
        args.image_keys.split("|"),
    )

    instructions = []
    for i in range(len(skills)):
        text_stack = []
        for j in range(len(skills[i])):
            token = (
                np.asarray(clip.tokenize(get_furniturebench_instruct(task_name, skills[i, j])))
                .astype(np.int32)
                .squeeze()
            )
            text_stack.append(token)
        instructions.append(np.stack(text_stack))
    instructions = np.asarray(instructions)

    features = {key: [] for key in ["color_image1", "color_image2", "instruct"]}
    batch_size = 64
    for i in trange(0, len(skills), batch_size, leave=False, ncols=0, desc=f"extract feature per batch {batch_size}"):
        _range = range(i, min(i + batch_size, len(skills)))
        batch = {
            "instruct": instructions[_range],
            "image": {ik: images[ik][_range] for ik in image_keys},
        }
        jax_input = batch_to_jax(batch)
        feature = model.get_visual_text_feature(jax_input)
        features["color_image2"].extend(feature["image"]["color_image2"])
        features["color_image1"].extend(feature["image"]["color_image1"])
        features["instruct"].extend(feature["instruct"])
    return {key: np.asarray(val) for key, val in features.items()}


def extract_video_feature(model, **kwargs):
    images, actions, args = kwargs["images"], kwargs["actions"], kwargs["args"]
    image_keys, window_size, skip_frame = (
        args.image_keys.split("|"),
        args.window_size,
        args.skip_frame,
    )

    stacked_images = {ik: [] for ik in image_keys}
    stacked_timesteps, stacked_attn_masks, stacked_actions = [], [], []

    image_stacks = {key: {ik: deque([], maxlen=window_size) for ik in image_keys} for key in range(skip_frame)}
    timestep_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}
    attn_mask_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}
    action_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}

    for _ in range(window_size):
        for j in range(skip_frame):
            for ik in image_keys:
                image_stacks[j][ik].append(np.zeros(observation_dim, dtype=np.uint8))
            timestep_stacks[j].append(0)
            attn_mask_stacks[j].append(0)
            action_stacks[j].append(np.zeros((action_dim,), dtype=np.float32))

    for i in range(len(actions)):
        mod = i % skip_frame
        image_stack, timestep_stack, attn_mask_stack, action_stack = (
            image_stacks[mod],
            timestep_stacks[mod],
            attn_mask_stacks[mod],
            action_stacks[mod],
        )
        for ik in image_keys:
            image_stack[ik].append(images[ik][i])
            stacked_images[ik].append(np.stack(image_stack[ik]))

        timestep_stack.append(i)
        mask = 1.0 if i != len(actions) - 1 else 0.0
        attn_mask_stack.append(mask)
        # token = np.asarray(clip.tokenize(get_furniturebench_instruct(task_name, skills[i], output_type="all"))).astype(np.int32)
        action_stack.append(actions[i])

        stacked_timesteps.append(np.stack(timestep_stack))
        stacked_attn_masks.append(np.stack(attn_mask_stack))
        # stacked_texts.append(token)
        stacked_actions.append(np.stack(action_stack))

    stacked_images = {ik: np.asarray(val) for ik, val in stacked_images.items()}
    stacked_timesteps = np.asarray(stacked_timesteps)
    stacked_attn_masks = np.asarray(stacked_attn_masks)
    # stacked_texts = np.asarray(stacked_texts)
    stacked_actions = np.asarray(stacked_actions)

    video_features = []
    batch_size = 64
    for i in trange(0, len(actions), batch_size, leave=False, ncols=0, desc="reward compute per batch"):
        _range = range(i, min(i + batch_size, len(actions)))
        batch = {
            # "instruct": stacked_texts[_range],
            "image": {ik: stacked_images[ik][_range] for ik in image_keys},
            "timestep": stacked_timesteps[_range],
            "attn_mask": stacked_attn_masks[_range],
            "action": stacked_actions[_range],
        }
        jax_input = batch_to_jax(batch)
        vf = list(np.asarray(model.get_video_feature(jax_input)))
        video_features.extend(vf)

    return np.asarray(video_features)


###############################################################################################################
################################## Main function for execution ################################################
###############################################################################################################


def main():
    # Include argument parser
    parser = argparse.ArgumentParser(description="label furniturebench demonstrations.")
    parser.add_argument("--task-name", type=str, default="one_leg")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--model-type", type=str, choices=["CLIP", "LIV"], default="LIV")
    parser.add_argument("--feature-type", type=str, choices=["video", "image"], default="image")
    parser.add_argument("--feature-dim", type=int, help="feature dimension", default=512)
    parser.add_argument("--ckpt-path", type=str, default="", help="ckpt of trained reward model.")
    parser.add_argument("--input-dir", type=str, required=True, help="path to input files")
    parser.add_argument("--skip-frame", type=int, help="skip frame.")
    parser.add_argument("--save-key", type=str, help="name for saving rewards in h5py file.")
    parser.add_argument("--window-size", type=int, help="window size.")
    parser.add_argument("--image-keys", type=str, help="image keys for reward computation", default="color_image2")
    parser.add_argument("--num-demos", type=int, help="number of demonstrations.")
    parser.add_argument("--seed", type=int, help="seed.", default=0)

    args = parser.parse_args()
    set_random_seed(args.seed)
    ckpt_path = pathlib.Path(args.ckpt_path).expanduser()

    # load h5py file to be labeled with reward model.
    out_dir = pathlib.Path(args.input_dir).expanduser()
    out_dir = out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(out_dir / f"data_w{args.window_size}_s{args.skip_frame}.hdf5", "a")

    # load_episodes(pathlib.Path(args.input_dir), shard_file)
    model = load_reward_model(args.model_type, ckpt_path)
    feature_fn = load_feature_fn(model_type=args.model_type, model=model, feature_type=args.feature_type)

    # make dataset for computed rewards in h5py files.
    extract_feature(h5_file, args, feature_fn, feature_dim=args.feature_dim)


if __name__ == "__main__":
    main()
