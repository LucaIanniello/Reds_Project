import argparse
import pathlib
import pickle
from collections import deque
from functools import partial

import clip
import h5py
import numpy as np
from tqdm import trange

from bpref_v2.data.instruct import get_furniturebench_instruct
from bpref_v2.reward_learning.algos import CLIPLearner, REDSLearner
from bpref_v2.utils.jax_utils import batch_to_jax
from bpref_v2.utils.utils import set_random_seed

observation_dim = (224, 224, 3)
action_dim = 8

###############################################################################################################
######################################## Base function ########################################################
###############################################################################################################


def load_reward_model(rm_type, ckpt_path):
    if not ckpt_path.is_dir():
        with ckpt_path.open("rb") as fin:
            checkpoint_data = pickle.load(fin)
            config, state = checkpoint_data["config"], checkpoint_data["state"]

    if rm_type == "REDS":
        import transformers

        cfg = transformers.GPT2Config(**config)
        reward_model = REDSLearner(cfg, observation_dim, action_dim, state=state)

    elif rm_type == "CLIP":
        cfg = CLIPLearner.get_default_config()
        cfg.transfer_type = "clip_vit_b16"
        reward_model = CLIPLearner(cfg)

    elif rm_type == "LIV":
        cfg = CLIPLearner.get_default_config()
        cfg.transfer_type = "liv"
        reward_model = CLIPLearner(cfg)

    return reward_model


def load_reward_fn(rm_type, reward_model):
    if rm_type == "REDS":
        fn = compute_reds_reward
    elif rm_type == "CLIP":
        fn = compute_clip_reward
    elif rm_type == "LIV":
        fn = compute_clip_reward

    return partial(fn, reward_model=reward_model)


def label_reward(h5_file, args, reward_fn):
    image_keys, reward_key = (args.image_keys.split("|"), f"reward_{args.save_key}")
    window_size = args.window_size
    N = h5_file["action"].shape[0]
    try:
        print(f"delete previous [{reward_key}] in h5_file.")
        del h5_file[reward_key]
    except Exception as e:
        print(f"{reward_key} doesn't exists: {e}")
    h5_file.create_dataset(reward_key, (N, window_size), dtype=np.float32)

    demo_indicator = np.unique(h5_file["demo_idx"])
    assert (
        len(demo_indicator) - 1 >= args.num_demos
    ), f"[ERROR] This h5 file contains {len(demo_indicator) - 1} demonstrations, which is lower than {args.num_demos}"
    for i in trange(min(len(demo_indicator) - 1, args.num_demos), desc="labeling reward", ncols=0):
        demo_range = range(demo_indicator[i], demo_indicator[i + 1])
        images, actions, skills, timesteps, attn_masks = (
            {ik: h5_file[ik][demo_range] for ik in image_keys},
            h5_file["action"][demo_range],
            h5_file["skill"][demo_range],
            h5_file["timestep"][demo_range],
            h5_file["attn_mask"][demo_range],
        )
        rewards = reward_fn(
            images=images, actions=actions, skills=skills, timesteps=timesteps, attn_masks=attn_masks, args=args
        )
        demo_indicator = np.unique(h5_file["demo_idx"])

        reward_stack = deque([], maxlen=window_size)
        for _ in range(window_size):
            reward_stack.append(np.asarray(0).astype(np.float32))

        stacked_reward = []
        for ts in range(len(rewards)):
            reward_stack.append(rewards[ts])
            stacked_reward.append(np.stack(reward_stack))
        h5_file[reward_key][demo_range] = np.asarray(stacked_reward)


def compute_rtg(h5_file, args):
    reward_key, rtg_key = f"reward_{args.save_key}", f"rtg_{args.save_key}"
    window_size = args.window_size
    N = h5_file["action"].shape[0]
    try:
        print(f"delete previous [{rtg_key}] in h5_file.")
        del h5_file[rtg_key]
    except Exception as e:
        print(f"{rtg_key} doesn't exists: {e}")
    h5_file.create_dataset(rtg_key, (N, window_size), dtype=np.float32)

    def discount_cumsum(x, gamma=1.0):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    demo_indicator = np.unique(h5_file["demo_idx"])
    for i in trange(min(len(demo_indicator) - 1, args.num_demos), desc="compute return-to-go.", ncols=0):
        demo_range = range(demo_indicator[i], demo_indicator[i + 1])
        rtg = discount_cumsum(h5_file[reward_key][demo_range, -1])
        rtg_stack = deque([], maxlen=window_size)
        for _ in range(window_size):
            rtg_stack.append(np.asarray(rtg[0]).astype(np.float32))
        stacked_rtg = []
        for ts in range(len(rtg)):
            rtg_stack.append(rtg[ts])
            stacked_rtg.append(np.stack(rtg_stack))
        h5_file[rtg_key][demo_range] = np.asarray(stacked_rtg)


###############################################################################################################
############################# Reward computation function per algorithm #######################################
###############################################################################################################


def compute_reds_reward(reward_model, **kwargs):
    images, actions, skills, timesteps, attn_masks, args = (
        kwargs["images"],
        kwargs["actions"],
        kwargs["skills"],
        kwargs["timesteps"],
        kwargs["attn_masks"],
        kwargs["args"],
    )
    task_name, image_keys = (
        args.task_name,
        args.image_keys.split("|"),
    )
    stacked_texts = []

    for i in range(len(actions)):
        # text_stack = text_stacks[mod]
        token = np.asarray(
            clip.tokenize(get_furniturebench_instruct(task_name, skills[i, -1], output_type="all"))
        ).astype(np.int32)
        stacked_texts.append(token)

    stacked_texts = np.asarray(stacked_texts)

    rewards = []
    batch_size = 64
    for i in trange(0, len(actions), batch_size, leave=False, ncols=0, desc="reward compute per batch"):
        _range = range(i, min(i + batch_size, len(actions)))
        batch = {
            "instruct": stacked_texts[_range],
            "image": {ik: images[ik][_range] for ik in image_keys},
            "timestep": timesteps[_range],
            "attn_mask": attn_masks[_range],
            "action": actions[_range],
        }
        jax_input = batch_to_jax(batch)
        reward = list(np.asarray(reward_model.get_reward(jax_input)))
        rewards.extend(reward)
    return np.asarray(rewards)


def compute_clip_reward(reward_model, **kwargs):
    images, skills, args = kwargs["images"], kwargs["skills"], kwargs["args"]
    task_name, image_keys = (
        args.task_name,
        args.image_keys.split("|"),
    )

    instructions = []
    for i in range(len(skills)):
        token = np.asarray(
            clip.tokenize(get_furniturebench_instruct(task_name, skills[i, -1], output_type="all"))
        ).astype(np.int32)
        instructions.append(token)
    instructions = np.asarray(instructions)

    rewards = []
    batch_size = 64
    for i in trange(0, len(skills), batch_size, leave=False, ncols=0, desc=f"reward compute per batch {batch_size}"):
        _range = range(i, min(i + batch_size, len(skills)))
        batch = {
            "instruct": instructions[_range],
            "image": {ik: images[ik][_range] for ik in image_keys},
        }
        jax_input = batch_to_jax(batch)
        reward = list(np.asarray(reward_model.get_reward(jax_input)))
        rewards.extend(reward)
    return np.asarray(rewards)


###############################################################################################################
################################## Main function for execution ################################################
###############################################################################################################


def main():
    # Include argument parser
    parser = argparse.ArgumentParser(description="label furniturebench demonstrations.")
    parser.add_argument("--task-name", type=str, default="one_leg")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--rm-type", type=str, choices=["ARP-V1", "REDS", "CLIP", "LIV"], default="REDS")
    parser.add_argument("--ckpt-path", type=str, default="", help="ckpt of trained reward model.")
    parser.add_argument("--input-dir", type=str, required=True, help="path to input files")
    parser.add_argument("--skip-frame", type=int, help="skip frame.")
    parser.add_argument("--save-key", type=str, help="name for saving rewards in h5py file.")
    parser.add_argument("--window-size", type=int, help="window size.")
    parser.add_argument("--image-keys", type=str, help="image keys for reward computation", default="color_image2")
    parser.add_argument("--num-demos", type=int, help="number of demonstrations.")
    parser.add_argument("--seed", type=int, help="seed.")

    args = parser.parse_args()
    set_random_seed(args.seed)
    ckpt_path = pathlib.Path(args.ckpt_path).expanduser()

    # load h5py file to be labeled with reward model.
    out_dir = pathlib.Path(args.input_dir).expanduser()
    out_dir = out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(out_dir / "data.hdf5", "a")

    # load_episodes(pathlib.Path(args.input_dir), shard_file)
    reward_model = load_reward_model(rm_type=args.rm_type, ckpt_path=ckpt_path)
    reward_fn = load_reward_fn(rm_type=args.rm_type, reward_model=reward_model)

    # make dataset for computed rewards in h5py files.
    label_reward(
        h5_file,
        args,
        reward_fn,
    )
    # with labeled reward, compute return-to-go.
    compute_rtg(h5_file, args)


if __name__ == "__main__":
    main()
