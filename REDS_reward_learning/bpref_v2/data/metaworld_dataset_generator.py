# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

# Hack to import submodule. Must run this script from root directory, e.g.,
#   python data/run_scripted_policy.py
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple

# import argparse
import cv2
import h5py
import hydra
import numpy as np

# from metaworld_generalization.wrappers import make_wrapped_env
# from metaworld_generalization.wrappers import MetaWorldState
from metaworld import policies
from tqdm import tqdm

# from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
#                             ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from bpref_v2.data.replay_buffer import ReplayBufferStorage
from bpref_v2.envs.from_metaworld import MetaWorld

sys.path.append(".")


POLICIES = {
    "assembly-v2": policies.SawyerAssemblyV2Policy,
    "basketball-v2": policies.SawyerBasketballV2Policy,
    "bin-picking-v2": policies.SawyerBinPickingV2Policy,
    "box-close-v2": policies.SawyerBoxCloseV2Policy,
    "button-press-v2": policies.SawyerButtonPressV2Policy,
    "button-press-topdown-v2": policies.SawyerButtonPressTopdownV2Policy,
    "button-press-topdown-wall-v2": policies.SawyerButtonPressTopdownWallV2Policy,
    "button-press-wall-v2": policies.SawyerButtonPressWallV2Policy,
    "coffee-button-v2": policies.SawyerCoffeeButtonV2Policy,
    "coffee-pull-v2": policies.SawyerCoffeePullV2Policy,
    "coffee-push-v2": policies.SawyerCoffeePushV2Policy,
    "dial-turn-v2": policies.SawyerDialTurnV2Policy,
    "disassemble-v2": policies.SawyerDisassembleV2Policy,
    "door-close-v2": policies.SawyerDoorCloseV2Policy,
    "door-lock-v2": policies.SawyerDoorLockV2Policy,
    "door-open-v2": policies.SawyerDoorOpenV2Policy,
    "door-unlock-v2": policies.SawyerDoorUnlockV2Policy,
    "drawer-close-v2": policies.SawyerDrawerCloseV2Policy,
    "drawer-open-v2": policies.SawyerDrawerOpenV2Policy,
    "faucet-close-v2": policies.SawyerFaucetCloseV2Policy,
    "faucet-open-v2": policies.SawyerFaucetOpenV2Policy,
    "hammer-v2": policies.SawyerHammerV2Policy,
    "hand-insert-v2": policies.SawyerHandInsertV2Policy,
    "handle-press-v2": policies.SawyerHandlePressV2Policy,
    "handle-pull-v2": policies.SawyerHandlePullV2Policy,
    "handle-pull-side-v2": policies.SawyerHandlePullSideV2Policy,
    "lever-pull-v2": policies.SawyerLeverPullV2Policy,
    "peg-insert-side-v2": policies.SawyerPegInsertionSideV2Policy,
    "pick-out-of-hole-v2": policies.SawyerPickOutOfHoleV2Policy,
    "pick-place-v2": policies.SawyerPickPlaceV2Policy,
    "pick-place-wall-v2": policies.SawyerPickPlaceWallV2Policy,
    "plate-slide-v2": policies.SawyerPlateSlideV2Policy,
    "plate-slide-back-v2": policies.SawyerPlateSlideBackV2Policy,
    "plate-slide-side-v2": policies.SawyerPlateSlideSideV2Policy,
    "push-v2": policies.SawyerPushV2Policy,
    "push-back-v2": policies.SawyerPushBackV2Policy,
    "push-wall-v2": policies.SawyerPushWallV2Policy,
    "reach-v2": policies.SawyerReachV2Policy,
    "reach-wall-v2": policies.SawyerReachWallV2Policy,
    "shelf-place-v2": policies.SawyerShelfPlaceV2Policy,
    "soccer-v2": policies.SawyerSoccerV2Policy,
    "stick-pull-v2": policies.SawyerStickPullV2Policy,
    "stick-push-v2": policies.SawyerStickPushV2Policy,
    "sweep-into-v2": policies.SawyerSweepIntoV2Policy,
    "sweep-v2": policies.SawyerSweepV2Policy,
    "window-close-v2": policies.SawyerWindowCloseV2Policy,
    "window-open-v2": policies.SawyerWindowOpenV2Policy,
}


def _filename(cfg, image_obs_size: Tuple[int, int] = None):
    filename = f'{cfg.env.env_name}-{",".join(cfg.env.camera_name)}'
    if image_obs_size:
        filename += f"-{image_obs_size[0]}x{image_obs_size[1]}"
    filename += f"-a{cfg.policy.action_noise}"
    if cfg.env.factors:
        filename += "-" + "-".join(cfg.env.factors.keys())
    filename += f"-n{cfg.num_episodes}_{cfg.num_episodes_per_randomize}"
    return filename


def _get_image_obs_size(mode: str, image_obs_size):
    if mode == "render":
        return None
    elif mode == "save_video":
        return (600, 400)
    else:
        return image_obs_size


def get_shape_dtype(k, dummy_timestep):
    # if k == "gripper_open" or k == "time":
    #     v_dtype = np.float32
    #     v_shape = (1,)
    # elif k == "cont_action":
    #     v_dtype = np.float32
    #     v_shape = (8,)
    # elif k == "disc_action":
    #     v_dtype = np.int32
    #     v_shape = (7,)
    # elif k == "gripper_pose_delta":
    #     v_dtype = np.float32
    #     v_shape = (7,)
    # elif k == "task_id":
    #     v_dtype = h5py.special_dtype(vlen=np.dtype("uint8"))
    #     v_shape = ()
    # elif k == "variation_id":
    #     v_dtype = np.uint8
    #     v_shape = (1,)
    # elif k == "ignore_collisions":
    #     v_dtype = np.uint8
    #     v_shape = (1,)
    # else:
    v_dtype = dummy_timestep[k].dtype
    v_shape = dummy_timestep[k].shape
    return v_shape, v_dtype


def create_hdf5(dummy_timestep, hdf5_name, num_frames=4, size=1000000):
    h5_file = h5py.File(hdf5_name, "a")
    keys = list(dummy_timestep.keys())
    for k in keys:
        # if k == "observations":
        #     for camera_name in dummy_timestep[k]:
        #         v_shape, v_dtype = get_shape_dtype(camera_name, dummy_timestep[k])
        #         h5_file.create_dataset(
        #             camera_name, (size, num_frames, *v_shape), dtype=v_dtype, chunks=(16, num_frames, *v_shape)
        #         )
        # else:
        v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)
        h5_file.create_dataset(
            k,
            (size, num_frames, *v_shape),
            dtype=v_dtype,
            chunks=(16, num_frames, *v_shape),
        )
    return h5_file


def add_episode(h5_file, episode, num_frames, total_timestep):
    data = defaultdict(list)
    stack = defaultdict(lambda: deque([], maxlen=num_frames))
    for idx, timestep in enumerate(episode):
        for k, v in timestep.items():
            # if k == "observations":
            #     for camera_name, image in timestep[k].items():
            #         if idx == 0:
            #             # image = np.transpose(image, (1, 2, 0))
            #             stack[camera_name].extend([image] * num_frames)
            #         else:
            #             stack[camera_name].append(image)
            #         data[camera_name].append(np.stack(stack[camera_name]))
            # else:
            if idx == 0:
                stack[k].extend([v] * num_frames)
            else:
                stack[k].append(v)
            data[k].append(np.stack(stack[k]))

    len_episode = len(episode)
    for key, val in data.items():
        # if key == "image":
        #   for camera_name, image in val.items():
        #     h5_file[camera_name][timestep - len_episode:timestep] = image
        # else:
        try:
            h5_file[key][total_timestep - len_episode : total_timestep] = val
        except ValueError:
            print(f"[ERROR] h5_file length: {len_episode}")
            print(f"[ERROR] traj length: {len(val)}")
            raise


@hydra.main(config_path="./", config_name="metaworld_data")
def main(cfg):
    assert cfg.mode in ["render", "save_video", "save_buffer"], cfg.mode
    image_obs_size = _get_image_obs_size(cfg.mode, cfg.env.image_obs_size)

    #   factor_kwargs = {factor: cfg.env.factors[factor] for factor in cfg.factors}
    #   eval_factor_kwargs = {
    #       factor: cfg.env.eval_factors[factor] for factor in cfg.factors}

    # Create environment for collecting data.
    #   env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[cfg.env.env_name + '-goal-observable']
    #   env_kwargs = dict(
    #     camera_name=cfg.env.camera_name,
    #     get_image_obs=(image_obs_size is not None),
    #     image_obs_size=image_obs_size,
    #   )
    env = MetaWorld(name=cfg.env.env_name, size=image_obs_size, camera=cfg.env.camera_name[0])
    #   env = env_cls(seed=cfg.seed)
    # eval_env = MetaWorld(name=cfg.env.env_name, size=image_obs_size)
    #   eval_env = env_cls(sed=cfg.seed)

    #   env = make_wrapped_env(
    #       cfg.env.env_name,
    #       use_train_xml=True,
    #       factor_kwargs=factor_kwargs,
    #       image_obs_size=image_obs_size,
    #       camera_name=cfg.env.camera_name,
    #       observe_goal=cfg.env.observe_goal,
    #       random_init=cfg.env.random_init,
    #       default_num_resets_per_randomize=cfg.env.num_resets_per_randomize)

    #   # Create environment for sampling factor values for evaluation only.
    #   eval_env = make_wrapped_env(
    #       cfg.env.env_name,
    #       use_train_xml=False,
    #       factor_kwargs=eval_factor_kwargs,
    #       image_obs_size=image_obs_size,
    #       camera_name=cfg.env.camera_name,
    #       observe_goal=cfg.env.observe_goal,
    #       random_init=cfg.env.random_init,
    #       default_num_resets_per_randomize=cfg.env.num_resets_per_randomize)

    #   env.unwrapped.seed(cfg.seed)
    #   eval_env.unwrapped.seed(cfg.eval_seed)
    # Get policy
    action_space_ptp = env.action_space.high - env.action_space.low
    noise = np.ones(env.action_space.shape) * cfg.policy.action_noise
    policy = POLICIES[cfg.task_name]()

    # os.makedirs(cfg.output_dir, exist_ok=True)
    output_dir = os.path.join(cfg.output_dir, cfg.task_name)
    os.makedirs(output_dir, exist_ok=True)

    # Replay buffer output
    replay_buffer = None
    if cfg.mode == "save_buffer":
        data_specs = {
            # 'states': MetaWorldState(env).observation_space['state'],
            "observations": env.obs_space["state"],
            # "images": env.obs_space["image"],
            "actions": env.action_space,
        }
        if cfg.save_image:
            data_specs.update(dict(images=env.obs_space["image"]))
        replay_buffer = ReplayBufferStorage(data_specs, Path(output_dir))

    # Video output
    # video_writers = None
    if cfg.save_video and cfg.mode in ["save_video", "save_buffer"]:
        task_str = cfg.task_name
        # task_str = '%s:%s' % (
        #     cfg.task_name,
        #     '-'.join(cfg.factors))
        # video_writers = {}
        # for cam_name in cfg.env.camera_name:
        video_path = os.path.join(output_dir, f"video-{task_str}-{cfg.env.camera_name[0]}.avi")
        #     video_writers[cam_name] = cv2.VideoWriter(
        #         video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, image_obs_size
        #     )
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, image_obs_size)

    h5_file_train = None
    h5_file_train_name = os.path.join(output_dir, f"{cfg.env.env_name}_train.hdf5")
    # h5_file_train, h5_file_val = None, None
    # h5_file_train_name, h5_file_train_shuffled_name = (
    #     os.path.join(output_dir, f"{cfg.env.env_name}_train.hdf5"),
    #     os.path.join(output_dir, f"{cfg.env.env_name}_train_shuffled.hdf5"),
    # )

    # h5_file_val_name, h5_file_val_shuffled_name = (
    #     os.path.join(output_dir, f"{cfg.env.env_name}_val.hdf5"),
    #     os.path.join(output_dir, f"{cfg.env.env_name}_val_shuffled.hdf5"),
    # )
    try:
        os.remove(h5_file_train_name)
        # os.remove(h5_file_valname)
        # os.remove(h5_file_train_shuffled_name)
        # os.remove(h5_file_val_shuffled_name)
    except OSError as e:
        print(f"error occurred. : {e}")
        pass

    o = env.reset()
    data_factor_values = [env.current_factor_values if hasattr(env, "current_factor_values") else None]

    # eval_env.reset()
    # eval_factor_values = [eval_env.current_factor_values if hasattr(eval_env, "current_factor_values") else None]

    num_successful_ep = 0
    num_failed_ep = 0

    total_episodes = int(cfg.num_episodes * 1.0)
    total_timesteps = [0, 0]
    with tqdm(total=total_episodes, ncols=0, desc="collecting demonstrations") as pbar:
        while num_successful_ep < total_episodes:
            # Roll out an episode.
            ts = 0
            episode = []
            done = False
            while not done:
                a = policy.get_action(o["state"])
                a = np.random.normal(a, noise * action_space_ptp)

                next_o, r, done, info = env.step(a)

                time_step = {
                    # "images": o["image"],
                    # 'states': MetaWorldState.state(o['state']),
                    "observations": o["state"],
                    "actions": a.astype(np.float32),
                    "rewards": np.array([r], dtype=np.float32),
                    "discounts": np.array([1.0], dtype=np.float32),
                    "terminals": np.array(info["success"], dtype=np.int32),
                    "task_rewards": np.array([info["unscaled_reward"]], dtype=np.float32),
                }
                if cfg.save_image:
                    time_step.update(dict(images=o["image"]))
                episode.append(time_step)

                if cfg.mode == "render":
                    env.render()

                if video_writer:
                    # for key in o["image"]:
                    #     video_writer = video_writers[key]
                    #     # image = np.transpose(o['image'][key], (1, 2, 0))
                    #     image = o["image"][key]
                    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #     video_writer.write(image)
                    # video_writer = video_writers[0]
                    # image = np.transpose(o['image'][key], (1, 2, 0))
                    image = o["image"]
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    video_writer.write(image)

                o = next_o
                ts += 1
                # if num_successful_ep < cfg.num_episodes:
                #     total_timesteps[0] += 1
                # else:
                #     total_timesteps[1] += 1

            # If unsuccessful, do not save episode.
            if ts >= env._env.max_path_length:
                num_failed_ep += 1
                if cfg.debug:
                    print("Failed episode")
                o = env.reset()
            else:
                total_timesteps[0] += ts
                num_successful_ep += 1
                if cfg.debug:
                    print("Successful episode")

                if replay_buffer is not None:
                    if num_successful_ep <= cfg.num_episodes:
                        if h5_file_train is None:
                            dummy_timestep = episode[0]
                            h5_file_train = create_hdf5(
                                dummy_timestep,
                                h5_file_train_name,
                                num_frames=cfg.num_frames,
                            )
                            # h5_file_train_shuffled = create_hdf5(
                            #     dummy_timestep,
                            #     h5_file_train_shuffled_name,
                            #     num_frames=cfg.num_frames,
                            # )
                        add_episode(h5_file_train, episode, cfg.num_frames, total_timesteps[0])
                    # else:
                    #     if h5_file_val is None:
                    #         dummy_timestep = episode[0]
                    #         h5_file_val = create_hdf5(dummy_timestep, h5_file_val_name, num_frames=cfg.num_frames,)
                    #         h5_file_val_shuffled = create_hdf5(
                    #             dummy_timestep, h5_file_val_shuffled_name, num_frames=cfg.num_frames,
                    #         )
                    #     add_episode(h5_file_val, episode, cfg.num_frames, total_timesteps[1])

                pbar.update(1)
                o = env.reset()
                if hasattr(env, "current_factor_values"):
                    data_factor_values.append(env.current_factor_values)

                # eval_env.reset()
                # if hasattr(eval_env, "current_factor_values"):
                #     eval_factor_values.append(eval_env.current_factor_values)

    print(f"Finished {num_successful_ep} episodes ({num_failed_ep} fails).")

    for idx, (h5_file, _) in enumerate(
        [(h5_file_train, None)]
        # [(h5_file_train, h5_file_train_shuffled), (h5_file_val, h5_file_val_shuffled)]
        # [(h5_file_train, h5_file_train_shuffled)]
    ):
        for k in h5_file.keys():
            v_shape = h5_file[k].shape[2:]
            # v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)
            h5_file[k].resize((total_timesteps[idx], cfg.num_frames, *v_shape))
            # h5_file_shuffled[k].resize((total_timesteps[idx], cfg.num_frames, *v_shape))

    for h5_file, _ in [
        (h5_file_train, None),
        # (h5_file_train, h5_file_train_shuffled),
        # (h5_file_val, h5_file_val_shuffled),
    ]:
        # indices = list(range(len(h5_file["rewards"])))
        # random.shuffle(indices)
        # for i, j in enumerate(tqdm(indices, desc="shuffling", ncols=0)):
        #     for k in h5_file.keys():
        #         h5_file_shuffled[k][i] = h5_file[k][j]

        h5_file.close()
        # h5_file_shuffled.close()

    # Save factor values
    if cfg.mode == "save_buffer":
        factors_pkl_path = os.path.join(output_dir, "data_factor_values.pkl")
        with open(factors_pkl_path, "wb") as fp:
            pickle.dump(data_factor_values, fp)

        # factors_pkl_path = os.path.join(output_dir, "eval_factor_values.pkl")
        # with open(factors_pkl_path, "wb") as fp:
        #     pickle.dump(eval_factor_values, fp)

    if video_writer:
        video_writer.release()
    # if video_writers:
    #     for key, val in video_writers.items():
    #         val.release()


if __name__ == "__main__":
    main()
