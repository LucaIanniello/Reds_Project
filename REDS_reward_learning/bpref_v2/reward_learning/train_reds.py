import copy
import os
import pprint
from collections import defaultdict

import absl.app
import absl.flags
import jax
import numpy as np
import torch
import transformers
from flax import jax_utils
from flax.jax_utils import prefetch_to_device
from flax.training.early_stopping import EarlyStopping
from rich.console import Console
from tqdm import trange

console = Console()

from bpref_v2.data.arp_furniturebench_dataset_inmemory_stream import (
    ARPFurnitureBenchDataset,
    worker_init_fn,
)
from bpref_v2.data.arp_robot_dataset_inmemory_stream import ARPRobotDataset
from bpref_v2.data.augmentations import single_pmap_image_aug_fn, tube_pmap_image_aug_fn
from bpref_v2.utils.jax_utils import batch_to_jax, next_rng
from bpref_v2.utils.utils import (
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    prefix_metrics,
    save_pickle,
    set_random_seed,
)
from bpref_v2.utils.viskit.logging import setup_logger

from .algos import REDSLearner

FLAGS_DEF = define_flags_with_default(
    env="halfcheetah-medium-v2",
    model_type="MLP",
    max_traj_length=1000,
    seed=42,
    data_seed=42,
    save_model=True,
    batch_size=64,
    num_workers=16,
    early_stop=False,
    min_delta=1e-4,
    patience=10,
    train_steps=5000,
    eval_steps=100,
    log_period=100,
    eval_period=1000,
    save_period=1000,
    comment="",
    furniturebench=ARPFurnitureBenchDataset.get_default_config(),
    robot=ARPRobotDataset.get_default_config(),
    use_failure=True,
    num_failure_demos=10,
    reds=REDSLearner.get_default_config(),
    logging=WandBLogger.get_default_config(),
    augmentations="none",
)


def main(_):
    #######################################################################################
    ################################ DEFINE HYPERPARAMETERS ###############################
    #######################################################################################

    FLAGS = absl.flags.FLAGS
    console.log("JAX process: %d / %d", jax.process_index(), jax.process_count())
    console.log("JAX local devices: %r", jax.local_devices())

    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    jax_process_index = jax.process_index()
    jax_process_count = jax.process_count()

    process_batch_size = FLAGS.batch_size // jax_process_count

    save_dir = FLAGS.logging.output_dir + "/" + str(FLAGS.model_type)
    save_dir += "/" + str(FLAGS.env) + "/"

    FLAGS.logging.group = f"{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += FLAGS.comment + "/"
    save_dir += "s" + str(FLAGS.seed)

    FLAGS.logging.output_dir = save_dir
    FLAGS.logging.project = "ICLR-REDS"

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    setup_logger(variant=variant, seed=FLAGS.seed, base_log_dir=save_dir, include_exp_prefix_sub_dir=False)
    wb_logger = WandBLogger(FLAGS.logging, variant=variant, enable=(jax_process_index == 0))

    set_random_seed(FLAGS.seed)

    step_per_log = FLAGS.log_period
    step_per_eval = FLAGS.eval_period
    step_per_save = FLAGS.save_period

    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)
    set_random_seed(FLAGS.seed)

    #######################################################################################
    ################################ DataLoader Setup #####################################
    #######################################################################################

    def create_dataset(config, split, demo_type, num_demos=None):
        dataset_class = ARPFurnitureBenchDataset if "furniturebench" in FLAGS.env else ARPRobotDataset
        data_config = copy.deepcopy(config)
        if num_demos is not None:
            data_config.num_demos = num_demos
        return dataset_class(
            update=data_config,
            split=split,
            start_offset_ratio=jax_process_index / jax_process_count,
            demo_type=demo_type,
        )

    def create_dataloader(dataset, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            prefetch_factor=2,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    if "furniturebench" in FLAGS.env:
        config = FLAGS.furniturebench
        action_dim = 8
    else:
        config = FLAGS.robot
        action_dim = 4

    image_size = config.image_size
    target_config = config

    train_dataset = create_dataset(config, "train", "success")
    val_dataset = create_dataset(config, "val", "success", num_demos=10)

    train_loader = create_dataloader(train_dataset, process_batch_size, target_config.num_workers)
    val_batch_size = min(process_batch_size, FLAGS.batch_size // jax_process_count)
    val_loader = create_dataloader(val_dataset, val_batch_size, max(target_config.num_workers // 4, 1))

    if FLAGS.use_failure:
        neg_config = copy.deepcopy(config)
        if "furniturebench" in FLAGS.env:
            neg_config.data_dir = config.finetune_data_dir
        else:
            neg_config.num_demos = FLAGS.num_failure_demos

        neg_train_dataset = create_dataset(neg_config, "train", "failure")
        neg_val_dataset = create_dataset(neg_config, "val", "failure", num_demos=5)

        neg_train_loader = create_dataloader(neg_train_dataset, process_batch_size, target_config.num_workers)
        neg_val_loader = create_dataloader(neg_val_dataset, val_batch_size, max(target_config.num_workers // 4, 1))

    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)
    aug_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)
    if FLAGS.augmentations == "crop|jitter":
        single_aug_fn = single_pmap_image_aug_fn(
            image_size=image_size,
            padding=int(image_size * (4 / 84)),
            window_size=target_config.window_size,
            jax_devices=jax_devices,
        )
        tube_aug_fn = tube_pmap_image_aug_fn(
            image_size=image_size,
            padding=int(image_size * (4 / 84)),
            window_size=target_config.window_size,
            jax_devices=jax_devices,
        )

    elif FLAGS.augmentations == "none":
        single_aug_fn, tube_aug_fn = lambda x, y: (x, y), lambda x, y: (x, y)

    def generate_batch(iterator, rng, split="train"):
        while True:
            for batch in iterator:
                reshape_fn = lambda x: x.numpy().reshape(n_devices, -1, *x.shape[1:])
                data = {}
                for key, value in batch.items():
                    if split == "train" and "image" in key:
                        first_image = next(iter(value.values()))
                        is_sequence = first_image.ndim in [5, 6]

                        if is_sequence:
                            if first_image.ndim == 6:
                                # Handle 6D case (likely batch, pearson, num_views, height, width, channels)
                                images = jax.tree_util.tree_map(
                                    lambda x: x.numpy().reshape(n_devices, -1, *x.shape[3:]), value
                                )
                            else:
                                # Handle 5D case as before (batch, num_views, height, width, channels)
                                images = jax.tree_util.tree_map(
                                    lambda x: x.numpy().reshape(n_devices, -1, *x.shape[2:]), value
                                )
                            aug_fn = tube_aug_fn
                        else:
                            images = jax.tree_util.tree_map(reshape_fn, value)
                            aug_fn = single_aug_fn

                        for img_key, img_val in images.items():
                            images[img_key], new_rng = aug_fn(img_val, rng)

                        if is_sequence:
                            if first_image.ndim == 6:
                                data[key] = jax.tree_util.tree_map(
                                    lambda x: x.reshape(
                                        n_devices,
                                        -1,
                                        target_config.pearson_size,
                                        target_config.window_size,
                                        *x.shape[2:],
                                    ),
                                    images,
                                )
                            else:
                                data[key] = jax.tree_util.tree_map(
                                    lambda x: x.reshape(n_devices, -1, target_config.window_size, *x.shape[2:]), images
                                )
                        else:
                            data[key] = images
                    else:
                        data[key] = jax.tree_util.tree_map(reshape_fn, value)

                if split == "train":
                    rng = new_rng
                yield data

    #######################################################################################
    ################################ DEFINE REWARD MODEL ##################################
    #######################################################################################
    num_steps = FLAGS.train_steps

    def configure_learner(model_type, config_dict, target_config, num_steps, image_size, action_dim, jax_devices):
        config = transformers.GPT2Config(**config_dict)
        config.warmup_steps = int(0.1 * num_steps)
        config.total_steps = num_steps
        config.image_keys = target_config.image_keys
        config.num_images = len(target_config.image_keys.split("|"))
        config.window_size = target_config.window_size
        image_dim = (image_size, image_size, 3)
        task_name = FLAGS.env.split("-", 1)[-1]

        learner_map = {
            "REDS": REDSLearner,
        }

        LearnerClass = learner_map.get(model_type)
        if not LearnerClass:
            raise ValueError(f"Unknown model type: {model_type}")

        return LearnerClass(config, task_name, image_dim, action_dim, jax_devices=jax_devices)

    reward_learner = configure_learner(
        FLAGS.model_type,
        getattr(FLAGS, FLAGS.model_type.lower(), FLAGS.model_type),
        target_config,
        num_steps,
        image_size,
        action_dim,
        jax_devices,
    )

    #######################################################################################
    ################################ GLOBAL TRAINING PHASE ################################
    #######################################################################################
    if os.path.exists(os.path.join(save_dir, "best_model.pkl")):
        console.print("Global model is already trained.")
    else:
        total_steps = num_steps + 1
        progress_bar = trange(total_steps, desc="Reward Learning Progress", ncols=0)
        metrics = defaultdict(list)

        train_loader.dataset.set_mode("global")
        val_loader.dataset.set_mode("global")
        train_iterator = prefetch_to_device(generate_batch(train_loader, aug_rng, split="train"), 2, jax_devices)
        val_iterator = prefetch_to_device(generate_batch(val_loader, aug_rng, split="val"), 2, jax_devices)
        if FLAGS.use_failure:
            neg_train_loader.dataset.set_mode("global")
            neg_val_loader.dataset.set_mode("global")
            neg_train_iterator = prefetch_to_device(
                generate_batch(neg_train_loader, aug_rng, split="train"), 2, jax_devices
            )
            neg_val_iterator = prefetch_to_device(generate_batch(neg_val_loader, aug_rng, split="val"), 2, jax_devices)

        early_stopper = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)
        for step in progress_bar:
            train_batch = next(train_iterator)
            neg_train_batch = next(neg_train_iterator) if FLAGS.use_failure else None
            if step % step_per_log == 0:
                metrics.clear()
            train_batch = batch_to_jax(train_batch)
            train_metrics, sharded_rng = reward_learner.train_step(train_batch, sharded_rng, neg_batch=neg_train_batch)
            for key, val in prefix_metrics(train_metrics, "reward").items():
                metrics[key].append(val)

            if step and (step % step_per_eval == 0 or step == total_steps - 1):
                for _ in trange(FLAGS.eval_steps, desc="Validation", ncols=0):
                    val_batch = batch_to_jax(next(val_iterator))
                    neg_val_batch = batch_to_jax(next(neg_val_iterator)) if FLAGS.use_failure else None
                    eval_metrics, sharded_rng = reward_learner.eval_step(
                        val_batch, sharded_rng, neg_batch=neg_val_batch
                    )
                    for key, val in prefix_metrics(eval_metrics, "reward_eval").items():
                        metrics[key].append(val)

                eval_loss = np.mean(metrics["reward_eval/total_loss"])
                has_improved, should_stop = early_stopper.update(eval_loss)
                log_metrics = {k: np.mean(v) for k, v in metrics.items()}
                log_metrics["step"] = step
                console.log("\n" + pprint.pformat(log_metrics) + "\n")
                if should_stop and FLAGS.early_stop:
                    console.print("Early stopping criteria met. Terminating training.")
                    break
                elif has_improved:
                    metrics["best_step"] = step
                    metrics["reward_eval/total_loss_best"] = eval_loss

            if step and step % step_per_log == 0:
                log_metrics = {k: np.mean(v) for k, v in metrics.items()}
                log_metrics["step"] = step
                console.log("\n" + pprint.pformat(log_metrics) + "\n")
                wb_logger.log(log_metrics, step=step)

            if FLAGS.save_model and step and step % step_per_save == 0:
                if jax_process_index == 0:
                    save_data = {
                        "step": step,
                        "state": jax.device_get(jax_utils.unreplicate(reward_learner._train_states)),
                        "config": reward_learner.config.to_dict(),
                    }
                    save_pickle(save_data, f"model_step{step}.pkl", save_dir)

    if FLAGS.save_model and jax_process_index == 0:
        save_data = {
            "step": step,
            "state": jax.device_get(jax_utils.unreplicate(reward_learner._train_states)),
            "config": reward_learner.config.to_dict(),
        }
        save_pickle(save_data, "last_model.pkl", save_dir)


if __name__ == "__main__":
    absl.app.run(main)
