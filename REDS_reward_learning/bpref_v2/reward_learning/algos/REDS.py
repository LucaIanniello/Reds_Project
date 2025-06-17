from functools import partial
from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import optax
from ml_collections import ConfigDict

from bpref_v2.data.instruct import TASK_TO_MAX_EPISODE_STEPS
from bpref_v2.networks.trans_reward_model import RPFRewardModel
from bpref_v2.third_party.openai.model import (
    IMAGE_RESOLUTION,
    load_clip_model,
    load_liv_model,
    normalize_image,
)
from bpref_v2.utils.jax_utils import (
    TrainState,
    cos_sim,
    next_rng,
    supervised_contrastive_loss,
    sync_state_fn,
)

from .core import RewardLearner


def video_score(video_features, text_features, logit=1.0):
    return (cos_sim(video_features, text_features) + 1) / 2 * logit


def text_score(video_features, text_features, logit=1.0):
    return (cos_sim(text_features, video_features) + 1) / 2 * logit


class REDSLearner(RewardLearner):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 1e-4
        config.optimizer_type = "adamw"
        config.scheduler_type = "CosineDecay"
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 768
        config.n_embd = config.embd_dim
        config.output_embd_dim = 512
        config.n_head = 8
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1

        config.activation = "relu"
        config.activation_final = "none"

        # transfer type
        config.transfer_type = "liv"

        # frozen visual/textual represntations
        config.window_size = 4
        config.visual_only = False
        config.frozen_visual = False
        config.frozen_textual = False

        # Make Bidirectional Transformer for temporal understanding.
        config.use_bidirectional = False

        config.image_keys = "image"
        config.num_images = 1

        # Optimizer parameters
        config.adam_beta1 = 0.9
        config.adam_beta2 = 0.98
        config.weight_decay = 0.02
        config.max_grad_norm = 1.0

        #### Local Training Parameters
        config.lambda_liv = 0.0
        config.gamma = 1.0
        config.epsilon = 1e-8

        #### Global Training Parameters
        # SupCon
        config.lambda_supcon = 0.0
        config.supcon_temperature = 0.1
        config.supcon_on_neg_batch = False
        # Future Pred
        config.lambda_future_pred = 0.0
        # EPIC
        config.discount = 0.99
        config.lambda_epic = 0.0
        config.epic_on_neg_batch = False
        config.epic_eps = 5e-2
        config.lambda_epic_reg = 1.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(
        self,
        config: ConfigDict = None,
        task_name: str = "one_leg",
        observation_dim: Sequence[int] = (224, 224, 3),
        action_dim: int = 8,
        state: flax.training.train_state.TrainState = None,
        jax_devices: Sequence[jax.Device] = None,
    ):
        self.config = config
        self.config.max_episode_steps = TASK_TO_MAX_EPISODE_STEPS[task_name.split("|")[0]]
        self.network = self._define_network(observation_dim, action_dim)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pvr_model, self.pvr_model_var = self._load_pvr_network()

        self._total_steps = 0
        if state is None:
            state = self._init_train_state(jax_devices)
            self.train_pmap = jax.pmap(self._train_step, axis_name="pmap", devices=jax_devices)
            self.eval_pmap = jax.pmap(self._eval_step, axis_name="pmap", devices=jax_devices)

        self._train_states = {}
        model_keys = ["trans"]
        self._model_keys = tuple(model_keys)
        self.load_state(state, jax_devices=jax_devices)

    def load_state(self, state, jax_devices=None, reset_optimizer=False, use_scheduler=True):
        if jax_devices is not None:
            if reset_optimizer:
                scheduler = optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=self.config.lr,
                    warmup_steps=self.config.warmup_steps,
                    decay_steps=self.config.total_steps,
                    end_value=0.0,
                )
                learning_rate = scheduler(self.config.lr) if use_scheduler else self.config.lr
                params = state["trans"].params
                tx = optax.chain(
                    optax.clip_by_global_norm(self.config.max_grad_norm),
                    optax.adamw(
                        # learning_rate=scheduler(self.config.lr),
                        learning_rate=learning_rate,
                        weight_decay=self.config.weight_decay,
                        b1=self.config.adam_beta1,
                        b2=self.config.adam_beta2,
                    ),
                )
                state = TrainState.create(params=params, batch_stats=None, tx=tx, apply_fn=self.network.apply)
            state = flax.jax_utils.replicate(state, jax_devices)
            self._train_states["trans"] = sync_state_fn(state)
        else:
            self._train_states["trans"] = state["trans"]

    def _load_pvr_network(self):
        if self.config.transfer_type == "liv":
            clip_model, clip_model_var, _ = load_liv_model()
            self.image_size = 224
        elif self.config.transfer_type.startswith("clip"):
            clip_type = self.config.transfer_type.split("_", 1)[-1]
            clip_model, clip_model_var, _ = load_clip_model(clip_type)
            self.image_size = IMAGE_RESOLUTION[clip_type]
        self.config.vision_embd_dim = clip_model.vision_features
        return clip_model, clip_model_var

    def _define_network(self, observation_dim, action_dim):
        return RPFRewardModel(
            config=self.config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            activation=self.config.activation,
            activation_final=self.config.activation_final,
        )

    def _init_train_state(self, jax_devices):
        num_patches = 1 + (
            self.image_size // self.pvr_model.vision_patch_size * self.image_size // self.pvr_model.vision_patch_size
        )
        variables = self.network.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.ones(
                (self.config.num_images, 1, self.config.window_size, num_patches, self.pvr_model.vision_features),
                # (self.config.num_images, 1, self.config.window_size, self.pvr_model.vision_features),
                dtype=jnp.float32,
            ),
            jnp.ones((1, self.config.output_embd_dim), dtype=jnp.int32),
        )

        variables = flax.core.frozen_dict.unfreeze(variables)
        params = flax.core.frozen_dict.unfreeze(variables["params"])
        batch_stats = (
            flax.core.frozen_dict.unfreeze(variables["batch_stats"])
            if variables.get("batch_stats") is not None
            else None
        )

        optimizer_class = {
            "adam": optax.adam,
            "adamw": partial(
                optax.adamw,
                weight_decay=self.config.weight_decay,
                b1=self.config.adam_beta1,
                b2=self.config.adam_beta2,
            ),
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            "CosineDecay": lambda lr: optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=0.0,
            ),
            "OnlyWarmup": lambda lr: optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(value=lr),
                ],
                [self.config.warmup_steps],
            ),
            "none": None,
        }[self.config.scheduler_type]

        partition_optimizers = {
            "trainable": optimizer_class(scheduler_class(self.config.lr)),
            "adapter": optimizer_class(self.config.lr * 0.1),
            # "phase_predictor": optimizer_class(self.config.lr),
            "frozen": optax.set_to_zero(),
        }

        def param_partition_condition(path, _):
            return "trainable"
            # if any([v in "-".join(path) for v in ["adapter", "residual_weight", "video_proj", "image_input"]]):
            #     return "adapter"
            # if any(
            #     [
            #         v in "-".join(path)
            #         for v in [
            #             "temporal_decoder",
            #         ]
            #     ]
            # ):
            #     return "trainable"
            # else:
            #     return "phase_predictor"

        param_partitions = flax.traverse_util.path_aware_map(param_partition_condition, params)
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.multi_transform(partition_optimizers, param_partitions),
        )
        states = TrainState.create(params=params, batch_stats=batch_stats, tx=tx, apply_fn=self.network.apply)
        return states

    def get_reward(self, batch, get_video_feature=False, get_text_feature=False):
        return self._get_reward_step(
            self._train_states, batch, get_video_feature=get_video_feature, get_text_feature=get_text_feature
        )

    @partial(jax.jit, static_argnames=("self", "get_video_feature", "get_text_feature"))
    def _get_reward_step(self, train_states, batch, get_video_feature=False, get_text_feature=False):
        obs = jnp.array(list(batch["image"].values()))
        timestep = batch["timestep"]
        attn_mask = batch["attn_mask"]

        param_dict = {"params": train_states["trans"].params}
        if train_states["trans"].batch_stats is not None:
            param_dict.update({"batch_stats": train_states["trans"].batch_stats})

        pvr_image_feature = self._get_pvr_feature(obs) if obs.ndim == 6 else obs
        image_feature = pvr_image_feature

        output = dict()
        tokens = jnp.array(list(batch["instruct"].values()))
        # task_embeddings = self._get_task_embedding(train_states["trans"].params, batch["phase"])
        task_embeddings = self._extract_text_feature(train_states["trans"].params, tokens, training=False)

        video_feature = self.network.apply(
            param_dict,
            image_feature,
            timestep,
            attn_mask=attn_mask,
            training=False,
            method=self.network.encode_video,
        )
        # cont_video_out_feature = self.network.apply(
        #     param_dict, video_feature, training=False, method=self.network.project_cont_video_feature
        # )
        cont_matrices = text_score(video_feature[:, -1], task_embeddings.mean(axis=-2))

        diag_cont_matrices = jnp.diagonal(cont_matrices, axis1=-2, axis2=-1)
        # NOTE: correct logit for preventing unexpected phase changes (2024-06-03)
        N = task_embeddings.shape[0]
        eps = 5e-2
        diag_cont_matrices += jnp.linspace(eps * (N - 1), 0.0, N).reshape(-1, 1)
        target_text_indices = jnp.argmax(diag_cont_matrices, axis=0, keepdims=True)
        target_text_indices = target_text_indices[..., None, None]
        task_embedding = jnp.take_along_axis(task_embeddings, target_text_indices, axis=0).squeeze(0)

        rewards = self._extract_score(
            train_states["trans"].params, video_feature, task_embedding.mean(axis=1), training=False
        )
        # liv_rewards = self._extract_score(param_dict, video_feature, task_embedding, None, training=False)

        # liv_rewards = video_score(liv_video_out_feature, jnp.mean(text_feature, axis=1))
        # epic_rewards = video_score(epic_video_out_feature, jnp.mean(text_feature, axis=1))

        # output is [-1, 1] scale, so change it to (0, 1) scale again.
        output["rewards"] = rewards
        output["target_text_indices"] = target_text_indices.squeeze([0, 2, 3])
        # output["rewards"] = (jnp.diag(liv_rewards) + 1) * jnp.diag(epic_rewards)
        # output["liv_rewards"] = jnp.diag(liv_rewards)
        # output["cont_rewards"] = jnp.diagonal(cont_matrices, axis1=-2, axis2=-1)
        output["cont_rewards"] = diag_cont_matrices
        # output["epic_rewards"] = jnp.diag(epic_rewards)

        # if get_video_feature:
        #     output["video_features"] = liv_video_out_feature
        # elif get_text_feature:
        #     output["text_features"] = jnp.mean(text_feature, axis=1)
        return output

    def get_visual_text_feature(self, batch):
        return self._get_visual_text_feature(self._train_states, batch)

    @partial(jax.jit, static_argnames=("self"))
    def _get_visual_text_feature(self, train_states, batch):
        obs = batch["image"]
        tokens = batch.get("instruct", None)

        res = {}
        param_dict = {"params": train_states["trans"].params}
        if train_states["trans"].batch_stats is not None:
            param_dict.update({"batch_stats": train_states["trans"].batch_stats})

        image_res = {}
        for key in obs.keys():
            image_feat = self.network.apply(param_dict, obs[key], method=self.network.get_clip_visual_feature)
            image_res[key] = image_feat
        res["image"] = image_res

        if tokens is not None:
            text_feat = self.network.apply(param_dict, tokens, method=self.network.get_clip_text_feature)
            res["instruct"] = text_feat

        return res

    def get_text_feature(self, batch):
        return self._get_text_feature(self._train_states, batch)

    @partial(jax.jit, static_argnames=("self"))
    def _get_text_feature(self, train_states, batch):
        tokens = batch["instruct"]

        param_dict = {"params": train_states["trans"].params}
        if train_states["trans"].batch_stats is not None:
            param_dict.update({"batch_stats": train_states["trans"].batch_stats})

        if tokens.shape[-1] != self.config.embd_dim:
            text_feature = self._get_ptr_feature(tokens)
        else:
            text_feature = tokens

        # encoding images using adapter
        text_feature = self.network.apply(param_dict, text_feature, training=False, method=self.network.encode_text)
        return text_feature.mean(axis=-2)

    @partial(jax.jit, static_argnames=("self"))
    def _eval_pref_step(self, train_states, rng, batch):
        return None

    @partial(jax.jit, static_argnames=("self"))
    def _train_pref_step(self, train_states, rng, batch):
        return None

    def compute_pearson_distance(self, rewa: jnp.ndarray, rewb: jnp.ndarray, dist=None) -> float:
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

        # def _check_dist(dist: jnp.ndarray) -> None:
        #     assert jnp.allclose(jnp.sum(dist), 1)
        #     assert jnp.all(dist >= 0)

        def _center(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            mean = jnp.average(x, weights=weights)
            return x - mean

        dist = jnp.ones_like(rewa) / jnp.prod(jnp.asarray(rewa.shape))
        # _check_dist(dist)
        assert rewa.shape == dist.shape
        assert rewa.shape == rewb.shape, f"{rewa.shape} != {rewb.shape}"

        dist = dist.flatten()
        rewa = _center(rewa.flatten(), dist)
        rewb = _center(rewb.flatten(), dist)

        vara = jnp.average(jnp.square(rewa), weights=dist)
        varb = jnp.average(jnp.square(rewb), weights=dist)
        cov = jnp.average(rewa * rewb, weights=dist)
        corr = cov / (jnp.sqrt(vara) * jnp.sqrt(varb) + 1e-10)
        corr = jnp.where(corr > 1.0, 1.0, corr)
        return jnp.sqrt(0.5 * (1 - corr))

    def _get_pvr_feature(self, images):
        original_shape = images.shape[:-3]
        images = jnp.reshape(images, (-1,) + images.shape[-3:])
        images = (images / 255.0).astype(jnp.float32)
        if images.shape[-3] != 224:
            images = jax.image.resize(
                images, (images.shape[0], 224, 224, images.shape[-1]), method="bicubic"
            )  # to meet the input size of the clip model
        images = normalize_image(images)
        image_feature_map = self.pvr_model.apply(
            self.pvr_model_var,
            images,
            method=self.pvr_model.encode_image,
            normalize=False,
        )
        # image_feat = jnp.reshape(image_feature_map[:, 0], original_shape + (-1,))
        # return image_feat

        image_feature_map = jnp.reshape(image_feature_map, original_shape + image_feature_map.shape[-2:])
        return image_feature_map

    def _get_ptr_feature(self, tokens):
        original_shape = tokens.shape[:-1]
        tokens = jnp.reshape(tokens, (-1, tokens.shape[-1]))
        text_feat = self.pvr_model.apply(self.pvr_model_var, tokens, method=self.pvr_model.encode_text, normalize=False)
        text_feat = jnp.reshape(text_feat, original_shape + (-1,))
        return text_feat

    def _extract_visual_feature(self, params, obs, timestep, attn_mask, rng, training=False, encode_tp="cont"):
        num_image, batch_size, seq_length = obs.shape[:3]
        if obs.ndim == 6:
            pvr_image_feature = jax.lax.stop_gradient(self._get_pvr_feature(obs))
        else:
            pvr_image_feature = obs

        image_feature = pvr_image_feature

        video_feature = self.network.apply(
            {"params": params},
            image_feature,
            timestep,
            training=training,
            attn_mask=attn_mask,
            rngs={"dropout": rng},
            out_features=True,
            method=self.network.encode_video,
        )
        return video_feature, image_feature

    def _project_video_feature(self, params, video_feature, rng, training=False, encode_tp="cont"):
        assert encode_tp in ["nfp", "cont", "epic", "liv"], f"choose appropriate encode_tp: {encode_tp}"
        if encode_tp == "nfp":
            return self.network.apply(
                {"params": params},
                video_feature,
                training=training,
                rngs={"dropout": rng},
                method=self.network.project_nfp_video_feature,
            )
        if encode_tp == "cont":
            return self.network.apply(
                {"params": params},
                video_feature,
                training=training,
                rngs={"dropout": rng},
                method=self.network.project_cont_video_feature,
            )
        if encode_tp == "epic":
            return self.network.apply(
                {"params": params},
                video_feature,
                training=training,
                rngs={"dropout": rng},
                method=self.network.project_epic_video_feature,
            )
        if encode_tp == "liv":
            return self.network.apply(
                {"params": params},
                video_feature,
                training=training,
                rngs={"dropout": rng},
                method=self.network.project_liv_video_feature,
            )

    def _get_task_embedding(self, params, phases):
        return self.network.apply(
            {"params": params},
            phases,
            method=self.network.get_task_embedding,
        )

    def _extract_text_feature(self, params, tokens, training=False):
        if tokens.shape[-1] != self.config.embd_dim:
            text_feature = jax.lax.stop_gradient(self._get_ptr_feature(tokens))
        else:
            text_feature = tokens
        # encoding text using adapter
        text_feature = self.network.apply(
            {"params": params}, text_feature, training=training, method=self.network.encode_text
        )
        return text_feature

    def _extract_score(self, params, video_out_feature, phases, training=False):
        reward = self.network.apply(
            {"params": params}, video_out_feature, phases, training=training, method=self.network.predict_reward
        )
        return reward

    def _compute_epic_loss(self, params, batch, rng, neg_batch=None, training=False):
        aux = {}
        rng, key = jax.random.split(rng)

        def process_batch(b, prefix=""):
            images = jnp.asarray(list(b["image"].values()))
            pearson_images = jnp.asarray(list(b["pearson_image"].values())).reshape(
                images.shape[0], -1, *images.shape[2:]
            )
            phase = self._extract_text_feature(params, b["instruct"][:, -1])
            pearson_phase = self._extract_text_feature(
                params, b["pearson_instruct"].reshape(-1, b["pearson_instruct"].shape[-1])
            )
            reward = b["reward"][:, -1]
            pearson_reward = b["pearson_reward"].reshape(-1)
            attn_mask = b["attn_mask"]
            return {
                f"{prefix}images": images,
                f"{prefix}pearson_images": pearson_images,
                f"{prefix}phase": phase,
                f"{prefix}pearson_phase": pearson_phase,
                f"{prefix}reward": reward,
                f"{prefix}pearson_reward": pearson_reward,
                f"{prefix}attn_mask": attn_mask,
            }

        data = process_batch(batch, "pos_")
        if neg_batch is not None and self.config.epic_on_neg_batch:
            data.update(process_batch(neg_batch, "neg_"))
            for key in ["images", "pearson_images", "phase", "pearson_phase", "reward", "pearson_reward", "attn_mask"]:
                data[key] = jnp.concatenate(
                    [data[f"pos_{key}"], data[f"neg_{key}"]], axis=1 if key.endswith("images") else 0
                )
            batch_size = 2 * batch["reward"].shape[0]
        else:
            for key in ["images", "pearson_images", "phase", "pearson_phase", "reward", "pearson_reward", "attn_mask"]:
                data[key] = data[f"pos_{key}"]
            batch_size = batch["reward"].shape[0]

        def extract_features(images, rng, attn_mask=None):
            rng, key = jax.random.split(rng)
            return self._extract_visual_feature(params, images, None, attn_mask, key, training=training)[0], key

        video_feature, rng = extract_features(data["images"], rng, data["attn_mask"])
        canonical_video_feature, rng = extract_features(data["pearson_images"], rng)

        Rss = self._extract_score(params, video_feature, data["phase"], training=training)
        canon_Rss = self._extract_score(
            params, canonical_video_feature, data["pearson_phase"], training=training
        ).reshape(batch_size, -1)

        canon_video = Rss.squeeze() - canon_Rss.mean(axis=1)
        canon_gt_reward = data["reward"] - data["pearson_reward"].reshape(batch_size, -1).mean(axis=1)

        epic_loss = self.compute_pearson_distance(canon_video, canon_gt_reward)
        aux["epic_loss"] = epic_loss

        if self.config.lambda_epic_reg > 0.0:
            curr_video_feature, rng = extract_features(data["pos_images"], rng, data["pos_attn_mask"])
            next_video_feature, rng = extract_features(jnp.asarray(list(batch["random_next_image"].values())), rng)

            curr_R = self._extract_score(params, curr_video_feature, data["pos_phase"], training=training)
            next_phase = self._extract_text_feature(params, batch["random_next_instruct"][:, -1], training=training)
            next_R = self._extract_score(params, next_video_feature, next_phase, training=training)

            reg_loss = jnp.sum(jnp.maximum(0, self.config.epic_eps - next_R + curr_R))
            epic_loss += self.config.lambda_epic_reg * reg_loss
            aux.update(dict(reg_loss=reg_loss, total_epic_loss=epic_loss))

        return epic_loss, aux

    def _compute_supcon_loss(
        self,
        params,
        rng,
        video_feature,
        text_feature,
        phases,
        neg_video_feature=None,
        neg_text_feature=None,
        neg_phases=None,
        training=False,
    ):
        supcon_loss, vt_supcon_loss, neg_vt_supcon_loss = 0.0, 0.0, 0.0
        rng, key = jax.random.split(rng)
        cont_video_out_feature = video_feature[:, -1]

        vt_labels = jnp.concatenate(
            [
                phases[:, -1],
                phases[:, -1],
            ],
            axis=0,
        )
        vt_concat_features = jnp.concatenate(
            [cont_video_out_feature, text_feature[:, -1]],
            axis=0,
        )
        vt_supcon_loss = supervised_contrastive_loss(
            vt_concat_features, labels=vt_labels, temperature=self.config.supcon_temperature
        )
        supcon_loss = vt_supcon_loss

        if neg_video_feature is not None and neg_phases is not None and self.config.supcon_on_neg_batch:
            rng, key = jax.random.split(rng)
            neg_cont_video_out_feature = neg_video_feature[:, -1]

            neg_vt_labels = jnp.concatenate(
                [
                    neg_phases[:, -1],
                    neg_phases[:, -1],
                ],
                axis=0,
            )
            neg_vt_concat_features = jnp.concatenate(
                [neg_cont_video_out_feature, neg_text_feature[:, -1]],
                axis=0,
            )
            neg_vt_supcon_loss = supervised_contrastive_loss(
                neg_vt_concat_features, labels=neg_vt_labels, temperature=self.config.supcon_temperature
            )

            supcon_loss += neg_vt_supcon_loss

        return supcon_loss, dict(
            vt_supcon_loss=vt_supcon_loss,
            neg_vt_supcon_loss=neg_vt_supcon_loss,
            supcon_loss=supcon_loss,
        )

    def loss_fn(self, params, batch, rng, neg_batch=None, training=False, task_mask: jnp.ndarray = None):
        obs = jnp.array(list(batch["image"].values()))
        num_image, batch_size, seq_length = obs.shape[:3]

        video_feature, _ = self._extract_visual_feature(
            params, obs, batch["timestep"], batch["attn_mask"], rng, training=training
        )
        rng, key = jax.random.split(rng)
        text_feature = self._extract_text_feature(params, batch["instruct"], training=training)

        if neg_batch is not None:
            rng, key = jax.random.split(rng)
            neg_video_feature, *_ = self._extract_visual_feature(
                params,
                jnp.array(list(neg_batch["image"].values())),
                neg_batch["timestep"],
                neg_batch["attn_mask"],
                key,
                training=training,
            )
            neg_text_feature = self._extract_text_feature(params, neg_batch["instruct"], training=training)
            neg_phases = neg_batch["phase"]
        else:
            neg_video_feature, neg_text_feature, neg_phases = None, None, None

        # choose random task
        if training:
            epic_loss, supcon_loss = 0.0, 0.0
            epic_aux, supcon_aux = {}, {}
            if self.config.lambda_epic > 0:
                epic_loss, epic_aux = self._compute_epic_loss(
                    params,
                    batch,
                    rng,
                    neg_batch=neg_batch,
                    training=training,
                )
            if self.config.lambda_supcon > 0:
                supcon_loss, supcon_aux = self._compute_supcon_loss(
                    params,
                    rng,
                    video_feature,
                    text_feature,
                    batch["phase"],
                    neg_video_feature=neg_video_feature,
                    neg_text_feature=neg_text_feature,
                    neg_phases=neg_phases,
                    training=training,
                )
            rng, key = jax.random.split(rng)
            total_loss = jnp.asarray([epic_loss, supcon_loss])
            coeffs = jnp.array([self.config.lambda_epic, self.config.lambda_supcon])
            loss = jnp.sum(coeffs * task_mask * total_loss)
            aux = {
                **epic_aux,
                **supcon_aux,
                "loss": loss,
                "total_loss": jnp.asarray([epic_loss, supcon_loss]).mean(),
                "task_mask": jnp.argmax(task_mask),
            }
        else:
            epic_loss, supcon_loss = 0.0, 0.0
            epic_aux, supcon_aux = {}, {}
            if self.config.lambda_epic > 0:
                epic_loss, epic_aux = self._compute_epic_loss(
                    params,
                    batch,
                    rng,
                    neg_batch=neg_batch,
                    training=training,
                )
            if self.config.lambda_supcon > 0:
                supcon_loss, supcon_aux = self._compute_supcon_loss(
                    params,
                    rng,
                    video_feature,
                    text_feature,
                    batch["phase"],
                    neg_video_feature=neg_video_feature,
                    neg_text_feature=neg_text_feature,
                    neg_phases=neg_phases,
                    training=training,
                )
            total_loss = jnp.asarray([epic_loss, supcon_loss])
            aux = {**epic_aux, **supcon_aux, "total_loss": total_loss.mean()}
            loss = total_loss.mean()

        return loss, aux

    def train_step(self, batch, rng, neg_batch=None):
        self._total_steps += 1
        self._train_states, metrics, rng = self.train_pmap(self._train_states, batch, rng, neg_batch=neg_batch)
        return metrics, rng

    @partial(jax.jit, static_argnames=("self"))
    def _train_step(self, train_states, batch, rng, neg_batch=None):
        next_rng, split_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        task_mask = jnp.eye(2)[jax.random.randint(split_rng, (1,), 0, 2)]
        (loss, metrics), grads = jax.lax.pmean(
            grad_fn(
                train_states["trans"].params, batch, split_rng, neg_batch=neg_batch, training=True, task_mask=task_mask
            ),
            axis_name="pmap",
        )

        new_train_states = {"trans": train_states["trans"].apply_gradients(grads=grads)}
        return new_train_states, metrics, next_rng

    def eval_step(self, batch, rng, neg_batch=None):
        metrics, rng = self.eval_pmap(self._train_states, batch, rng, neg_batch=neg_batch)
        return metrics, rng

    def _eval_step(self, train_states, batch, rng, neg_batch=None):
        next_rng, split_rng = jax.random.split(rng)
        _, metrics = jax.lax.pmean(
            self.loss_fn(train_states["trans"].params, batch, split_rng, neg_batch=neg_batch, training=False),
            axis_name="pmap",
        )
        return metrics, next_rng
