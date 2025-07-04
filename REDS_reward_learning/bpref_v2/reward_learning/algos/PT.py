from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from bpref_v2.networks.trans_reward_model import TransRewardModel
from bpref_v2.utils.jax_utils import cross_ent_loss, next_rng, value_and_multi_grad

from .core import RewardLearner


class PTLearner(RewardLearner):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 1e-4
        config.optimizer_type = "adamw"
        config.scheduler_type = "CosineDecay"
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256

        config.train_type = "mean"

        # Weighted Sum option
        config.use_weighted_sum = False

        config.activation = "relu"
        config.activation_final = "none"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def _define_network(self, observation_dim, action_dim):
        return TransRewardModel(
            config=self.config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            activation=self.config.activation,
            activation_final=self.config.activation_final,
        )

    def _init_train_state(self):
        params = self.network.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.zeros((10, 25, self.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32),
        )
        self._train_states["trans"] = TrainState.create(params=params, tx=self.tx, apply_fn=None)
        model_keys = ["trans"]
        self._model_keys = tuple(model_keys)

    @partial(jax.jit, static_argnames=("self"))
    def _get_reward_step(self, train_states, batch):
        obs = batch["observations"]
        act = batch["actions"]
        timestep = batch["timestep"]
        attn_mask = batch["attn_mask"]

        train_params = {key: train_states[key].params for key in self.model_keys}
        pred, attn_weights = self.network.apply(
            train_params["trans"], obs, act, timestep, attn_mask=attn_mask, reverse=False
        )
        return pred["value"], attn_weights[-1]

    @partial(jax.jit, static_argnames=("self"))
    def _eval_pref_step(self, train_states, rng, batch):
        def loss_fn(train_params, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            timestep_1 = batch["timestep"]
            timestep_2 = batch["timestep_2"]
            labels = batch["labels"]

            B, T, _ = act_1.shape[-3:]

            rng, _ = jax.random.split(rng)

            pred_1, _ = self.network.apply(
                train_params["trans"], obs_1, act_1, timestep_1, training=False, attn_mask=None, rngs={"dropout": rng}
            )
            pred_2, _ = self.network.apply(
                train_params["trans"], obs_2, act_2, timestep_2, training=False, attn_mask=None, rngs={"dropout": rng}
            )

            if self.config.use_weighted_sum:
                pred_1 = pred_1["weighted_sum"]
                pred_2 = pred_2["weighted_sum"]
            else:
                pred_1 = pred_1["value"]
                pred_2 = pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            loss = cross_ent_loss(logits, label_target)
            cse_loss = loss
            loss_collection["trans"] = loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_cse_loss=aux_values["cse_loss"],
            eval_loss=aux_values["loss"],
        )

        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_pref_step(self, train_states, rng, batch):
        def loss_fn(train_params, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            timestep_1 = batch["timestep"]
            timestep_2 = batch["timestep_2"]
            labels = batch["labels"]

            B, T, _ = act_1.shape[-3:]

            rng, _ = jax.random.split(rng)

            pred_1, _ = self.network.apply(
                train_params["trans"], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng}
            )
            pred_2, _ = self.network.apply(
                train_params["trans"], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng}
            )

            if self.config.use_weighted_sum:
                pred_1 = pred_1["weighted_sum"]
                pred_2 = pred_2["weighted_sum"]
            else:
                pred_1 = pred_1["value"]
                pred_2 = pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            loss = cross_ent_loss(logits, label_target)
            cse_loss = loss

            loss_collection["trans"] = loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key]) for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            cse_loss=aux_values["cse_loss"],
            loss=aux_values["loss"],
        )

        return new_train_states, metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_semi_pref_step(self, train_states, rng, labeled_batch, unlabeled_batch, lmd, tau):
        def compute_logits(train_params, batch, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            timestep_1 = batch["timestep"]
            timestep_2 = batch["timestep_2"]
            labels = batch["labels"]

            B, T, _ = act_1.shape[-3:]

            rng, _ = jax.random.split(rng)

            pred_1, _ = self.network.apply(
                train_params["trans"], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng}
            )
            pred_2, _ = self.network.apply(
                train_params["trans"], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng}
            )

            if self.config.use_weighted_sum:
                pred_1 = pred_1["weighted_sum"]
                pred_2 = pred_2["weighted_sum"]
            else:
                pred_1 = pred_1["value"]
                pred_2 = pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            rng, _ = jax.random.split(rng)
            logits, labels = compute_logits(train_params, labeled_batch, rng)
            u_logits, _ = compute_logits(train_params, unlabeled_batch, rng)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            loss = cross_ent_loss(logits, label_target)

            u_confidence = jnp.max(jax.nn.softmax(u_logits, axis=-1), axis=-1)
            pseudo_labels = jnp.argmax(u_logits, axis=-1)
            pseudo_label_target = jax.lax.stop_gradient(pseudo_labels)

            loss_ = optax.softmax_cross_entropy(
                logits=u_logits, labels=jax.nn.one_hot(pseudo_label_target, num_classes=2)
            )
            u_loss = jnp.sum(jnp.where(u_confidence > tau, loss_, 0)) / (jnp.count_nonzero(u_confidence > tau) + 1e-4)
            u_ratio = jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100

            # labeling neutral cases.
            binarized_idx = jnp.where(unlabeled_batch["labels"][:, 0] != 0.5, 1.0, 0.0)
            real_label = jnp.argmax(unlabeled_batch["labels"], axis=-1)
            u_acc = (
                jnp.sum(jnp.where(pseudo_label_target == real_label, 1.0, 0.0) * binarized_idx)
                / jnp.sum(binarized_idx)
                * 100
            )

            loss_collection["trans"] = last_loss = loss + lmd * u_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(
            train_params, lmd, tau, rng
        )

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key]) for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            loss=aux_values["loss"],
            u_loss=aux_values["u_loss"],
            last_loss=aux_values["last_loss"],
            u_ratio=aux_values["u_ratio"],
            u_train_acc=aux_values["u_acc"],
        )

        return new_train_states, metrics
