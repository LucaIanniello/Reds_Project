from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from bpref_v2.networks.lstm_reward_model import LSTMRewardModel
from bpref_v2.utils.jax_utils import cross_ent_loss, next_rng, value_and_multi_grad

from .core import RewardLearner


class NMRLearner(RewardLearner):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 1e-3
        config.optimizer_type = "adam"
        config.scheduler_type = "none"
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_inner = config.embd_dim // 2
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1

        config.train_type = "sum"
        config.train_diff_bool = False

        config.explicit_sparse = False
        config.k = 5

        config.activation = "relu"
        config.activation_final = "none"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def _define_network(self, observation_dim, action_dim):
        return LSTMRewardModel(
            config=self.config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            activation=self.config.activation,
            activation_final=self.config.activation_final,
        )

    def _init_train_state(self):
        params = self.network.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 10, self.observation_dim)),
            jnp.zeros((10, 10, self.action_dim)),
            jnp.ones((10, 10), dtype=jnp.int32),
        )
        self._train_states["lstm"] = TrainState.create(params=params, tx=self.tx, apply_fn=None)

        model_keys = ["lstm"]
        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    @partial(jax.jit, static_argnames=("self"))
    def _get_reward_step(self, train_states, batch):
        obs = batch["observations"]
        act = batch["actions"]
        timestep = batch["timestep"]

        train_params = {key: train_states[key].params for key in self.model_keys}
        pred, _ = self.network.apply(train_params["lstm"], obs, act, timestep)
        return pred, None

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

            B, T, _ = batch["observations"].shape
            B, T, _ = batch["actions"].shape

            rng, _ = jax.random.split(rng)

            pred_1, _ = self.network.apply(
                train_params["lstm"], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng}
            )
            pred_2, _ = self.network.apply(
                train_params["lstm"], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng}
            )

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
            loss_collection["lstm"] = loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
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

            B, T, _ = batch["observations"].shape
            B, T, _ = batch["actions"].shape

            rng, _ = jax.random.split(rng)

            pred_1, _ = self.network.apply(
                train_params["lstm"], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng}
            )
            pred_2, _ = self.network.apply(
                train_params["lstm"], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng}
            )

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            if self.config.train_type == "sum":
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

            loss_collection["lstm"] = loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key]) for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            loss=aux_values["loss"],
        )

        return new_train_states, metrics
