from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from bpref_v2.networks.mlp_reward_model import MLPRewardModel
from bpref_v2.utils.jax_utils import cross_ent_loss, next_rng, value_and_multi_grad

from .core import RewardLearner


class MRLearner(RewardLearner):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 3e-4
        config.optimizer_type = "adam"
        config.scheduler_type = "none"

        config.reward_arch = "256-256"
        config.orthogonal_init = False
        config.activation = "relu"
        config.activation_final = "none"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, observation_dim, action_dim):
        self.config = self.get_default_config(config)
        self.network = self._define_network(observation_dim, action_dim)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "adamw": optax.adamw,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        self.tx = optimizer_class(learning_rate=self.config.lr)

        self._total_steps = 0
        self._init_train_state()

    def _define_network(self, observation_dim, action_dim):
        return MLPRewardModel(
            observation_dim=observation_dim,
            action_dim=action_dim,
            arch=self.config.reward_arch,
            orthogonal_init=self.config.orthogonal_init,
            activations=self.config.activation,
            activation_final=self.config.activation_final,
        )

    def _init_train_state(self):
        params = self.network.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states["mlp"] = TrainState.create(
            params=params,
            tx=self.tx,
            apply_fn=None,
        )
        model_keys = ["mlp"]
        self._model_keys = tuple(model_keys)

    @partial(jax.jit, static_argnames=("self"))
    def _get_reward_step(self, train_states, batch):
        obs = batch["observations"]
        act = batch["actions"]
        train_params = {key: train_states[key].params for key in self.model_keys}
        pred = self.network.apply(train_params["mlp"], obs, act)
        return pred

    @partial(jax.jit, static_argnames=("self"))
    def _eval_pref_step(self, train_states, rng, batch):
        def loss_fn(train_params, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            labels = batch["labels"]

            B, T, obs_dim = batch["observations"].shape
            B, T, act_dim = batch["actions"].shape

            obs_1 = obs_1.reshape(-1, obs_dim)
            obs_2 = obs_2.reshape(-1, obs_dim)
            act_1 = act_1.reshape(-1, act_dim)
            act_2 = act_2.reshape(-1, act_dim)

            pred_1 = self.network.apply(train_params["mlp"], obs_1, act_1)
            pred_2 = self.network.apply(train_params["mlp"], obs_2, act_2)

            sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            loss = cross_ent_loss(logits, label_target)

            loss_collection["mlp"] = loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

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
            labels = batch["labels"]

            B, T, obs_dim = batch["observations"].shape
            B, T, act_dim = batch["actions"].shape

            obs_1 = obs_1.reshape(-1, obs_dim)
            obs_2 = obs_2.reshape(-1, obs_dim)
            act_1 = act_1.reshape(-1, act_dim)
            act_2 = act_2.reshape(-1, act_dim)

            pred_1 = self.network.apply(train_params["mlp"], obs_1, act_1)
            pred_2 = self.network.apply(train_params["mlp"], obs_2, act_2)

            sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            loss = cross_ent_loss(logits, label_target)

            loss_collection["mlp"] = loss
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

    def train_semi(self, labeled_batch, unlabeled_batch, lmd, tau):
        self._total_steps += 1
        self._train_states, metrics = self._train_semi_pref_step(
            self._train_states, labeled_batch, unlabeled_batch, lmd, tau, next_rng()
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_semi_pref_step(self, train_states, rng, labeled_batch, unlabeled_batch, lmd, tau):
        def compute_logits(batch):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            labels = batch["labels"]

            B, T, obs_dim = batch["observations"].shape
            B, T, act_dim = batch["actions"].shape

            obs_1 = obs_1.reshape(-1, obs_dim)
            obs_2 = obs_2.reshape(-1, obs_dim)
            act_1 = act_1.reshape(-1, act_dim)
            act_2 = act_2.reshape(-1, act_dim)

            pred_1 = self.network.apply(train_params["mlp"], obs_1, act_1)
            pred_2 = self.network.apply(train_params["mlp"], obs_2, act_2)

            sum_pred_1 = jnp.mean(pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            logits, labels = compute_logits(labeled_batch)
            u_logits, _ = compute_logits(unlabeled_batch)

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
            u_loss = jnp.where(u_confidence > tau, loss_, 0).mean()
            u_ratio = jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100

            loss_collection["mlp"] = loss + lmd * u_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(
            train_params, lmd, tau, rng
        )

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key]) for i, key in enumerate(self.model_keys)
        }

        metrics = dict(loss=aux_values["loss"], u_loss=aux_values["u_loss"], u_ratio=aux_values["u_ratio"])

        return new_train_states, metrics
