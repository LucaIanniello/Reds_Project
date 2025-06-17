import abc

import optax
from ml_collections import ConfigDict

from bpref_v2.utils.jax_utils import next_rng


class RewardLearner(metaclass=abc.ABCMeta):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 1e-3
        config.optimizer_type = "adam"
        config.scheduler_type = "none"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, observation_dim, action_dim, jax_devices, num_ensembles=1):
        self.config = config
        if num_ensembles == 1:
            self.network = self._define_network(observation_dim, action_dim)
        else:
            self.network = self._define_network(observation_dim, action_dim, num_ensembles)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.jax_devices = jax_devices
        self.n_devices = len(self.jax_devices)
        self.num_ensembles = num_ensembles
        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "adamw": optax.adamw,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            "CosineDecay": optax.warmup_cosine_decay_schedule(
                init_value=self.config.lr,
                peak_value=self.config.lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.lr,
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(value=self.config.lr),
                ],
                [self.config.warmup_steps],
            ),
            "none": None,
        }[self.config.scheduler_type]

        if scheduler_class:
            self.tx = optimizer_class(scheduler_class)
        else:
            self.tx = optimizer_class(learning_rate=self.config.lr)

        self._total_steps = 0
        self._init_train_state()

    def load(self, state):
        self._train_states = state

    @abc.abstractmethod
    def _define_network(self, observation_dim, action_dim):
        pass

    @abc.abstractmethod
    def _init_train_state(self):
        pass

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(self._train_states, next_rng(), batch)
        return metrics

    @abc.abstractmethod
    def _train_pref_step(self, train_states, rng, batch):
        pass

    def train_semi(self, labeled_batch, unlabeled_batch, lmd, tau):
        self._total_steps += 1
        self._train_states, metrics = self._train_semi_pref_step(
            self._train_states, labeled_batch, unlabeled_batch, lmd, tau, next_rng()
        )
        return metrics

    def _train_semi_pref_step(self, train_states, rng, labeled_batch, unlabeled_batch, lmb, tau):
        return None

    def evaluation(self, batch):
        metrics = self._eval_pref_step(self._train_states, next_rng(), batch)
        return metrics

    @abc.abstractmethod
    def _eval_pref_step(self, train_states, rng, batch):
        pass

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @abc.abstractmethod
    def _get_reward_step(self, train_states, batch):
        pass
