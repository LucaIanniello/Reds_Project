from collections import defaultdict
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from bpref_v2.networks.mlp_reward_model import MLPRewardModel


class EnsembleMLPRewardModel(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = "256-256-256"
    orthogonal_init: bool = False
    activations: str = "relu"
    activation_final: str = "none"
    num_ensembles: int = 5

    @nn.compact
    def __call__(self, observations, actions, aggregate=True):
        VmapMLPRewardModel = nn.vmap(
            MLPRewardModel,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_ensembles,
        )
        rewards = VmapMLPRewardModel(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            arch=self.arch,
            orthogonal_init=self.orthogonal_init,
            activations=self.activations,
            activation_final=self.activation_final,
        )(observations, actions)

        if aggregate:
            return jnp.stack(rewards).mean(axis=0)
        else:
            return jnp.stack(rewards)


class EnsembleTransRewardModel(nn.Module):
    func_def: Any = None
    num_ensembles: int = 5

    def setup(self):
        self.reward_model = [self.func_def()] * self.num_ensembles

    @nn.compact
    def __call__(self, observations, actions, timesteps, aggregate=True, **kwargs):
        preds = defaultdict(list)
        for en_idx in range(self.num_ensembles):
            output, _ = self.reward_model[en_idx](observations, actions, timesteps, **kwargs)
            for key, val in output.items():
                preds[key].append(val)

        if aggregate:
            return {key: jnp.stack(val).mean(axis=0) for key, val in preds.items()}, None
        else:
            return {key: jnp.stack(val) for key, val in preds.items()}, None
