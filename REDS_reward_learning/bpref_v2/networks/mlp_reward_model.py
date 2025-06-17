from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp

from ..utils.jax_utils import extend_and_repeat


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = "256-256-256"
    orthogonal_init: bool = False
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activation_final: Callable[[jnp.ndarray], jnp.ndarray] = None

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h, kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)), bias_init=jax.nn.initializers.zeros
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = self.activations(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim, kernel_init=jax.nn.initializers.orthogonal(1e-2), bias_init=jax.nn.initializers.zeros
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(1e-2, "fan_in", "uniform"),
                bias_init=jax.nn.initializers.zeros,
            )(x)

        if self.activation_final is not None:
            output = self.activation_final(output)
        return output


class MLPRewardModel(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = "256-256-256"
    orthogonal_init: bool = False
    activations: str = "relu"
    activation_final: str = "none"

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        activations = {
            "relu": nn.relu,
            "leaky_relu": nn.leaky_relu,
        }[self.activations]
        activation_final = {
            "none": None,
            "tanh": nn.tanh,
        }[self.activation_final]

        x = FullyConnectedNetwork(
            output_dim=1,
            arch=self.arch,
            orthogonal_init=self.orthogonal_init,
            activations=activations,
            activation_final=activation_final,
        )(x)
        return jnp.squeeze(x, -1)


class VisualMLPewardModel(nn.Module):
    observation_dim: int
    action_dim: int
    embd_dim: int = 256
    arch: str = "256-256-256"
    orthogonal_init: bool = False
    activations: str = "relu"
    activation_final: str = "none"
    transfer_type: str = "liv"
    vision_model: nn.Module = None
    frozen_visual: bool = False

    def setup(self):
        if self.transfer_type == "liv":
            from ..third_party.openai.model import normalize_image

            self.normalize_image = normalize_image

    @nn.compact
    @multiple_action_q_function
    def __call__(self, images, actions):
        images = jnp.reshape(
            images, (-1,) + images.shape[-3:]
        )  # (batch_size * num_image * num_timestep, image.shape[-3:])
        images = jax.image.resize(
            images, (images.shape[0], 224, 224, images.shape[-1]), method="bicubic"
        )  # to meet the input size of the clip model
        images = self.normalize_image(images)

        image_features = self.vision_model(images)[0]
        if getattr(self, "frozen_visual", False):
            image_features = jax.lax.stop_gradient(image_features)

        embd_images = nn.Dense(features=self.embd_dim, name="embd_images")(image_features)
        embd_action = nn.Dense(features=self.embd_dim, name="embd_action")(actions)

        x = jnp.concatenate([embd_images, embd_action], axis=-1)

        activations = {
            "relu": nn.relu,
            "leaky_relu": nn.leaky_relu,
        }[self.activations]
        activation_final = {
            "none": None,
            "tanh": nn.tanh,
        }[self.activation_final]

        x = FullyConnectedNetwork(
            output_dim=1,
            arch=self.arch,
            orthogonal_init=self.orthogonal_init,
            activations=activations,
            activation_final=activation_final,
        )(x)
        return jnp.squeeze(x, -1)


tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class StochasticFullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = "256-256-256"
    orthogonal_init: bool = False
    activations: str = "relu"
    activation_final: str = "none"
    temperature: float = 0.1
    state_dependent_std: bool = True
    log_std_scale: float = 1e-3
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h, kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)), bias_init=jax.nn.initializers.zeros
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = self.activations(x)

        if self.orthogonal_init:
            means = nn.Dense(
                self.output_dim, kernel_init=jax.nn.initializers.orthogonal(1e-2), bias_init=jax.nn.initializers.zeros
            )(x)
        else:
            means = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(1e-2, "fan_in", "uniform"),
                bias_init=jax.nn.initializers.zeros,
            )(x)

        if self.state_dependent_std:
            log_stds = nn.Dense(1, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))(self.log_std_scale))(means)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (1,))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * self.temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return base_dist
