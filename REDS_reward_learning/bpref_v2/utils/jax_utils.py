from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class JaxRNG(object):
    """A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
    pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def wrap_function_with_rng(rng):
    """To be used as decorator, automatically bookkeep a RNG for the wrapped function."""

    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)

        return wrapped

    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def cross_ent_loss(logits, target):
    if len(target.shape) == 1:
        label = jax.nn.one_hot(target, num_classes=2)
    else:
        label = target

    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=label))
    return loss


def kld_loss(p, q):
    return jnp.mean(jnp.sum(jnp.where(p != 0, p * (jnp.log(p) - jnp.log(q)), 0), axis=-1))


def custom_softmax(array, axis=-1, temperature=1.0):
    array = array / temperature
    return jax.nn.softmax(array, axis=axis)


def pref_accuracy(logits, target):
    predicted_class = jnp.argmax(logits, axis=1)
    target_class = jnp.argmax(target, axis=1)
    return jnp.mean(predicted_class == target_class)


def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]

        return wrapped

    grad_fns = tuple(jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux) for i in range(n_outputs))

    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = jax.lax.pmean(grad_fn(*args, **kwargs), axis_name="pmap")
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)

    return multi_grad_fn


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


def cos_sim(x1, x2):
    normed_x1 = x1 / jnp.linalg.norm(x1, axis=-1, keepdims=True)
    normed_x2 = x2 / jnp.linalg.norm(x2, axis=-1, keepdims=True)
    return normed_x1 @ normed_x2.T


def l2_norm(x1, x2):
    return jnp.linalg.norm(x1 - x2)


def supervised_contrastive_loss(features, labels=None, temperature=0.1):
    # features shape: (batch_size * 2, feature_dim)
    # labels: (batch_size * 2)

    batch_size, n_views = features.shape[:2]
    labels = labels.reshape(-1, 1)
    mask = jnp.equal(labels, jnp.transpose(labels)).astype(jnp.float32)

    # score: (batch_size * 2, batch_size * 2)
    logits = cos_sim(features, features) / temperature
    logits_max = jnp.max(logits, axis=1, keepdims=True)
    logits = logits - jax.lax.stop_gradient(logits_max)

    logits_mask = jnp.ones_like(mask) - jnp.eye(batch_size)
    mask = mask * logits_mask

    exp_logits = jnp.exp(logits) * logits_mask
    log_prob = logits - jnp.log(exp_logits.sum(axis=1, keepdims=True))

    mask_pos_pairs = mask.sum(axis=1)
    mask_pos_pairs = jnp.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(axis=1) / mask_pos_pairs

    loss = -mean_log_prob_pos
    loss = loss.reshape(1, batch_size).mean()
    return loss


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)),
        0,
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


class TrainState(train_state.TrainState):
    batch_stats: Any = None


@partial(jax.pmap, axis_name="pmap", donate_argnums=0)
def sync_state_fn(state):
    i = jax.lax.axis_index("pmap")

    def select(x):
        return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

    return jax.tree_util.tree_map(select, state)
