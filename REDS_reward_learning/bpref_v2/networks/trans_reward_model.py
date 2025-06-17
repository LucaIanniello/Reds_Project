import copy
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from bpref_v2.utils.jax_utils import (
    cos_sim,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
)

from bpref_v2.third_party.openai.model import load_clip_model

from . import flaxmodel_ops


class GPT2SelfAttention(nn.Module):
    """
    GPT2 Self Attention.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """

    config: dict = None

    def setup(self):
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd
        self.num_heads = self.config.n_head
        self.head_dim = self.embd_dim // self.num_heads
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.scale_attn_weights = True

    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        """
        Run attention.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        x = nn.Dense(features=3 * self.embd_dim)(x)

        query, key, value = jnp.split(x, 3, axis=2)

        query = flaxmodel_ops.split_heads(query, self.num_heads, self.head_dim)
        value = flaxmodel_ops.split_heads(value, self.num_heads, self.head_dim)
        key = flaxmodel_ops.split_heads(key, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jnp.concatenate((past_key, key), axis=-2)
            value = jnp.concatenate((past_value, value), axis=-2)

        present = (key, value) if use_cache else None

        query_len, key_len = query.shape[-2], key.shape[-2]
        causal_mask = jnp.tril(jnp.ones((1, 1, self.max_pos, self.max_pos)))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        # causal_mask = jnp.ones((1, 1, self.max_pos, self.max_pos))[:, :, key_len - query_len :key_len, :key_len]
        causal_mask = causal_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=self.attn_dropout)
        out, _attn_weights = flaxmodel_ops.attention(
            query, key, value, causal_mask, -1e4, attn_dropout, self.scale_attn_weights, training, attn_mask, head_mask
        )
        out = flaxmodel_ops.merge_heads(out, self.num_heads, self.head_dim)

        out = nn.Dense(features=self.embd_dim)(out)

        out = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
        return out, present, _attn_weights


class BidirectionalSelfAttention(GPT2SelfAttention):
    """
    GPT2 Self Attention.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """

    config: dict = None

    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        """
        Run attention.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        x = nn.Dense(features=3 * self.embd_dim)(x)

        query, key, value = jnp.split(x, 3, axis=2)

        query = flaxmodel_ops.split_heads(query, self.num_heads, self.head_dim)
        value = flaxmodel_ops.split_heads(value, self.num_heads, self.head_dim)
        key = flaxmodel_ops.split_heads(key, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jnp.concatenate((past_key, key), axis=-2)
            value = jnp.concatenate((past_value, value), axis=-2)

        present = (key, value) if use_cache else None

        query_len, key_len = query.shape[-2], key.shape[-2]
        non_causal_mask = jnp.ones((1, 1, self.max_pos, self.max_pos))[:, :, key_len - query_len : key_len, :key_len]
        # causal_mask = jnp.ones((1, 1, self.max_pos, self.max_pos))[:, :, key_len - query_len :key_len, :key_len]
        non_causal_mask = non_causal_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=self.attn_dropout)
        out, _attn_weights = flaxmodel_ops.attention(
            query,
            key,
            value,
            non_causal_mask,
            -1e4,
            attn_dropout,
            self.scale_attn_weights,
            training,
            attn_mask,
            head_mask,
        )
        out = flaxmodel_ops.merge_heads(out, self.num_heads, self.head_dim)

        out = nn.Dense(features=self.embd_dim)(out)

        out = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
        return out, present, _attn_weights


class GPT2MLP(nn.Module):
    """
    GPT2 MLP.

    Attributes:
        intermediate_dim (int): Dimension of the intermediate layer.
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """

    intermediate_dim: int
    config: dict = None

    def setup(self):
        self.embd_dim = self.config.n_embd
        self.resid_dropout = self.config.resid_pdrop
        self.activation = self.config.activation_function

    @nn.compact
    def __call__(self, x, training=False):
        """
        Run the MLP.

        Args:
            x (tensor): Input tensor.
            training (bool): Training mode.
        """
        x = nn.Dense(features=self.intermediate_dim)(x)
        x = flaxmodel_ops.apply_activation(x, activation=self.activation)
        x = nn.Dense(features=self.embd_dim)(x)
        x = nn.Dropout(rate=self.resid_dropout)(x, deterministic=not training)
        return x


class GPT2Block(nn.Module):
    """
    GPT2 Block.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """

    config: dict = None

    def setup(self):
        self.embd_dim = self.config.n_embd
        self.eps = self.config.layer_norm_epsilon
        self.inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * self.embd_dim

    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        """
        Run the block.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        kwargs = {
            "layer_past": layer_past,
            "attn_mask": attn_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,
            "training": training,
        }
        if not hasattr(self.config, "use_bidirectional") or not self.config.use_bidirectional:
            x, present, _attn_weights = GPT2SelfAttention(config=self.config)(x, **kwargs)
        else:
            x, present, _attn_weights = BidirectionalSelfAttention(config=self.config)(x, **kwargs)
        x += residual
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = GPT2MLP(intermediate_dim=self.inner_dim, config=self.config)(x, training)
        x += residual
        return x, present, _attn_weights


class GPT2Model(nn.Module):
    """
    The GPT2 Model.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        pretrained (str): Which pretrained model to use, None for random initialization.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded.
        \\ If None, a temp directory will be used.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """

    config: dict = None
    pretrained: str = None
    ckpt_dir: str = None

    def setup(self):
        self.vocab_size = self.config.vocab_size
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd
        self.embd_dropout = self.config.embd_pdrop
        self.num_layers = self.config.n_layer
        self.eps = self.config.layer_norm_epsilon

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        input_embds=None,
        position_ids=None,
        attn_mask=None,
        head_mask=None,
        use_cache=False,
        training=False,
    ):
        """
        Run the model.

        Args:
            input_ids (tensor): Input token ids, shape [B, seq_len].
            past_key_values (Tuple): Precomputed hidden keys and values, tuple of tuples.
                                     If past_key_values is used, only input_ids that do not have their
                                     past calculated should be passed as input_ids.
            input_embds (tensor): Input embeddings, shape [B, seq_len, embd_dim].
            labels (tensor): Labels for language modeling, shape [B, seq_len]. Will be shifted inside the model. Ignore label = -100.
            position_ids (tensor): Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
            attn_mask (tensor): Mask to avoid performing attention on padding token indices, shape [B, seq_len].
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.

        Returns:
            (dict): Dictionary containing 'last_hidden_state', 'past_key_values'.
        """
        if input_ids is not None and input_embds is not None:
            raise ValueError("You cannot specify both input_ids and input_embd at the same time.")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = jnp.reshape(input_ids, shape=(-1, input_shape[-1]))
            batch_size = input_ids.shape[0]
        elif input_embds is not None:
            input_shape = input_embds.shape[:-1]
            batch_size = input_embds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or input_embd.")

        if position_ids is not None:
            position_ids = jnp.reshape(position_ids, shape=(-1, input_shape[-1]))

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.num_layers)
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = jnp.arange(start=past_length, stop=input_shape[-1] + past_length)
            position_ids = jnp.reshape(jnp.expand_dims(position_ids, axis=0), shape=(-1, input_shape[-1]))

        if input_embds is None:
            input_embds = nn.Embed(num_embeddings=self.vocab_size, features=self.embd_dim)(input_ids)

        if attn_mask is not None:
            attn_mask = flaxmodel_ops.get_attention_mask(attn_mask, batch_size)

        if head_mask is not None:
            head_mask = flaxmodel_ops.get_head_mask(head_mask, self.num_layers)
        else:
            head_mask = [None] * self.num_layers

        # position_embds = nn.Embed(num_embeddings=self.max_pos, features=self.embd_dim)(position_ids)

        # x = input_embds + position_embds
        x = input_embds

        x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training)
        input_shape + (x.shape[-1],)

        presents = () if use_cache else None
        attn_weights_list = []
        for i in range(self.num_layers):
            kwargs = {
                "layer_past": past_key_values[i],
                "attn_mask": attn_mask,
                "head_mask": head_mask[i],
                "use_cache": use_cache,
                "training": training,
            }
            x, present, attn_weights = GPT2Block(config=self.config)(x, **kwargs)

            if use_cache:
                presents = presents + (present,)
            attn_weights_list.append(attn_weights)

        x = nn.LayerNorm(epsilon=self.eps)(x)
        return {"last_hidden_state": x, "past_key_values": presents, "attn_weights_list": attn_weights_list}


class TransRewardModel(nn.Module):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None
    max_episode_steps: int = 1000

    def setup(self):
        self.config.activation_function = self.activation
        self.config.activation_final = self.activation_final
        self.vocab_size = self.config.vocab_size
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd
        self.pref_attn_embd_dim = self.config.pref_attn_embd_dim
        self.embd_dropout = self.config.embd_pdrop
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.num_layers = self.config.n_layer
        self.inner_dim = self.config.n_embd // 2
        self.eps = self.config.layer_norm_epsilon

    @nn.compact
    def __call__(
        self,
        states,
        actions,
        timesteps,
        attn_mask=None,
        training=False,
        reverse=False,
        target_idx=1,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        embd_state = nn.Dense(features=self.embd_dim)(states)
        embd_action = nn.Dense(features=self.embd_dim)(actions)
        embd_timestep = nn.Embed(num_embeddings=self.max_episode_steps + 1, features=self.embd_dim)(timesteps)

        embd_state = embd_state + embd_timestep
        embd_action = embd_action + embd_timestep

        if reverse:
            stacked_inputs = (
                jnp.stack([embd_state, embd_action], axis=-3)
                .swapaxes(-2, -3)
                # .transpose(0, 2, 1, 3)
                .reshape(batch_size, 2 * seq_length, self.embd_dim)
            )
        else:
            stacked_inputs = (
                jnp.stack([embd_action, embd_state], axis=-3)
                .swapaxes(-2, -3)
                # .transpose(0, 2, 1, 3)
                .reshape(batch_size, 2 * seq_length, self.embd_dim)
            )

        stacked_inputs = nn.LayerNorm(epsilon=self.eps)(stacked_inputs)

        stacked_attn_mask = (
            jnp.stack([attn_mask, attn_mask], axis=1).transpose(0, 2, 1).reshape(batch_size, 2 * seq_length)
        )

        transformer_outputs = GPT2Model(config=self.config)(
            input_embds=stacked_inputs,
            attn_mask=stacked_attn_mask,
            training=training,
        )

        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).transpose(0, 2, 1, 3)
        hidden_output = x[:, target_idx]

        if self.config.use_weighted_sum:
            """
            add additional Attention Layer for Weighted Sum.
            x (= output, tensor): Predicted Reward, shape [B, seq_len, embd_dim]
            """
            x = nn.Dense(features=2 * self.pref_attn_embd_dim + 1)(hidden_output)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1

            # query: [B, seq_len, embd_dim]
            # key: [B, seq_len, embd_dim]
            # value: [B, seq_len, 1]

            query, key, value = jnp.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], axis=2)
            query = flaxmodel_ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = flaxmodel_ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = flaxmodel_ops.split_heads(value, num_heads, 1)

            # query: [B, 1, seq_len, embd_dim]
            # key: [B, 1, seq_len, embd_dim]
            # value: [B, 1, seq_len, 1]

            query_len, key_len = query.shape[-2], key.shape[-2]
            # causal_mask = jnp.tril(jnp.ones((1, 1, self.config.n_positions, self.config.n_positions)))[:, :, key_len - query_len :key_len, :key_len]
            # causal_mask = causal_mask.astype(bool)
            causal_mask = jnp.ones((1, 1, seq_length, seq_length))[:, :, key_len - query_len : key_len, :key_len]
            causal_mask = causal_mask.astype(bool)

            # attn_dropout = nn.Dropout(rate=self.attn_dropout) # split dropout rate
            attn_dropout = nn.Dropout(rate=0.0)  # boilerplate code.
            new_attn_mask = flaxmodel_ops.get_attention_mask(attn_mask, batch_size)

            out, last_attn_weights = flaxmodel_ops.attention(
                query,
                key,
                value,
                causal_mask,
                -1e-4,
                attn_dropout,
                scale_attn_weights=True,
                training=training,
                attn_mask=new_attn_mask,
                head_mask=None,
            )
            attn_weights_list.append(last_attn_weights)
            # out: [B, 1, seq_len, 1]
            output = flaxmodel_ops.merge_heads(out, num_heads, 1)
            # output: [B, seq_len, 1]

            # output = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
            return {"weighted_sum": output, "value": value}, attn_weights_list

        else:
            x = nn.Dense(features=self.inner_dim)(hidden_output)
            x = flaxmodel_ops.apply_activation(x, activation=self.activation)
            output = nn.Dense(features=1)(x)
            if self.activation_final != "none":
                output = flaxmodel_ops.apply_activation(output, activation=self.activation_final)

            return {"value": output}, attn_weights_list


class VisualTransRewardModel(TransRewardModel):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None
    max_episode_steps: int = 1000
    vision_model: nn.Module = None

    def setup(self):
        self.config.activation_function = self.activation
        self.config.activation_final = self.activation_final
        self.vocab_size = self.config.vocab_size
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd
        self.pref_attn_embd_dim = self.config.pref_attn_embd_dim
        self.embd_dropout = self.config.embd_pdrop
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.num_layers = self.config.n_layer
        self.inner_dim = self.config.n_embd // 2
        self.eps = self.config.layer_norm_epsilon

        from ..third_party.openai.model import normalize_image

        self.normalize_image = normalize_image

    @nn.compact
    def __call__(
        self,
        images,
        actions,
        timesteps,
        attn_mask=None,
        training=False,
    ):
        batch_size, seq_length = images.shape[0], images.shape[1]

        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        images = jnp.reshape(
            images, (-1,) + images.shape[-3:]
        )  # (batch_size * num_image * num_timestep, image.shape[-3:])
        images = jax.image.resize(
            images, (images.shape[0], 224, 224, images.shape[-1]), method="bicubic"
        )  # to meet the input size of the clip model
        images = self.normalize_image(images)

        image_features = self.vision_model(images)[0]
        image_features = image_features.reshape(batch_size, seq_length, -1)
        if getattr(self.config, "frozen_visual", False):
            image_features = jax.lax.stop_gradient(image_features)

        embd_images = nn.Dense(features=self.embd_dim, name="embd_images")(image_features)
        embd_action = nn.Dense(features=self.embd_dim, name="embd_action")(actions)
        embd_timestep = nn.Embed(
            num_embeddings=self.max_episode_steps + 1, features=self.embd_dim, name="embd_timestep"
        )(timesteps)

        embd_images = embd_images + embd_timestep
        embd_action = embd_action + embd_timestep

        stacked_inputs = (
            jnp.stack([embd_images, embd_action], axis=-3)
            .swapaxes(-2, -3)
            .reshape(batch_size, 2 * seq_length, self.embd_dim)
        )

        stacked_inputs = nn.LayerNorm(epsilon=self.eps)(stacked_inputs)

        stacked_attn_mask = (
            jnp.stack([attn_mask, attn_mask], axis=1).transpose(0, 2, 1).reshape(batch_size, 2 * seq_length)
        )

        transformer_outputs = GPT2Model(config=self.config)(
            input_embds=stacked_inputs,
            attn_mask=stacked_attn_mask,
            training=training,
        )

        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).transpose(0, 2, 1, 3)
        hidden_output = x[:, -1]

        assert self.config.use_weighted_sum is False, "Please do not use_weighted_sum in VisualTransRewardModel"
        if self.config.use_weighted_sum:
            """
            add additional Attention Layer for Weighted Sum.
            x (= output, tensor): Predicted Reward, shape [B, seq_len, embd_dim]
            """
            x = nn.Dense(features=2 * self.pref_attn_embd_dim + 1)(hidden_output)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1

            # query: [B, seq_len, embd_dim]
            # key: [B, seq_len, embd_dim]
            # value: [B, seq_len, 1]

            query, key, value = jnp.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], axis=2)
            query = flaxmodel_ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = flaxmodel_ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = flaxmodel_ops.split_heads(value, num_heads, 1)

            # query: [B, 1, seq_len, embd_dim]
            # key: [B, 1, seq_len, embd_dim]
            # value: [B, 1, seq_len, 1]

            query_len, key_len = query.shape[-2], key.shape[-2]
            # causal_mask = jnp.tril(jnp.ones((1, 1, self.config.n_positions, self.config.n_positions)))[:, :, key_len - query_len :key_len, :key_len]
            # causal_mask = causal_mask.astype(bool)
            causal_mask = jnp.ones((1, 1, seq_length, seq_length))[:, :, key_len - query_len : key_len, :key_len]
            causal_mask = causal_mask.astype(bool)

            # attn_dropout = nn.Dropout(rate=self.attn_dropout) # split dropout rate
            attn_dropout = nn.Dropout(rate=0.0)  # boilerplate code.
            new_attn_mask = flaxmodel_ops.get_attention_mask(attn_mask, batch_size)

            out, last_attn_weights = flaxmodel_ops.attention(
                query,
                key,
                value,
                causal_mask,
                -1e-4,
                attn_dropout,
                scale_attn_weights=True,
                training=training,
                attn_mask=new_attn_mask,
                head_mask=None,
            )
            attn_weights_list.append(last_attn_weights)
            # out: [B, 1, seq_len, 1]
            output = flaxmodel_ops.merge_heads(out, num_heads, 1)
            # output: [B, seq_len, 1]

            # output = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
            return {"weighted_sum": output, "value": value}, attn_weights_list

        else:
            x = nn.Dense(features=self.inner_dim, name="out")(hidden_output)
            x = flaxmodel_ops.apply_activation(x, activation=self.activation)
            output = nn.Dense(features=1)(x)
            if self.activation_final != "none":
                output = flaxmodel_ops.apply_activation(output, activation=self.activation_final)

            return {"value": output}, attn_weights_list


class VisualOnlyTransRewardModel(VisualTransRewardModel):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None
    max_episode_steps: int = 1000
    vision_model: nn.Module = None

    @nn.compact
    def __call__(
        self,
        images,
        actions,
        timesteps,
        attn_mask=None,
        training=False,
    ):
        batch_size, seq_length = images.shape[0], images.shape[1]

        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        images = jnp.reshape(
            images, (-1,) + images.shape[-3:]
        )  # (batch_size * num_image * num_timestep, image.shape[-3:])
        images = jax.image.resize(
            images, (images.shape[0], 224, 224, images.shape[-1]), method="bicubic"
        )  # to meet the input size of the clip model
        images = self.normalize_image(images)

        image_features = self.vision_model(images)[0]
        image_features = image_features.reshape(batch_size, seq_length, -1)
        if getattr(self.config, "frozen_visual", False):
            image_features = jax.lax.stop_gradient(image_features)

        embd_images = nn.Dense(features=self.embd_dim, name="embd_images")(image_features)
        embd_timestep = nn.Embed(
            num_embeddings=self.max_episode_steps + 1, features=self.embd_dim, name="embd_timestep"
        )(timesteps)

        embd_images = embd_images + embd_timestep

        stacked_inputs = (
            jnp.stack([embd_images], axis=-3)
            .swapaxes(-2, -3)
            # .transpose(0, 2, 1, 3)
            .reshape(batch_size, seq_length, self.embd_dim)
        )

        stacked_inputs = nn.LayerNorm(epsilon=self.eps)(stacked_inputs)

        stacked_attn_mask = jnp.stack([attn_mask], axis=1).transpose(0, 2, 1).reshape(batch_size, seq_length)

        transformer_outputs = GPT2Model(config=self.config)(
            input_embds=stacked_inputs,
            attn_mask=stacked_attn_mask,
            training=training,
        )

        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 1, self.embd_dim).transpose(0, 2, 1, 3)
        hidden_output = x[:, -1]

        assert self.config.use_weighted_sum is False, "Please do not use_weighted_sum in VisualTransRewardModel"
        x = nn.Dense(features=self.inner_dim, name="out")(hidden_output)
        x = flaxmodel_ops.apply_activation(x, activation=self.activation)
        output = nn.Dense(features=1)(x)
        if self.activation_final != "none":
            output = flaxmodel_ops.apply_activation(output, activation=self.activation_final)

        return {"value": output}, attn_weights_list


class ARPV1RewardModel(nn.Module):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None
    max_episode_steps: int = 1000
    # vision_model: nn.Module = None
    # text_model: nn.Module = None
    clip_model: nn.Module = None
    logit_scale_val: float = None

    def setup(self):
        self.config.activation_function = self.activation
        self.config.activation_final = self.activation_final
        self.vocab_size = self.config.vocab_size
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd = self.config.embd_dim
        self.embd_dropout = self.config.embd_pdrop
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.num_layers = self.config.n_layer
        self.inner_dim = self.config.n_embd // 2
        self.eps = self.config.layer_norm_epsilon
        self.logit_scale = self.config.logit_scale

        from ..third_party.openai.model import normalize_image

        # self.clip_model, self.clip_vars, self.logit_scale = load_clip_model("vit_b32")
        
        self.normalize_image = normalize_image

        self.image_adapter = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim * 2, self.clip_model.visual.out_features])
        self.image_residual_weight = self.param(
            "image_residual_weight",
            nn.initializers.constant(4.0, dtype=jnp.float32),
            (1,),
        )

        self.text_adapter = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim * 2, self.clip_model.text.out_features])
        self.text_residual_weight = self.param(
            "text_residual_weight",
            nn.initializers.constant(4.0, dtype=jnp.float32),
            (1,),
        )

        self.action_predictor = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim * 2, self.action_dim])

    def get_clip_visual_feature(self, images, normalize=False):
        images = (images / 255.0).astype(jnp.float32)
        images = self.normalize_image(images)
        image_features = self.clip_model.visual(images)[0]
        if normalize:
            image_features /= jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        return image_features

    def get_clip_text_feature(self, tokens, normalize=False):
        text_features = self.clip_model.text(tokens)
        if normalize:
            text_features /= jnp.linalg.norm(text_features, axis=-1, keepdims=True)
        return text_features

    # def encode_image(
    #     self,
    #     images,
    #     image_features=None,
    #     training=False,
    # ):
    #     if image_features is None:
    #         image_features = self.get_clip_visual_feature(images)
    #     image_features = jax.lax.stop_gradient(image_features)

    #     res = nn.sigmoid(self.image_residual_weight)
    #     image_features = res * image_features + (1 - res) * self.image_adapter(image_features, training=training)
    #     image_features /= jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    #     return image_features

    def encode_text(self, tokens, training=False):
        text_features = self.get_clip_text_feature(tokens)
        text_features = jax.lax.stop_gradient(text_features)

        res = nn.sigmoid(self.text_residual_weight)
        text_features = res * text_features + (1 - res) * self.text_adapter(text_features, training=training)
        text_features /= jnp.linalg.norm(text_features, axis=-1, keepdims=True)
        return text_features

    def predict_action(self, before_image_features, image_features, text_features, training=False):
        concat_feature = jnp.concatenate([before_image_features, image_features, text_features], axis=-1)
        a_hat = self.action_predictor(concat_feature, training=training)
        return a_hat

    def video_score(self, video_features, text_features, training=False):
        return self.logit_scale * cos_sim(video_features, text_features)

    def text_score(self, video_features, text_features, training=False):
        return self.logit_scale * cos_sim(text_features, video_features)

    def __call__(self, images, tokens, training=False):
        image_feature = self.encode_image(images, training=training)
        text_feature = self.encode_text(tokens, training=training)
        concat_feature = jnp.concatenate([image_feature, image_feature, text_feature], axis=-1)
        a_hat = self.action_predictor(concat_feature, training=training)
        return a_hat


class RPFRewardModel(ARPV1RewardModel):
    config: Any = None
    pretrained: str = ""
    ckpt_dir: str = ""
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = ""
    activation_final: str = ""

    def setup(self):
        self.config.activation_function = self.activation
        self.config.activation_final = self.activation_final
        self.vocab_size = self.config.vocab_size
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd = self.config.embd_dim
        self.output_embd_dim = self.config.output_embd_dim
        self.vision_embd_dim = self.config.vision_embd_dim
        self.num_images = self.config.num_images
        self.embd_dropout = self.config.embd_pdrop
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.num_layers = self.config.n_layer
        self.inner_dim = self.config.n_embd // 2
        self.eps = self.config.layer_norm_epsilon

        from ..third_party.openai.model import normalize_image

        self.normalize_image = normalize_image

        decoder_config = copy.deepcopy(self.config)
        decoder_config.emb_dim = decoder_config.n_embd = decoder_config.embd_dim = self.embd_dim
        decoder_config.use_bidirectional = False

        self.text_adapter = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim * 2, self.embd_dim])
        self.text_residual_weight = self.param(
            "text_residual_weight",
            nn.initializers.constant(4.0, dtype=jnp.float32),
            (1,),
        )
        self.image_input = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim])
        self.temporal_decoder = GPT2Model(config=decoder_config)
        self.reward_predictor = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim, 1])

    def encode_text(self, text_features, training=False):
        res = nn.sigmoid(self.text_residual_weight)
        text_features = res * text_features + (1 - res) * self.text_adapter(text_features, training=training)
        return text_features

    def construct_cam_embed(self, x):
        cams = [vp for vp in self.config.image_keys.split("|")]
        _pos_embed = []
        embed_dim = x.shape[-1]
        for cam in cams:
            cam_token = self.param(
                f"{cam}_token",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, embed_dim),
            )
            _pos_embed.append(cam_token)

        img_pos_embed = jnp.concatenate(_pos_embed, axis=-1)  # (1, E * N)
        return img_pos_embed
        # if x.ndim == 5:  # x.shape: (N, B, T, P, E)
        #     pos_embed = jnp.tile(img_pos_embed, (1, x.shape[-2]))  # (1, E * N * P)
        # else:
        #     pos_embed = img_pos_embed
        # return pos_embed

    @nn.nowrap
    def no_decay_list(self):
        # model specific no decay list
        no_decay = [f"{cam}_token" for cam in self.config.image_keys.split("|")]
        return no_decay

    @nn.compact
    def encode_video(self, image_features, timesteps, attn_mask=None, out_features=False, training=False):
        """
        N: num_images
        B: batch_size
        T: sequence length
        P: number of patches = 1 (CLS) + patch_size ** 2
        E: embed_dim
        """

        N, B, T, P = image_features.shape[:4]
        cls_token, image_features = (
            image_features[..., :1, :],
            image_features[..., 1:, :],
        )
        image_features = image_features + get_2d_sincos_pos_embed(image_features.shape[-1], P - 1)
        image_features = jnp.concatenate([cls_token, image_features], axis=-2)
        cam_embed = self.construct_cam_embed(image_features)

        def concat_multiple_emb(emb):
            # input of img_emb: (N, B, T, P, E)
            emb = jnp.reshape(emb, (N * B, T, P, -1))
            # output of img_emb: (batch_size, seq_length, num_image * emb_dim)
            emb = jnp.concatenate(jnp.split(emb, N, axis=0), -1)  # (B, T, P, N * E)
            return emb

        image_features = concat_multiple_emb(image_features)  # (B, T, P, N * E)
        image_features = image_features + cam_embed
        image_features = jnp.reshape(image_features, (B, T, -1))
        image_feature_emb = self.image_input(image_features)  # (B, T, E)

        embd_timestep = get_1d_sincos_pos_embed(embed_dim=image_feature_emb.shape[-1], length=T)
        stacked_inputs = image_feature_emb + embd_timestep

        if attn_mask is None:
            attn_mask = jnp.ones((B, T), dtype=jnp.float32)

        decoded_outputs = self.temporal_decoder(
            input_embds=stacked_inputs,
            attn_mask=attn_mask,
            training=training,
        )
        video_features = decoded_outputs["last_hidden_state"]
        return video_features

    def predict_reward(self, video_features, text_feature, training=False):
        batch_size, seq_length = video_features.shape[:2]
        vid_t = video_features.reshape(batch_size, -1)
        reward_input = jnp.concatenate([vid_t, text_feature], axis=-1)
        return self.reward_predictor(reward_input, training=training)

        # vid_t = self.epic_video_proj(video_features[:, -1])
        # phase_embed = self.task_embed(phase)
        # return jnp.diag(cos_sim(vid_t, phase_embed))

    def project_nfp_video_feature(self, video_features, training=False):
        return self.nfp_video_proj(video_features[:, -1], training=training)

    def project_cont_video_feature(self, flattened_video_features, training=False):
        return self.cont_video_proj(flattened_video_features, training=training)

    def project_epic_video_feature(self, video_features, training=False):
        return self.epic_video_proj(video_features[:, -1], training=training)

    def project_liv_video_feature(self, video_features, training=False):
        return self.liv_video_proj(video_features[:, -1], training=training)

    def __call__(self, video, phases, attn_mask=None, training=False):
        num_image, batch_size, seq_length = video.shape[:3]
        # image_feature = video.reshape(-1, video.shape[-1]), training=training).reshape(
        #     num_image, batch_size, seq_length, -1
        # )
        _ = self.encode_text(text_features=phases, training=training)
        video_feature = self.encode_video(video, None, attn_mask=attn_mask, training=training)
        # _ = self.project_cont_video_feature(video_feature[:, -1], training=training)
        reward = self.predict_reward(video_feature, phases, training=training)
        # _ = self.project_nfp_video_feature(video_feature, training=training)
        return reward
        # epic_video_out_feature = self.project_epic_video_feature(video_feature, training=training)
        # liv_video_out_feature = self.project_liv_video_feature(video_feature, training=training)
        # video_out_feature = (cont_video_out_feature + epic_video_out_feature + liv_video_out_feature) / 3
        # return video_out_feature


class RPFNoTransRewardModel(RPFRewardModel):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None

    def setup(self):
        self.config.activation_function = self.activation
        self.config.activation_final = self.activation_final
        self.vocab_size = self.config.vocab_size
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd = self.config.embd_dim
        self.output_embd_dim = self.config.output_embd_dim
        self.vision_embd_dim = self.config.vision_embd_dim
        self.num_images = self.config.num_images
        self.embd_dropout = self.config.embd_pdrop
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.num_layers = self.config.n_layer
        self.inner_dim = self.config.n_embd // 2
        self.eps = self.config.layer_norm_epsilon

        from ..third_party.openai.model import normalize_image

        self.normalize_image = normalize_image

        self.text_adapter = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim * 2, self.embd_dim])
        self.text_residual_weight = self.param(
            "text_residual_weight",
            nn.initializers.constant(4.0, dtype=jnp.float32),
            (1,),
        )
        self.image_input = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim])
        self.reward_predictor = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim, 1])

    @nn.compact
    def encode_video(self, image_features, timesteps, attn_mask=None, out_features=False, training=False):
        """
        N: num_images
        B: batch_size
        T: sequence length
        P: number of patches = 1 (CLS) + patch_size ** 2
        E: embed_dim
        """

        N, B, T, P = image_features.shape[:4]

        def concat_multiple_emb(emb):
            # input of img_emb: (batch_size * num_image * seq_length, emb_dim)
            emb = jnp.reshape(emb, (N * B, T, -1))
            # output of img_emb: (batch_size, seq_length, num_image * emb_dim)
            emb = jnp.concatenate(jnp.split(emb, N, axis=0), -1)  # (B, T, N * P * E)
            return emb

        image_features = concat_multiple_emb(image_features)  # (B, T, N * P * E)
        image_feature_emb = self.image_input(image_features)  # (B, T, E)
        return image_feature_emb


class RPFNoSupConRewardModel(RPFRewardModel):
    config: Any = None
    pretrained: str = None
    ckpt_dir: str = None
    observation_dim: int = 29
    action_dim: int = 8
    activation: str = None
    activation_final: str = None

    def predict_reward(self, video_features, training=False):
        batch_size, seq_length = video_features.shape[:2]
        vid_t = video_features.reshape(batch_size, -1)
        reward_input = vid_t
        return self.reward_predictor(reward_input, training=training)

    def __call__(self, video, attn_mask=None, training=False):
        num_image, batch_size, seq_length = video.shape[:3]
        video_feature = self.encode_video(video, None, attn_mask=attn_mask, training=training)
        reward = self.predict_reward(video_feature, training=training)
        return reward
