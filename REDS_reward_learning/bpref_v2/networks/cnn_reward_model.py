import copy
from typing import Any, Sequence, Callable, Tuple
from functools import partial

import flax.linen as nn
import jax.numpy as jnp

from . import flaxmodel_ops
from .trans_reward_model import GPT2Model
from bpref_v2.utils.jax_utils import get_1d_sincos_pos_embed

ModuleDef = Any


class D4PGEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        # batch_size = observations.shape[0]
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.moveaxis(x, -4, -1)
        # x = jnp.reshape(x, (-1, *x.shape[-3:]))
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=flaxmodel_ops.default_init(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        # return x.reshape((batch_size, -1))
        return x.reshape((*x.shape[:-3], -1))


class ResNetV2Block(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.norm()(x)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides)(residual)

        return residual + y


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetV2Encoder(nn.Module):
    """ResNetV2."""

    stage_sizes: Tuple[int]
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)

        x = x.astype(jnp.float32) / 255.0
        x = jnp.moveaxis(x, -4, -1)
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        if x.shape[-2] == 224:
            x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        else:
            x = conv(self.num_filters, (3, 3))(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetV2Block(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)

        x = norm()(x)
        x = self.act(x)
        return x.reshape((*x.shape[:-3], -1))


class CNNRewardModel(nn.Module):
    config: Any = None

    def setup(self):
        self.encoder = D4PGEncoder(
            features=self.config.features, filters=self.config.filters, strides=self.config.strides
        )
        self.reward_predictor = flaxmodel_ops.MLP(
            [self.config.embd_dim, 1], activations=nn.sigmoid, activate_final=False
        )

    def predict_reward(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        N, B, T, H, W, C = observations.shape

        def concat_multiple_emb(emb):
            # input of img_emb: (N, B, T, H, W, C)
            emb = jnp.reshape(emb, (N * B, T, H, W, C))
            # output of img_emb: (N * B, T, H, W, C)
            emb = jnp.concatenate(jnp.split(emb, N, axis=0), 1)  # (B, N * T, H, W, C)
            return emb

        feat = self.encoder(concat_multiple_emb(observations))
        return self.reward_predictor(feat, training=training)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.predict_reward(observations)


class PVRRewardModel(nn.Module):
    config: Any = None

    def setup(self):
        self.reward_predictor = flaxmodel_ops.MLP(
            [self.config.embd_dim, 1], activations=nn.sigmoid, activate_final=False
        )

    def predict_reward(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        N, B = observations.shape[:2]
        observations = jnp.concatenate(jnp.split(observations, N, axis=0), axis=-1)
        feat = jnp.reshape(observations, (B, -1))
        return self.reward_predictor(feat, training=training)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.predict_reward(observations)


class MultiStagePVRRewardModel(nn.Module):
    config: Any = None

    def setup(self):
        VmapNet = nn.vmap(
            flaxmodel_ops.MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.config.n_stages,
        )
        self.nets = VmapNet([self.config.embd_dim, 1], activations=nn.sigmoid, activate_final=False)
        self.trained = [False] * self.config.n_stages

    def set_trained(self, stage_idx: int):
        self.trained[stage_idx] = True

    def predict_reward(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        N, B = observations.shape[:2]
        observations = jnp.concatenate(jnp.split(observations, N, axis=0), axis=-1)
        feat = jnp.reshape(observations, (B, -1))
        return self.nets(feat)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.predict_reward(observations)


class MultiStageCNNRewardModel(nn.Module):
    config: Any = None

    def setup(self):
        VmapNet = nn.vmap(
            flaxmodel_ops.MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.config.n_stages,
        )
        self.nets = VmapNet([self.config.embd_dim, 1], activations=nn.sigmoid, activate_final=False)
        self.encoder = D4PGEncoder(
            features=self.config.features, filters=self.config.filters, strides=self.config.strides
        )
        # self.encoder = ResNetV2Encoder((2, 2, 2, 2))

    def set_trained(self, stage_idx: int):
        self.trained[stage_idx] = True

    def predict_reward(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        N, B, T, H, W, C = observations.shape

        def concat_multiple_emb(emb):
            # input of img_emb: (N, B, T, H, W, C)
            emb = jnp.reshape(emb, (N * B, T, H, W, C))
            # output of img_emb: (N * B, T, H, W, C)
            emb = jnp.concatenate(jnp.split(emb, N, axis=0), 1)  # (B, N * T, H, W, C)
            return emb

        feat = self.encoder(concat_multiple_emb(observations))
        return self.nets(feat, training=training)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.predict_reward(observations)


class RPFCNNRewardModel(nn.Module):
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

        decoder_config = copy.deepcopy(self.config)
        decoder_config.emb_dim = decoder_config.n_embd = decoder_config.embd_dim = self.embd_dim
        decoder_config.use_bidirectional = False

        self.encoder = D4PGEncoder(
            features=self.config.features, filters=self.config.filters, strides=self.config.strides
        )
        # self.encoder = ResNetV2Encoder((2, 2, 2, 2))
        self.text_adapter = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim * 2, self.embd_dim])
        self.text_residual_weight = self.param(
            "text_residual_weight",
            nn.initializers.constant(4.0, dtype=jnp.float32),
            (1,),
        )
        self.image_input = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim])
        self.reward_predictor = flaxmodel_ops.MLP(hidden_dims=[self.embd_dim, 1])
        self.temporal_decoder = GPT2Model(config=decoder_config)

    def encode_image(self, images, training=False):
        N, B, T, H, W, C = images.shape

        feat = self.encoder(images.reshape(-1, 1, H, W, C))
        feat = jnp.reshape(feat, (N, B, T, -1))
        return feat

    def encode_text(self, text_features, training=False):
        res = nn.sigmoid(self.text_residual_weight)
        text_features = res * text_features + (1 - res) * self.text_adapter(text_features, training=training)
        return text_features

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
        E: embed_dim
        """

        N, B, T = image_features.shape[:3]

        def concat_multiple_emb(emb):
            # input of img_emb: (batch_size * num_image * seq_length, emb_dim)
            emb = jnp.reshape(emb, (N * B, T, -1))
            # output of img_emb: (batch_size, seq_length, num_image * emb_dim)
            emb = jnp.concatenate(jnp.split(emb, N, axis=0), -1)  # (B, T, N * E)
            return emb

        image_features = concat_multiple_emb(image_features)  # (B, T, N * E)
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

        return image_feature_emb

    def predict_reward(self, video_features, text_feature, training=False):
        batch_size, seq_length = video_features.shape[:2]
        vid_t = video_features.reshape(batch_size, -1)
        reward_input = jnp.concatenate([vid_t, text_feature], axis=-1)
        return self.reward_predictor(reward_input, training=training)

    def __call__(self, video, phases, attn_mask=None, training=False):
        num_image, batch_size, seq_length = video.shape[:3]
        image_feature = self.encode_image(video, training=training)
        _ = self.encode_text(text_features=phases, training=training)
        video_feature = self.encode_video(image_feature, None, attn_mask=attn_mask, training=training)
        reward = self.predict_reward(video_feature, phases, training=training)
        return reward
