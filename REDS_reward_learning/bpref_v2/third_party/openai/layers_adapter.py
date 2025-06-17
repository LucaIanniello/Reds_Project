import functools
from typing import Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from .layers import MLP, LayerNorm, ModifiedResNet, quick_gelu


class Adapter(nn.Module):
    d_prj: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        D = x.shape[-1]
        x = nn.Dense(features=self.d_prj, kernel_init=nn.initializers.normal(stddev=2e-2), name="down")(x)
        x = quick_gelu(x)
        x = nn.Dense(features=D, kernel_init=nn.initializers.zeros, name="up")(x)
        return x


class ResidualAttentionBlockwithAdapter(nn.Module):
    """Self-attention block of Transformer.

    Attributes:
      num_heads: Number of heads.
    """

    num_heads: int
    d_prj: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask=None) -> jnp.ndarray:
        xn = LayerNorm(name="ln_1")(x)
        x = x + Adapter(d_prj=self.d_prj, name="adapter1")(
            nn.SelfAttention(self.num_heads, name="attn", deterministic=True)(xn, attn_mask)
        )
        xn = LayerNorm(name="ln_2")(x)
        x = x + Adapter(d_prj=self.d_prj, name="adapter2")(MLP(name="mlp")(xn))
        return x


class TransformerwithAdapter(nn.Module):
    """Transformer module.

    Attributes:
      features: Number of features.
      num_layers: Number of layers for each block.
      num_heads: Number of heads.
    """

    features: int
    num_layers: int
    num_heads: int
    d_prj: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        for i in range(self.num_layers):
            x = ResidualAttentionBlockwithAdapter(num_heads=self.num_heads, d_prj=self.d_prj, name=f"resblocks.{i}")(
                x, attn_mask
            )
            self.sow("intermediates", "intermediate_layer_%d" % i, x)
        return x


class VisionTransformerwithAdapter(nn.Module):
    """Vision Transformer.

    Attributes:
      patch_size: The size of the patches to embed.
      features: Number of features.
      num_layers: Number of transformer blocks (self-attn + MLP).
      num_heads: Number of attention heads.
      out_features: Number of output features. If None, return transformer output.
    """

    patch_size: int
    features: int
    num_layers: int
    num_heads: int
    out_features: Optional[int]
    d_prj: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        x = nn.Conv(
            self.features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=False,
            name="conv1",
        )(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        scale = 1.0 / jnp.sqrt(self.features)
        class_embedding = self.param(
            "class_embedding",
            jax.nn.initializers.normal(stddev=scale),
            (self.features,),
        )
        x = jnp.concatenate((jnp.tile(class_embedding[None, None, :], (x.shape[0], 1, 1)), x), axis=1)
        if (
            self.variables["params"].get("positional_embedding") is not None
            and x.shape[1] != self.variables["params"]["positional_embedding"].shape[0]
        ):
            positional_embedding = self.variables["params"]["positional_embedding"]
            positional_embedding = positional_embedding[: x.shape[1]]
        else:
            positional_embedding = self.param(
                "positional_embedding",
                jax.nn.initializers.normal(stddev=scale),
                (x.shape[1], self.features),
            )
        x = x + positional_embedding[None]

        x = LayerNorm(name="ln_pre")(x)
        x = feature_map = TransformerwithAdapter(
            features=self.features,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_prj=self.d_prj,
            name="transformer",
        )(x)

        if self.out_features is not None:
            x = LayerNorm(name="ln_post")(x[:, 0])
            x = nn.Dense(self.out_features, use_bias=False, name="proj")(x)
        else:
            x = LayerNorm(name="ln_post")(x)

        return x, feature_map


class TextEncoderwithAdapter(nn.Module):
    """Text Transformer.

    Attributes:
      vocab_size: Size of the vocabulary.
      features: Number of features.
      num_layers: Number of transformer blocks (self-attn + MLP).
      num_heads: Number of attention heads.
      out_features: Size of the final text embedding.
    """

    vocab_size: int
    features: int
    num_layers: int
    num_heads: int
    out_features: int

    @nn.compact
    def __call__(self, text: jnp.ndarray) -> jnp.ndarray:
        positional_embedding = self.param(
            "positional_embedding",
            jax.nn.initializers.zeros,
            (text.shape[1], self.features),
        )
        mask = nn.combine_masks(nn.make_attention_mask(text > 0, text > 0), nn.make_causal_mask(text))
        x = nn.Embed(self.vocab_size, self.features, name="token_embedding")(text)
        x = x + positional_embedding[None]
        x = TransformerwithAdapter(self.features, self.num_layers, self.num_heads, name="transformer")(
            x, attn_mask=mask
        )
        x = LayerNorm(name="ln_final")(x)
        x = x[jnp.arange(x.shape[0]), text.argmax(-1)]
        x = nn.Dense(self.out_features, use_bias=False, name="text_projection")(x)
        return x


class CLIPwithAdapter(nn.Module):
    """Clip model consisting of a vision and text transformer.

    Attributes:
      vocab_size: Size of the vocabulary.
      embed_dim: Size of the text and vision embeddings.
      text_features: Number of features in text transformer.
      text_num_layers: Number of text transformer blocks (self-attn + MLP).
      text_num_heads: Number of heads in text transformer.
      vision_features: Number of features in vision transformer.
      vision_num_layers: Number of vision transformer blocks (self-attn + MLP).
      vision_patch_size: Size of patches to embed in vision transformer.
    """

    vocab_size: int
    embed_dim: int
    # Text.
    text_features: int
    text_num_layers: int
    text_num_heads: int
    # Vision.
    vision_features: int
    vision_num_layers: Union[int, Sequence[int]]
    vision_patch_size: Optional[int] = None
    vision_return_map: bool = False

    def setup(self):
        if isinstance(self.vision_num_layers, (tuple, list)):
            self.vision_num_heads = self.vision_features * 32 // 64
            self.visual = ModifiedResNet(
                num_layers=self.vision_num_layers,
                features=self.vision_features,
                num_heads=self.vision_num_heads,
                out_features=None if self.vision_return_map else self.embed_dim,
            )
        else:
            self.vision_num_heads = self.vision_features // 64
            self.visual = VisionTransformerwithAdapter(
                patch_size=self.vision_patch_size,
                features=self.vision_features,
                num_layers=self.vision_num_layers,
                num_heads=self.vision_num_heads,
                out_features=None if self.vision_return_map else self.embed_dim,
            )
        self.text = TextEncoderwithAdapter(
            out_features=self.embed_dim,
            vocab_size=self.vocab_size,
            features=self.text_features,
            num_layers=self.text_num_layers,
            num_heads=self.text_num_heads,
        )
        self.logit_scale = self.param("logit_scale", jax.nn.initializers.zeros, ())

    def get_logit_scale(self):
        return self.logit_scale

    def encode_image(self, image: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
        x = self.visual(image)[0]
        if normalize:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def encode_text(self, text: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
        x = self.text(text)
        if normalize:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def __call__(
        self, image: jnp.ndarray, text: jnp.ndarray, normalize: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = y = None
        if image is not None:
            x = self.encode_image(image, normalize)
        if text is not None:
            y = self.encode_text(text, normalize)
        return x, y
