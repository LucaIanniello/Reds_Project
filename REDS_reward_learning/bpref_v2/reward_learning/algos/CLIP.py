from functools import partial

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from bpref_v2.third_party.openai.model import (
    load_clip_model,
    load_clip_model_with_adapter,
    load_liv_model,
    normalize_image,
)
from bpref_v2.utils.jax_utils import cos_sim

from .core import RewardLearner


def text_score(video_features, text_features, logit=1.0):
    return cos_sim(text_features, video_features)


class CLIPLearner(RewardLearner):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        # transfer type
        config.transfer_type = "liv"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, **kwargs):
        self.config = config
        self.network = self._define_network()
        self._init_train_state()

    def _define_network(self):
        if self.config.transfer_type == "liv":
            pvr_model, self.pvr_model_var, _ = load_liv_model()
        elif self.config.transfer_type.startswith("clip"):
            clip_type = self.config.transfer_type.split("_", 1)[-1]
            if clip_type == "vit_b16":
                clip_type = "vit_b16_original"
            if "adapter" in self.config.transfer_type:
                pvr_model, self.pvr_model_var, _ = load_clip_model_with_adapter(clip_type)
            else:
                pvr_model, self.pvr_model_var, _ = load_clip_model(clip_type)

        return pvr_model
        # bounded_model = pvr_model.bind(self.pvr_model_var)
        # return bounded_model

    def _init_train_state(self):
        self._train_states = {}

    def _get_ptr_feature(self, tokens):
        original_shape = tokens.shape[:-1]
        tokens = jnp.reshape(tokens, (-1, tokens.shape[-1]))
        text_feat = self.network.apply(self.pvr_model_var, tokens, method=self.network.encode_text, normalize=False)
        text_feat = jnp.reshape(text_feat, original_shape + (-1,))
        return text_feat

    def _extract_text_feature(self, tokens):
        text_feature = jax.lax.stop_gradient(self._get_ptr_feature(tokens))
        return text_feature

    @partial(jax.jit, static_argnames=("self"))
    def _get_reward_step(self, _, batch):
        images = jnp.array(list(batch["image"].values()))
        tokens = batch["instruct"]

        if images.ndim == 6:  # image: (num_images, batch_size, num_timestep, H, W, C)
            images = images[:, :, -1]
        images = jnp.reshape(images, (-1,) + images.shape[-3:])
        images = jax.image.resize(
            images, (images.shape[0], 224, 224, images.shape[-1]), method="bicubic"
        )  # to meet the input size of the clip model
        images = (images / 255.0).astype(jnp.float32)
        images = normalize_image(images)
        image_features = self.network.apply(self.pvr_model_var, images, method=self.network.encode_image)

        tokens = jnp.array(list(batch["instruct"].values()))
        task_embeddings = self._extract_text_feature(tokens).mean(axis=-2)

        cont_matrices = text_score(image_features, task_embeddings)
        diag_cont_matrices = jnp.diagonal(cont_matrices, axis1=-2, axis2=-1)
        target_text_indices = jnp.argmax(diag_cont_matrices, axis=0, keepdims=False)
        task_embedding = task_embeddings[target_text_indices, 0, :]

        rewards = jnp.diag(text_score(image_features, task_embedding))
        return rewards

    def get_visual_text_feature(self, batch):
        return self._get_visual_text_feature(self._train_states, batch)

    @partial(jax.jit, static_argnames=("self"))
    def _get_visual_text_feature(self, train_states, batch):
        obs = batch["image"]
        tokens = batch["instruct"]

        res = {}
        image_res = {}
        for key in obs.keys():
            image = obs[key]
            if image.ndim == 5:
                batch_size, seq_length = image.shape[:2]
            image = jnp.reshape(image, (-1,) + image.shape[-3:])
            image = jax.image.resize(
                image, (image.shape[0], 224, 224, image.shape[-1]), method="bicubic"
            )  # to meet the input size of the clip model
            image = (image / 255.0).astype(jnp.float32)
            image = normalize_image(image)
            image_feat = self.network.apply(
                self.pvr_model_var, image, method=self.network.encode_image, normalize=False
            )
            image_feat = image_feat.reshape(batch_size, seq_length, -1)
            image_res[key] = image_feat

        if tokens.ndim == 3:
            batch_size, seq_length = tokens.shape[:2]
            tokens = jnp.reshape(tokens, (-1, tokens.shape[-1]))
            text_feat = self.network.apply(self.pvr_model_var, tokens, method=self.network.encode_text, normalize=False)
            text_feat = text_feat.reshape(batch_size, seq_length, -1)
        else:
            text_feat = self.network.apply(self.pvr_model_var, tokens, method=self.network.encode_text, normalize=False)

        res["image"] = image_res
        res["instruct"] = text_feat

        return res

    @partial(jax.jit, static_argnames=("self"))
    def _eval_pref_step(self, train_states, rng, batch):
        return None

    @partial(jax.jit, static_argnames=("self"))
    def _train_pref_step(self, train_states, rng, batch):
        return None

    @partial(jax.jit, static_argnames=("self"))
    def _train_semi_pref_step(self, train_states, rng, labeled_batch, unlabeled_batch, lmd, tau):
        return None
