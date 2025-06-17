from functools import partial

import augmax
import jax
import jax.numpy as jnp


def random_crop(img, key, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnums=(2,))
def batched_random_crop(imgs, key, padding=12):
    keys = jax.random.split(key, imgs.shape[0] + 1)
    split_rngs, next_rng = keys[:-1], keys[-1]
    return jax.vmap(random_crop, (0, 0, None))(imgs, split_rngs, padding), next_rng


def resize_and_jitter(image_size=224):
    _transforms = [augmax.Resize(image_size, image_size), augmax.ByteToFloat()]
    # _transforms.append(augmax.RandomCrop(width=int(0.8 * image_size), height=int(0.8 * image_size)))
    _transforms.append(augmax.ColorJitter(brightness=0.3, contrast=0.5, p=1.0))
    # _transforms.append(augmax.Rotate(angle_range=(-30, 30), p=1.0))

    def single_image_aug_fn(image, rng):
        transform = augmax.Chain(*_transforms)
        return jnp.clip(255.0 * transform(rng, image), 0, 255).astype(jnp.uint8)

    return single_image_aug_fn


@partial(jax.jit, static_argnums=(2,))
def batched_resize_and_jitter(imgs, key, image_size=224):
    keys = jax.random.split(key, imgs.shape[0] + 1)
    split_rngs, next_rng = keys[:-1], keys[-1]
    return jax.vmap(resize_and_jitter(image_size), (0, 0))(imgs, split_rngs), next_rng


def single_image_aug_fn(image, key, image_size=224, padding=12):
    cropped_image = random_crop(image, key, padding=padding)
    return resize_and_jitter(image_size=image_size)(cropped_image, key)


def tube_pmap_image_aug_fn(image_size=224, padding=10, window_size=4, jax_devices=None, image_key=None):
    aug_fn = partial(single_image_aug_fn, image_size=image_size, padding=padding)

    @partial(jax.pmap, axis_name="pmap", devices=jax_devices)
    def _pmap_image_aug_fn(images, rng, image_size=image_size, padding=padding):
        num_rngs = images.shape[0] // window_size
        sub_rngs = jax.random.split(rng, num_rngs + 1)
        sub_rngs, new_rng = sub_rngs[:-1], sub_rngs[-1]
        sub_rngs = jnp.repeat(sub_rngs, window_size, axis=0)
        return jax.jit(jax.vmap(aug_fn))(images, sub_rngs), new_rng

    return _pmap_image_aug_fn


def single_pmap_image_aug_fn(image_size=224, padding=10, window_size=4, jax_devices=None, image_key=None):
    aug_fn = partial(single_image_aug_fn, image_size=image_size, padding=padding)

    @partial(jax.pmap, axis_name="pmap", devices=jax_devices)
    def _pmap_image_aug_fn(images, rng, image_size=image_size, padding=padding):
        num_rngs = images.shape[0]
        sub_rngs = jax.random.split(rng, num_rngs + 1)
        sub_rngs, new_rng = sub_rngs[:-1], sub_rngs[-1]
        return jax.jit(jax.vmap(aug_fn))(images, sub_rngs), new_rng

    return _pmap_image_aug_fn
