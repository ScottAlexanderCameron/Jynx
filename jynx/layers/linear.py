import typing as tp

import jax.nn.initializers as init
from jax import Array, lax
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer
from jax.typing import ArrayLike

from ..pytree import PyTree, static


class Linear(PyTree):
    weight: Array
    bias: tp.Optional[Array]

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        return _maybe_add_bias(self.bias, x @ self.weight)


def linear(
    in_size: int,
    out_size: int,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> Linear:
    k1, k2 = rnd.split(key)
    return Linear(
        weight_init(k1, (in_size, out_size)),
        bias_init(k2, (out_size,)) if bias_init is not None else None,
    )


class Conv(PyTree):
    kernel: Array
    bias: tp.Optional[Array]
    strides: tp.Sequence[int] = static()
    padding: tp.Union[tp.Sequence[tp.Tuple[int, int]], str] = static(default="VALID")

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        extra_dims = x.shape[: -len(self.kernel.shape)]
        x = x.reshape((-1, *x.shape[len(extra_dims) + 1 :]))
        x = _maybe_add_bias(
            self.bias,
            lax.conv_general_dilated(
                x,
                self.kernel,
                self.strides,
                self.padding,
            ),
        )
        return x.reshape((*extra_dims, -1, *x.shape[1:]))


def conv(
    in_channels: int,
    out_channels: int,
    kernel_shape: tp.Sequence[int],
    strides: tp.Optional[tp.Sequence[int]] = None,
    padding: tp.Union[tp.Sequence[tp.Tuple[int, int]], str] = "VALID",
    *,
    kernel_init: Initializer = init.kaiming_normal(1, 0),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> Conv:
    k1, k2 = rnd.split(key)
    if not isinstance(padding, str):
        padding = tuple(padding)

    return Conv(
        kernel_init(k1, (out_channels, in_channels) + tuple(kernel_shape)),
        (
            bias_init(k2, (out_channels,) + (1,) * len(kernel_shape))
            if bias_init is not None
            else None
        ),
        tuple(strides or (1,) * len(kernel_shape)),
        padding,
    )


class ConvTranspose(PyTree):
    kernel: Array
    bias: tp.Optional[Array]
    strides: tp.Sequence[int] = static()
    padding: tp.Union[tp.Sequence[tp.Tuple[int, int]], str] = static(default="VALID")

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        extra_dims = x.shape[: -len(self.kernel.shape)]
        x = x.reshape((-1, *x.shape[len(extra_dims) + 1 :]))
        x = _maybe_add_bias(
            self.bias,
            lax.conv_transpose(
                x,
                self.kernel,
                self.strides,
                self.padding,
                dimension_numbers=lax.conv_dimension_numbers(
                    x.shape, self.kernel.shape, None
                ),
            ),
        )
        return x.reshape((*extra_dims, -1, *x.shape[1:]))


def conv_transpose(
    in_channels: int,
    out_channels: int,
    kernel_shape: tp.Sequence[int],
    strides: tp.Optional[tp.Sequence[int]] = None,
    padding: tp.Union[tp.Sequence[tp.Tuple[int, int]], str] = "VALID",
    *,
    kernel_init: Initializer = init.kaiming_normal(1, 0),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> ConvTranspose:
    k1, k2 = rnd.split(key)
    if not isinstance(padding, str):
        padding = tuple(padding)

    return ConvTranspose(
        kernel_init(k1, (out_channels, in_channels) + tuple(kernel_shape)),
        (
            bias_init(k2, (out_channels,) + (1,) * len(kernel_shape))
            if bias_init is not None
            else None
        ),
        tuple(strides or (1,) * len(kernel_shape)),
        padding,
    )


class Embedding(PyTree):
    weight: Array

    def __call__(self, x: ArrayLike, *args, **kwargs) -> Array:
        del args, kwargs
        return jnp.take(self.weight, x, axis=0)


def embedding(
    size: int,
    num_classes: int,
    *,
    weight_init: Initializer = init.orthogonal(),
    key: Array,
) -> Embedding:
    return Embedding(weight_init(key, (num_classes, size)))


def _maybe_add_bias(bias, x):
    if bias is not None:
        x = x + bias
    return x
