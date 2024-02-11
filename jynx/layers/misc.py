import typing as tp
from functools import partial

from jax import Array, lax
from jax import numpy as jnp
from jax import random as rnd
from jax.typing import ArrayLike

from ..pytree import PyTree, static
from .module import Key, Module
from .static import Static


class Pooling(Static):
    init_value: ArrayLike
    op: tp.Callable[[Array, Array], Array]
    window_dimensions: tp.Sequence[int]
    window_strides: tp.Sequence[int] = ()
    padding: tp.Union[tp.Sequence[tp.Tuple[int, int]], str] = "VALID"
    normalize: bool = False

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        assert len(x.shape) >= len(
            self.window_dimensions
        ), "input must have at least as many dimensions as pooling window"
        window = (1,) * (len(x.shape) - len(self.window_dimensions)) + tuple(
            self.window_dimensions
        )
        strides = self.window_strides or window
        strides = (1,) * (len(x.shape) - len(strides)) + tuple(strides)
        pad = self.padding
        if not isinstance(pad, str):
            pad = ((0, 0),) * (len(x.shape) - len(pad)) + tuple(pad)

        y = lax.reduce_window(x, self.init_value, self.op, window, strides, pad)

        if self.normalize:
            y /= lax.reduce_window(
                jnp.ones_like(x), self.init_value, self.op, window, strides, pad
            )

        return y


MaxPooling = partial(Pooling, -jnp.inf, lax.max)
MinPooling = partial(Pooling, +jnp.inf, lax.min)
AvgPoolng = partial(Pooling, 0, lax.add, normalize=True)


class SkipConnection[**T](PyTree):
    residual: Module[T, Array]
    shortcut: Module[T, Array]

    def __call__(self, *x: T.args, **kwargs: T.kwargs) -> Array:
        if "key" in kwargs:
            k1, k2 = rnd.split(tp.cast(Array, kwargs.pop("key")))
            kwds1, kwds2 = {**kwargs, "key": k1}, {**kwargs, "key": k2}
        else:
            kwds1 = kwds2 = kwargs
        return self.residual(*x, **kwds1) + self.shortcut(*x, **kwds2)


class Norm(PyTree):
    weight: tp.Optional[Array] = None
    bias: tp.Optional[Array] = None
    axis: int = static(default=-1)
    subtract_mean: bool = static(default=True)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        if self.subtract_mean:
            x = x - jnp.mean(x, axis=self.axis, keepdims=True)

        x = x / jnp.square(x).sum(axis=self.axis, keepdims=True)

        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def norm(
    shape: tuple,
    axis: int,
    *,
    use_weight: bool = True,
    use_bias: bool = True,
    subtract_mean: bool = True,
    key: Key = None,  # for compatibility with other factories
) -> Norm:
    del key
    return Norm(
        jnp.ones(shape) if use_weight else None,
        jnp.zeros(shape) if use_bias else None,
        axis=axis,
        subtract_mean=subtract_mean,
    )


layer_norm = partial(norm, axis=-1)
