import typing as tp

from jax import Array
from jax import numpy as jnp
from jax import random as rnd
from jax.typing import ArrayLike

from ..pytree import PyTree
from .module import Key


class Static(PyTree):
    def tree_flatten(self):
        return (), self

    def tree_flatten_with_keys(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux, children):
        del children
        return aux


class Fn[T, U](Static):
    fn: tp.Callable[[T], U]

    def __call__(self, x: T, *args, **kwargs) -> U:
        del args, kwargs
        return self.fn(x)


class StarFn[T, U](Static):
    fn: tp.Callable[..., U]

    def __call__(self, x: tp.Sequence[T], *args, **kwargs) -> U:
        del args, kwargs
        return self.fn(*x)


class Reshape(Static):
    shape: tuple

    def __call__(self, x: ArrayLike, *args, **kwargs) -> Array:
        del args, kwargs
        return jnp.reshape(x, self.shape)


class Dropout(Static):
    dropout_prob: float

    def __call__(self, x: Array, *args, key: Key = None, **kwargs) -> Array:
        del args, kwargs
        if key is None or self.dropout_prob == 0:
            return x
        else:
            mask = rnd.bernoulli(key, self.dropout_prob, x.shape)
            return mask.astype(x.dtype) * x
