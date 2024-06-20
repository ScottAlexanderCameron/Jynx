from collections.abc import Callable, Iterator, Sequence

from jax import Array
from jax import numpy as jnp
from jax import random as rnd
from jax.typing import ArrayLike

from ..pytree import PyTree


class Static(PyTree):
    """A base class for static elements within the neural network,
    not subject to training updates.

    This class serves as a foundation for defining static operations or
    components in a neural network model, which do not change during
    training. It inherits from `PyTree` to integrate seamlessly with
    JAX's functional transformations and optimizations.

    Methods:
        tree_flatten(self):
            A method to satisfy the PyTree protocol with no dynamic attributes.
        tree_flatten_with_keys(self):
            An extension to `tree_flatten`, including keys for attributes.
        tree_unflatten(cls, aux, children):
            Reconstructs the instance from the flattened state. In this
            case, it simply returns the auxiliary data.

    """

    def tree_flatten(self):
        return (), self

    def tree_flatten_with_keys(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux, children):
        del children
        return aux


class Fn[T, U](Static):
    """Represents a static function within the neural network,
    encapsulating a callable as a PyTree node.

    This class allows wrapping pure functions to be used within JAX
    computational graphs, ensuring they are treated as static components.

    Attributes:
        fn (Callable[[T], U]): The function to be wrapped and treated
            as a static component.

    Methods:
        __call__(self, x, *args, **kwargs) -> U:
            Invokes the wrapped function with the provided arguments.

    """

    fn: Callable[[T], U]

    def __call__(self, x: T, *args, **kwargs) -> U:
        del args, kwargs
        return self.fn(x)


class StarFn[T, U](Static):
    """A variant of `Fn` that applies the wrapped function to a sequence
    of arguments, unpacking them.

    This class is useful for functions that take multiple arguments,
    allowing them to be passed as a sequence and then unpacked when the
    function is called.

    Attributes:
        fn (Callable[..., U]): The function to be wrapped, accepting a
            variable number of arguments.

    Methods:
        __call__(self, x, *args, **kwargs) -> U:
            Invokes the wrapped function, unpacking the sequence `x` into separate arguments.

    """

    fn: Callable[..., U]

    def __call__(self, x: Sequence[T], *args, **kwargs) -> U:
        del args, kwargs
        return self.fn(*x)


class Reshape(Static):
    """A static operation that reshapes its input array to a specified shape.

    Attributes:
        shape (tuple): The target shape to which the input array will be reshaped.

    Methods:
        __call__(self, x, *args, **kwargs) -> Array:
            Reshapes the input array `x` to the target shape.

    """

    shape: tuple

    def __call__(self, x: ArrayLike, *args, **kwargs) -> Array:
        del args, kwargs
        return jnp.reshape(x, self.shape)


class Dropout(Static):
    """A static dropout layer, randomly setting a fraction of input
    units to 0 at each update during training.

    Attributes:
        dropout_prob (float): The probability of setting each input unit to 0.

    Methods:
        __call__(self, x, *args, key=None, **kwargs) -> Array:
            Applies dropout to the input array `x` using the provided `key` for randomness.
            If `key` is not provided, then dropout is disabled.

    """

    dropout_prob: float

    def __call__(
        self,
        x: Array,
        *args,
        rng: Iterator[Array] | None = None,
        **kwargs,
    ) -> Array:
        del args, kwargs
        if rng is None or self.dropout_prob == 0:
            return x
        else:
            key = next(rng)
            mask = rnd.bernoulli(key, self.dropout_prob, x.shape)
            return mask.astype(x.dtype) * x
