from collections.abc import Callable, Sequence
from functools import partial

from jax import Array, lax
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..pytree import PyTree, static
from .static import Static


class Pooling(Static):
    """A generic pooling layer that applies a specified pooling operation over inputs.

    This layer supports various pooling operations like max pooling,
    min pooling, and average pooling through specialization via partial
    application of the Pooling class constructor. The operation is
    defined by parameters such as the window dimensions, strides, and
    padding. The layer can optionally normalize the pooling output.

    Attributes:
        init_value (ArrayLike): The initial value for the pooling
            operation, determining the type of pooling.
        op (Callable[[Array, Array], Array]): The pooling operation to be applied.
        window (Sequence[int]): The dimensions of the pooling window.
        strides (Sequence[int]): The stride of the pooling
            window. Defaults to the window dimensions if not provided.
        padding (Union[Sequence[Tuple[int, int]], str]): The padding
            strategy, can be 'VALID' or 'SAME', or explicitly defined.
        normalize (bool): If True, normalizes the pooling output.

    Specializations:
        - MaxPooling: Performs max pooling over the inputs.
        - MinPooling: Performs min pooling over the inputs.
        - AvgPooling: Performs average pooling over the inputs, with normalization.

    Example:
        ```python
        # Max Pooling example
        max_pool = MaxPooling(window=(2, 2), strides=(2, 2))
        pooled_output = max_pool(input_array)
        ```

    """

    init_value: ArrayLike
    op: Callable[[Array, Array], Array]
    window: Sequence[int]
    strides: Sequence[int] = ()
    padding: Sequence[tuple[int, int]] | str = "VALID"
    normalize: bool = False

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        assert len(x.shape) >= len(
            self.window,
        ), "input must have at least as many dimensions as pooling window"
        window = (1,) * (len(x.shape) - len(self.window)) + tuple(
            self.window,
        )
        strides = self.strides or window
        strides = (1,) * (len(x.shape) - len(strides)) + tuple(strides)
        pad = self.padding
        if not isinstance(pad, str):
            pad = ((0, 0),) * (len(x.shape) - len(pad)) + tuple(pad)

        y = lax.reduce_window(x, self.init_value, self.op, window, strides, pad)

        if self.normalize:
            y /= lax.reduce_window(
                jnp.ones_like(x),
                self.init_value,
                self.op,
                window,
                strides,
                pad,
            )

        return y


MaxPooling = partial(Pooling, -jnp.inf, lax.max)
MinPooling = partial(Pooling, +jnp.inf, lax.min)
AvgPoolng = partial(Pooling, 0, lax.add, normalize=True)


class Norm(PyTree):
    """A generic normalization layer that can be used for various normalization techniques, including layer normalization.

    This layer normalizes the input data along a specified axis, with options to subtract the mean and/or apply
    a scale (weight) and shift (bias). It is particularly useful for layer normalization when the axis is set to
    the feature axis.

    Attributes:
        weight (Optional[Array]): The scale factor applied to the normalized data. If None, no scaling is applied.
        bias (Optional[Array]): The shift applied to the scaled data. If None, no shifting is applied.
        axis (int): The axis along which to normalize the data.
        subtract_mean (bool): If True, subtracts the mean from the data before normalization.

    Example:
        ```python
        # Layer normalization example
        layer = layer_norm(shape=(feature_dim,))
        normalized_output = layer(input_array)
        ```

    """

    weight: Array | None = None
    bias: Array | None = None
    axis: int | Sequence[int] = static(default=-1)
    subtract_mean: bool = static(default=True)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        del args, kwargs
        if self.subtract_mean:
            x = x - jnp.mean(x, axis=self.axis, keepdims=True)

        denom = jnp.sqrt(jnp.square(x).mean(axis=self.axis, keepdims=True))
        denom = jnp.clip(denom, 1e-6, None)
        x = x / denom

        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def norm(
    shape: tuple,
    axis: int | Sequence[int],
    *,
    use_weight: bool = True,
    use_bias: bool = True,
    subtract_mean: bool = True,
    key: Array | None = None,  # for compatibility with other factories
) -> Norm:
    """Constructs a Norm layer with specified configurations for normalization.

    This function initializes a Norm layer, allowing customization of the normalization process. It supports
    the use of scaling (weight) and shifting (bias), and can subtract the mean from the data before normalization.
    It is often used for implementing layer normalization, but can be configured for other normalization types.

    Args:
        shape (tuple): The shape of the scale and shift parameters, typically matching the dimensionality of the feature axis.
        axis (int): The axis along which to normalize the data.
        use_weight (bool): If True, initializes a scale parameter (weight) with ones. If False, weight is set to None.
        use_bias (bool): If True, initializes a shift parameter (bias) with zeros. If False, bias is set to None.
        subtract_mean (bool): If True, subtracts the mean from the data before normalization.
        key (Array, optional): A JAX random key, unused in this function but included for compatibility with other
            layer constructors.

    Returns:
        Norm: An instance of the Norm layer configured with the specified parameters.

    Example:
        ```python
        # Example usage for layer normalization
        layer = norm(shape=(feature_dim,), axis=-1)
        normalized_output = layer(input_array)
        ```

    """
    del key
    return Norm(
        jnp.ones(shape) if use_weight else None,
        jnp.zeros(shape) if use_bias else None,
        axis=axis,
        subtract_mean=subtract_mean,
    )


layer_norm = partial(norm, axis=-1)
