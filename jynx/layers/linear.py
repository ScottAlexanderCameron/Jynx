from collections.abc import Sequence

import jax.nn.initializers as init
from jax import Array, lax
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer
from jax.typing import ArrayLike

from ..pytree import PyTree, static


class Linear(PyTree):
    """A fully connected neural network layer that performs a linear transformation to the input data.

    This layer applies a linear transformation to the incoming data, defined as `y = xW + b` where `x` is the input,
    `W` is the layer's weights, and `b` is the bias term. The bias term is optional and can be omitted.

    Attributes:
        weight (Array): The weight matrix of the layer.
        bias (Optional[Array]): The bias vector of the layer, if any.

    Methods:
        __call__(self, x, *args, **kwargs) -> Array:
            Performs the linear transformation on the input data `x`.

    """

    weight: Array
    bias: Array | None

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        """Applies the linear transformation to the input data `x`.

        Args:
            x (Array): The input data to the layer.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Array: The transformed data.

        """
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
    """Initializes a Linear layer with the specified dimensions and initializers.

    Args:
        in_size (int): The size (number of features) of the input data.
        out_size (int): The size (number of features) of the output data.
        weight_init (Initializer): The initializer for the weight matrix.
        bias_init (Initializer): The initializer for the bias vector.
        key (Array): A JAX random key used for initializing weights and biases.

    Returns:
        Linear: An instance of the Linear layer initialized with the specified parameters.

    """
    k1, k2 = rnd.split(key)
    return Linear(
        weight_init(k1, (in_size, out_size)),
        bias_init(k2, (out_size,)) if bias_init is not None else None,
    )


class Conv(PyTree):
    """A convolutional layer that applies a convolution operation to the input data.

    This layer performs a convolution over the input data using a set
    of learnable filters, optionally followed by adding a bias term.
    The operation is defined by parameters such as the kernel size,
    stride, and padding.

    Attributes:
        kernel (Array): The convolution kernel (set of filters).
        bias (Optional[Array]): The bias vector for each filter, if any.
        strides (Sequence[int]): The stride of the convolution.
        padding (Union[Sequence[Tuple[int, int]], str]): The padding strategy.
            Can be a string ('SAME', 'VALID') or a sequence of tuples for explicit padding.

    Methods:
        __call__(self, x, *args, **kwargs) -> Array:
            Applies the convolution operation to the input data `x`.

    """

    kernel: Array
    bias: Array | None
    strides: Sequence[int] = static()
    padding: Sequence[tuple[int, int]] | str = static(default="VALID")
    channels_last: bool = static(default=False)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        """Applies the convolution operation to the input data `x`.

        Args:
            x (Array): The input data to the layer.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Array: The convolved data.

        """
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
                dimension_numbers=_dimension_numbers(
                    x.shape,
                    self.kernel.shape,
                    self.channels_last,
                ),
            ),
        )
        return x.reshape((*extra_dims, -1, *x.shape[1:]))


def conv(
    in_channels: int,
    out_channels: int,
    kernel_shape: Sequence[int],
    strides: Sequence[int] | None = None,
    padding: Sequence[tuple[int, int]] | str = "VALID",
    *,
    kernel_init: Initializer = init.kaiming_normal(1, 0),
    bias_init: Initializer | None = init.normal(),
    channels_last: bool | None = None,
    key: Array,
) -> Conv:
    """Initializes a Conv layer with the specified dimensions, initializers, and convolution parameters.

    This function creates a Conv layer instance, setting up the convolution kernel and bias with the provided
    initializers, and configuring the convolution operation with the specified stride and padding.

    Args:
        in_channels (int): The number of input channels (depth of the input).
        out_channels (int): The number of output channels (depth of the output).
        kernel_shape (Sequence[int]): The shape of the convolution kernel.
        strides (Optional[Sequence[int]]): The stride of the convolution. If None, defaults to 1 along each dimension.
        padding (Union[Sequence[Tuple[int, int]], str]): The padding strategy. Can be a string ('SAME', 'VALID') or
            a sequence of tuples for explicit padding.
        kernel_init (Initializer): The initializer for the convolution kernel.
        bias_init (Initializer): The initializer for the bias vector.
        key (Array): A JAX random key used for initializing the kernel and bias.

    Returns:
        Conv: An instance of the Conv layer initialized with the specified parameters and convolution configuration.

    """
    k1, k2 = rnd.split(key)
    if not isinstance(padding, str):
        padding = tuple(padding)

    if channels_last is None:
        channels_last = False

    if channels_last:
        bias_shape = (out_channels,)
    else:
        bias_shape = (out_channels,) + (1,) * len(kernel_shape)

    return Conv(
        kernel_init(k1, (out_channels, in_channels) + tuple(kernel_shape)),
        (bias_init(k2, bias_shape) if bias_init is not None else None),
        tuple(strides or (1,) * len(kernel_shape)),
        padding,
        channels_last,
    )


class ConvTranspose(PyTree):
    """A convolution transpose (deconvolution) layer for neural networks.

    This layer applies a convolution transpose operation to the input data, which is often used for upsampling in
    models like autoencoders or generative networks. The operation is defined by parameters such as the kernel size,
    stride, and padding, and can optionally include a bias term.

    Attributes:
        kernel (Array): The convolution transpose kernel (set of filters).
        bias (Optional[Array]): The bias vector for each filter, if any.
        strides (Sequence[int]): The stride of the convolution transpose.
        padding (Union[Sequence[Tuple[int, int]], str]): The padding strategy. Can be a string ('SAME', 'VALID') or
            a sequence of tuples for explicit padding.

    Methods:
        __call__(self, x, *args, **kwargs) -> Array:
            Applies the convolution transpose operation to the input data `x`.

    """

    kernel: Array
    bias: Array | None
    strides: Sequence[int] = static()
    padding: Sequence[tuple[int, int]] | str = static(default="VALID")
    channels_last: bool = static(default=False)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        """Applies the convolution transpose operation to the input data `x`.

        Args:
            x (Array): The input data to the layer.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Array: The output data after applying the convolution transpose operation.

        """
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
                dimension_numbers=_dimension_numbers(
                    x.shape,
                    self.kernel.shape,
                    self.channels_last,
                ),
            ),
        )
        return x.reshape((*extra_dims, -1, *x.shape[1:]))


def _dimension_numbers(x_shape, kernel_shape, channels_last):
    assert len(x_shape) >= 2, "input tensor must have at least 2 dimensions"

    if channels_last:
        chars = "".join(chr(ord("a") + i) for i in range(len(x_shape) - 2))
        in_spec = f"N{chars}C"
        kernel_spec = f"OI{chars}"
        dim_spec = (in_spec, kernel_spec, in_spec)
    else:
        dim_spec = None

    return lax.conv_dimension_numbers(
        x_shape,
        kernel_shape,
        dim_spec,
    )


def conv_transpose(
    in_channels: int,
    out_channels: int,
    kernel_shape: Sequence[int],
    strides: Sequence[int] | None = None,
    padding: Sequence[tuple[int, int]] | str = "VALID",
    *,
    kernel_init: Initializer = init.kaiming_normal(1, 0),
    bias_init: Initializer | None = init.normal(),
    channels_last: bool | None = None,
    key: Array,
) -> ConvTranspose:
    """Initializes a ConvTranspose layer with specified dimensions, initializers, and convolution transpose parameters.

    This function creates a ConvTranspose layer instance, setting up the convolution transpose kernel and bias with
    the provided initializers, and configuring the convolution transpose operation with the specified stride and padding.

    Args:
        in_channels (int): The number of input channels (depth of the input).
        out_channels (int): The number of output channels (depth of the output).
        kernel_shape (Sequence[int]): The shape of the convolution transpose kernel.
        strides (Optional[Sequence[int]]): The stride of the convolution transpose. If None, defaults to 1 along each dimension.
        padding (Union[Sequence[Tuple[int, int]], str]): The padding strategy. Can be a string ('SAME', 'VALID') or
            a sequence of tuples for explicit padding.
        kernel_init (Initializer): The initializer for the convolution transpose kernel.
        bias_init (Initializer): The initializer for the bias vector.
        key (Array): A JAX random key used for initializing the kernel and bias.

    Returns:
        ConvTranspose: An instance of the ConvTranspose layer initialized with the specified parameters and
        convolution transpose configuration.

    """
    k1, k2 = rnd.split(key)
    if not isinstance(padding, str):
        padding = tuple(padding)

    if channels_last is None:
        channels_last = False

    if channels_last:
        bias_shape = (out_channels,)
    else:
        bias_shape = (out_channels,) + (1,) * len(kernel_shape)

    return ConvTranspose(
        kernel_init(k1, (out_channels, in_channels) + tuple(kernel_shape)),
        (bias_init(k2, bias_shape) if bias_init is not None else None),
        tuple(strides or (1,) * len(kernel_shape)),
        padding,
        channels_last,
    )


class Embedding(PyTree):
    """An embedding layer for neural networks.

    This layer maps positive integer indices to dense vectors of fixed size. It's commonly used in models that deal
    with categorical data, especially in natural language processing tasks.

    Attributes:
        weight (Array): The embedding matrix with shape (num_embeddings, embedding_dim).

    Methods:
        __call__(self, x, *args, **kwargs) -> Array:
            Looks up the embedding vectors corresponding to the indices in `x`.

    """

    weight: Array

    def __call__(self, x: ArrayLike, *args, **kwargs) -> Array:
        """Looks up the embedding vectors corresponding to the indices in `x`.

        Args:
            x (ArrayLike): The input indices to the layer.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Array: The resulting embedding vectors.

        """
        del args, kwargs
        return jnp.take(self.weight, x, axis=0)


def embedding(
    size: int,
    num_classes: int,
    *,
    weight_init: Initializer = init.orthogonal(),
    key: Array,
) -> Embedding:
    """Initializes an Embedding layer with the specified dimensions and initializer.

    Args:
        size (int): The size of each embedding vector.
        num_classes (int): The number of distinct embeddings (often the vocabulary size in NLP applications).
        weight_init (Initializer): The initializer for the embedding matrix.
        key (Array): A JAX random key used for initializing the embedding matrix.

    Returns:
        Embedding: An instance of the Embedding layer initialized with the specified parameters.

    """
    return Embedding(weight_init(key, (num_classes, size)))


def _maybe_add_bias(bias, x):
    """Adds a bias vector to the input data if a bias is provided.

    This is a utility function used internally by the layer classes to apply bias terms to the output of linear
    and convolution operations.

    Args:
        bias (Optional[Array]): The bias vector to be added.
        x (Array): The input data to which the bias should be added.

    Returns:
        Array: The result of adding the bias to `x`, or `x` unchanged if no bias is provided.

    """
    if bias is not None:
        x = x + bias
    return x
