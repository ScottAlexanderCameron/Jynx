import typing as tp
from collections.abc import Callable, Sequence
from functools import partial

import jax.nn.initializers as init
from jax import Array, nn
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer

from ..pytree import PyTree, static
from .containers import Recurrent, Sequential, sequential
from .linear import conv, conv_transpose, linear
from .misc import MaxPooling
from .module import Module
from .rnn import gru_cell, lstm_cell, rnn_cell
from .static import Dropout, Fn


def mlp(
    sizes: Sequence[int],
    activation: Callable[[Array], Array] = nn.relu,
    final_activation: Callable[[Array], Array] | None = None,
    dropout: float | None = None,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> Sequential:
    """Multilayer perceptron.

    Args:
        `sizes`: A sequence of integers representing the sizes of each
            layer in the MLP, including the input and output layers.
        `activation`: The activation function applied to the output of
            each hidden layer.
        `final_activation`: An optional activation function applied to
            the output layer. If `None`, no activation is applied to the
            final layer.
        `dropout`: An optional dropout rate applied after each hidden
            layer to prevent overfitting. If `None`, no dropout is applied.
        `weight_init`: The initializer for the weight matrices of
            each layer.
        `bias_init`: The initializer for the bias vectors of each layer.
        `key`: A JAX random key used for initializing the weights
            and biases.

    Returns:
        A `Sequential` container encapsulating the layers of the MLP,
        including linear transformations, activations, and optionally
        dropout.

    Example:
        ```python
        import jax
        from jax import random

        # Define the sizes of each layer in the MLP, including the input and output layers
        layer_sizes = [784, 256, 128, 10]

        # Generate a random key for initializing the layers
        key = random.PRNGKey(42)

        key, k1 = random.split(key)
        # Create the MLP model with ReLU activations between layers and a softmax activation at the output
        mlp_model = mlp(
            sizes=layer_sizes,
            activation=jax.nn.relu,
            final_activation=jax.nn.softmax,
            dropout=0.2,
            key=k1,
        )

        # Assuming `input_data` is a batch of input vectors
        # output = mlp_model(input_data, key=key)
        # or to turn off dropout:
        # output = mlp_model(input_data)
        ```

    """
    depth = len(sizes) - 1
    step = 2 if dropout is None else 3
    n_layers = 1 + (depth - 1) * step

    layers: list = [Fn(activation)] * n_layers
    layers[::step] = (
        linear(
            si,
            so,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k,
        )
        for si, so, k in zip(sizes[:-1], sizes[1:], rnd.split(key, depth), strict=False)
    )

    if dropout is not None:
        layers[2::step] = [Dropout(dropout)] * (depth - 1)

    if final_activation is not None:
        layers.append(Fn(final_activation))

    return Sequential(layers)


class RNNCellFactory(tp.Protocol):
    def __call__(
        self,
        in_size: int,
        state_size: int,
        *,
        weight_init: Initializer,
        bias_init: Initializer,
        key: Array,
    ) -> tp.Any:
        ...


def rnn(
    in_size: int,
    state_size: int,
    out_size: int,
    num_layers: int = 1,
    dropout: float | None = None,
    final_activation: Callable[[Array], Array] | None = None,
    cell_factory: RNNCellFactory = rnn_cell,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> Recurrent:
    key, k = rnd.split(key)
    layers: list = [
        cell_factory(
            in_size,
            state_size,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k,
        ),
    ]

    for i in range(1, num_layers):
        if dropout is not None:
            layers.append(Dropout(dropout))
        key, k = rnd.split(key)
        layers.append(
            cell_factory(
                state_size,
                state_size if i < num_layers - 1 else out_size,
                weight_init=weight_init,
                bias_init=bias_init,
                key=k,
            ),
        )

    if final_activation is not None:
        layers.append(Fn(final_activation))

    return Recurrent(layers)


lstm = partial(rnn, cell_factory=lstm_cell)
gru = partial(rnn, cell_factory=gru_cell)


class UNet[**P](PyTree):
    type Block = Module[tp.Concatenate[Array, P], Array]
    type UpDownBlock = Module[[Array], Array]

    front: Block
    down: UpDownBlock
    middle: Block | tp.Self
    up: UpDownBlock
    back: Block

    concat_axis: int = static(default=1)

    def __call__(self, x: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        x = self.front(x, *args, **kwargs)
        z = self.down(x)
        z = self.middle(z, *args, **kwargs)
        z = self.up(z)
        x = jnp.concatenate((x, z), axis=self.concat_axis)
        x = self.back(x, *args, **kwargs)
        return x


class UNetBlockFactory[Block](tp.Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: Array,
    ) -> Block:
        ...


def unet_max_pooling(
    in_channels: int,
    out_channels: int,
    key: Array,
) -> Module[[Array], Array]:
    del in_channels, out_channels, key
    return MaxPooling((2, 2), (2, 2))


def unet_conv_transpose(
    in_channels: int,
    out_channels: int,
    key: Array,
) -> Module[[Array], Array]:
    return conv_transpose(in_channels, out_channels, (2, 2), (2, 2), key=key)


def unet_conv_block(
    in_channels: int,
    out_channels: int,
    key: Array,
    hidden_channels: int | None = None,
    activation: Module[[Array], Array] = nn.silu,
    kernel_shape: Sequence[int] = (3, 3),
) -> Module[[Array], Array]:
    if hidden_channels is None:
        hidden_channels = max(in_channels, out_channels)

    k1, k2 = rnd.split(key)
    return sequential(
        conv(in_channels, hidden_channels, kernel_shape, padding="SAME", key=k1),
        activation,
        conv(hidden_channels, out_channels, kernel_shape, padding="SAME", key=k2),
    )


def unet[**P](
    depth: int,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    *,
    concat_axis: int = 1,
    expansion_factor: int = 2,
    block_factory: UNetBlockFactory[UNet[P].Block] = unet_conv_block,
    down_block_factory: UNetBlockFactory[UNet[P].UpDownBlock] = unet_max_pooling,
    up_block_factory: UNetBlockFactory[UNet[P].UpDownBlock] = unet_conv_transpose,
    key: Array,
) -> UNet[P]:
    k1, k2, k3, k4, k5 = rnd.split(key, 5)
    middle: UNet[P].Block | UNet[P]
    if depth == 0:
        middle = block_factory(hidden_channels, hidden_channels, key=k1)
    else:
        middle = unet(
            depth - 1,
            hidden_channels,
            hidden_channels,
            expansion_factor * hidden_channels,
            concat_axis=concat_axis,
            block_factory=block_factory,
            down_block_factory=down_block_factory,
            up_block_factory=up_block_factory,
            key=k1,
        )

    return UNet(
        front=block_factory(in_channels, hidden_channels, key=k2),
        down=down_block_factory(hidden_channels, hidden_channels, key=k3),
        middle=middle,
        up=up_block_factory(hidden_channels, hidden_channels, key=k4),
        back=block_factory(2 * hidden_channels, out_channels, key=k5),
        concat_axis=concat_axis,
    )
