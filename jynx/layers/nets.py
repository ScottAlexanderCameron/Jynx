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
from .module import Module, RecurrentModule
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
    """A protocol for rnn cell constructors."""

    def __call__(
        self,
        in_size: int,
        state_size: int,
        *,
        weight_init: Initializer,
        bias_init: Initializer,
        key: Array,
    ) -> RecurrentModule:
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
    """Constructs a recurrent neural network with a given cell type:
    an RNNCell, LSTMCell, or GRUCell.

    Args:
        in_size (int): size of the input vector.
        state_size (int): size of the hidden state, also the input size
            of the hidden layers.
        out_size (int): the size of the output vector.
        num_layers (int): number of rnn layers, default 1.
        dropout: An optional dropout rate applied after each hidden
            layer to prevent overfitting. If `None`, no dropout is applied.
        final_activation: An optional activation function applied to
            the output layer. If `None`, no activation is applied to the
            final layer.
        cell_factory (RNNCellFactory): the factory function for creating
            the recurent cells. Can be one of `rnn_cell`, `lstm_cell`, or
            `gru_cell`.
        weight_init: The initializer for the weight matrices of
            each layer.
        bias_init: The initializer for the bias vectors of each layer.
        key: A JAX random key used for initializing the weights
            and biases.
    """
    key, k = rnd.split(key)
    layers = []

    for i in range(num_layers):
        if dropout is not None and i > 0:
            layers.append(Dropout(dropout))
        key, k = rnd.split(key)
        layers.append(
            cell_factory(
                state_size if i > 0 else in_size,
                state_size,
                weight_init=weight_init,
                bias_init=bias_init,
                key=k,
            ),
        )

    key, k = rnd.split(key)
    layers.append(linear(state_size, out_size, key=k))

    if final_activation is not None:
        layers.append(Fn(final_activation))

    return Recurrent(layers)


lstm = partial(rnn, cell_factory=lstm_cell)
gru = partial(rnn, cell_factory=gru_cell)


class UNet[**P](PyTree):
    """Generic UNet module. The UNet architecture can be illustrated as
    follows:
    ```
        ╭─────╮                        ╭──────╮
       →│front│┬────────concat────────┬│ back │→
        ╰─────╯│┌────┐          ┌────┐│╰──────╯
               └│down│┐╔══════╗┌│ up │┘
                └────┘└║middle║┘└────┘
                       ╚══════╝
    ```
    These blocks can be modules of almost any type, provided they
    take Arrays as input and return Arrays as output. Deep UNets
    are defined recursively with the `middle` block being another UNet.
    `down` and `up` are typically downsampling and upsampling blocks
    such as MaxPooling and ConvTranspose. `*args` and `**kwargs` are
    passed on to the `front`, `middle` and `back` blocks.

    Type Aliases:
        Block: generic module which can take extra args
        UpDownBlock: module for up and down sampling

    Attributes:
        front (Block): first layer
        down (UpDownBlock): down sampling
        middle (Block | UNet): recursive unet or middle block
        up (UpDownBlock): up sampling
        back (Block): final layer
        concat_axis (int): the feature axis for concatenating skip connections.
            defaults to 1, which is reasonable for inputs with shape
            `(Batch, Channels, Spatial...)` etc. But may not be the correct
            choice when multiple batch dimensions are used or when feature
            and spatial dimensions are in a different order.
    """

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
    """Protocol for UNet block constructors.

    Args:
        in_channels (int): number of features the block should take as input
        out_channels (int): number of features the block shoud output
        depth (int): depth from the bottom of the UNet that this block lies.
            Will be zero for the `middle` block on the deepest level,
            1 for the `front` and `back` blocks just above that etc.
            This value can be used to control the complexity of the
            blocks as a function of their position in the model.
        key (Array): rng key for layer initialization.
    """

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        *,
        key: Array,
    ) -> Block:
        ...


def unet_max_pooling(
    in_channels: int,
    out_channels: int,
    depth: int = 0,
    *,
    key: Array,
) -> Module[[Array], Array]:
    """Standard 2x2 max pooling. Downsamples images by a
    factor 2 in each dimension.
    """
    del in_channels, out_channels, depth, key
    return MaxPooling((2, 2), (2, 2))


def unet_conv_transpose(
    in_channels: int,
    out_channels: int,
    depth: int = 0,
    *,
    key: Array,
) -> Module[[Array], Array]:
    """Conv transpose with 2x2 kernel and 2x2 strides.
    Upsamples images by a factor 2 in each dimension.
    """
    del depth
    return conv_transpose(in_channels, out_channels, (2, 2), (2, 2), key=key)


def unet_conv_block(
    in_channels: int,
    out_channels: int,
    depth: int = 0,
    *,
    key: Array,
    hidden_channels: int | None = None,
    activation: Module[[Array], Array] = nn.silu,
    kernel_shape: Sequence[int] = (3, 3),
) -> Module[[Array], Array]:
    """A simple convolution block consisting of two convolution layers with
    an activation function between.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        depth (int): ignored. For compatibility with unet constructor.
        key (Array): rng key.
        hidden_channels (int): number of hidden channels, defaults to the
            maximum of input and output channels.
        activation (Callable): nonlinearity. defaults to silu
        kernel_shape: shape of the convolution kernel, defaults to (2, 2)

    Returns:
        Conv block.
    """
    del depth
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
    expansion_factor: int | Sequence[int] = 2,
    block_factory: UNetBlockFactory[UNet[P].Block] = unet_conv_block,
    down_block_factory: UNetBlockFactory[UNet[P].UpDownBlock] = unet_max_pooling,
    up_block_factory: UNetBlockFactory[UNet[P].UpDownBlock] = unet_conv_transpose,
    key: Array,
) -> UNet[P]:
    """Constructs a UNet recursively. See `UNet` for details of this architecture.

    Args:
        depth (int): how deep the network is. A depth of 1 will have 1
            downsampling layer, a middle block and 1 upsampling layer. A depth
            of 2 will have 2 downsampling layers before the middle block
            and 2 upsampling layers, etc. Must be >= 1.
        in_channels (int): number of input channels/features.
        out_channels (int): number of output channels/features.
        hidden_channels (int): number of channels/features in the skip connection
            from the first block to the last.
        concat_axis (int): feature axis. Since this architecture is commonly used
            with convolution neural networks, the default is 1, although this may
            be changed in future.
        expansion_factor (int | Sequence[int]): multiplicative factor for the number
            of features in deeper layers of the UNet. For example, with `depth` 4
            `hidden_channels` set to 8, and `expansion_factor` set to `[2, 4, 4]`,
            the number of output channels of the downsampling layers will be
            8, 16, 64, 256. Note that these factors are cumulative, so the number
            of output channels from the final downsampling layer here is
            4 * 4 * 2 * 8. If an int is provided, then the same expansion_factor
            is used at all depths. Eg: this is equivalent to providing
            `[expansion_factor] * (depth - 1)`
        block_factory (UNetBlockFactory): a callable to create the blocks in the
            front, back, and middle of the network. Takes as input arguments
            `in_channels`, `out_channels`, `depth` and `key`, and should return
            a network module which has the corresponding number of input and
            output channels/features. `depth` is provided in case the user wishes
            to vary the architecture of the model by depth. The middle block
            is constrructed with a depth of 0.
        down_block_factory (UNetBlockFactory): callable to construct the downsampling
            blocks. Defaults to max pooling.
        up_block_factory (UNetBlockFactory): callable to construct the upsampling
            blocks. Defaults to conv transpose.
        key (Array): rng key.

    Returns:
        UNet
    """
    assert depth >= 1
    k1, k2, k3, k4, k5 = rnd.split(key, 5)
    middle: UNet[P].Block | UNet[P]
    if isinstance(expansion_factor, int):
        expansion_factor = [expansion_factor] * (depth - 1)
    elif len(expansion_factor) < depth - 1:
        expansion_factor = list(expansion_factor) + [expansion_factor[-1]] * (
            depth - len(expansion_factor)
        )

    if depth == 1:
        middle = block_factory(hidden_channels, hidden_channels, depth - 1, key=k1)
    else:
        middle = unet(
            depth - 1,
            hidden_channels,
            hidden_channels,
            expansion_factor[0] * hidden_channels,
            concat_axis=concat_axis,
            expansion_factor=expansion_factor[1:],
            block_factory=block_factory,
            down_block_factory=down_block_factory,
            up_block_factory=up_block_factory,
            key=k1,
        )

    return UNet(
        front=block_factory(in_channels, hidden_channels, depth, key=k2),
        down=down_block_factory(hidden_channels, hidden_channels, depth, key=k3),
        middle=middle,
        up=up_block_factory(hidden_channels, hidden_channels, depth, key=k4),
        back=block_factory(2 * hidden_channels, out_channels, depth, key=k5),
        concat_axis=concat_axis,
    )
