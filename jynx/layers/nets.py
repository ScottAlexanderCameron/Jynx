import typing as tp
from functools import partial

import jax.nn.initializers as init
from jax import Array, nn
from jax import random as rnd
from jax.nn.initializers import Initializer

from .containers import Recurrent, Sequential
from .linear import linear
from .rnn import gru_cell, lstm_cell, rnn_cell
from .static import Dropout, Fn


def mlp(
    sizes: tp.Sequence[int],
    activation: tp.Callable[[Array], Array] = nn.relu,
    final_activation: tp.Callable[[Array], Array] | None = None,
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
    final_activation: tp.Callable[[Array], Array] | None = None,
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
