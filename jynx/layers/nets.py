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
    final_activation: tp.Optional[tp.Callable[[Array], Array]] = None,
    dropout: tp.Optional[float] = None,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> Sequential:
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
        for si, so, k in zip(sizes[:-1], sizes[1:], rnd.split(key, depth))
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
    dropout: tp.Optional[float] = None,
    final_activation: tp.Optional[tp.Callable[[Array], Array]] = None,
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
        )
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
            )
        )

    if final_activation is not None:
        layers.append(Fn(final_activation))

    return Recurrent(layers)


lstm = partial(rnn, cell_factory=lstm_cell)
gru = partial(rnn, cell_factory=gru_cell)
