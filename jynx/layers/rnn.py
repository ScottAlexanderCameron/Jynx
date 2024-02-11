import typing as tp

import jax.nn.initializers as init
from jax import Array, nn
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer

from ..pytree import PyTree, static
from .linear import _maybe_add_bias


class RNNCell(PyTree):
    weight_x: Array
    weight_h: Array
    bias: tp.Optional[Array]
    activation: tp.Callable[[Array], Array] = static(default=nn.sigmoid)

    def __call__(
        self, x: Array, state: Array, *args, **kwargs
    ) -> tp.Tuple[Array, Array]:
        del args, kwargs
        state = self.activation(
            _maybe_add_bias(self.bias, x @ self.weight_x + state @ self.weight_h)
        )
        return state, state

    @property
    def hidden_size(self):
        return self.weight_h.shape[0]

    @property
    def initial_state(self):
        return jnp.zeros((1, self.hidden_size))


def rnn_cell(
    in_size: int,
    state_size: int,
    activation: tp.Callable[[Array], Array] = nn.sigmoid,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> RNNCell:
    k1, k2, k3 = rnd.split(key, 3)
    return RNNCell(
        weight_init(k1, (in_size, state_size)),
        weight_init(k2, (state_size, state_size)),
        bias_init(k3, (state_size,)) if bias_init is not None else None,
        activation,
    )


class GRUCell(PyTree):
    update_gate: RNNCell
    reset_gate: RNNCell
    candidate_cell: RNNCell

    def __call__(
        self, x: Array, state: Array, *args, **kwargs
    ) -> tp.Tuple[Array, Array]:
        del args, kwargs
        r, _ = self.reset_gate(x, state)
        z, _ = self.update_gate(x, state)
        h_hat, _ = self.candidate_cell(x, r * state)
        state = (1 - z) * state + z * h_hat

        return state, state

    @property
    def hidden_size(self):
        return self.update_gate.hidden_size

    @property
    def initial_state(self):
        return jnp.zeros((1, self.hidden_size))


def gru_cell(
    in_size: int,
    state_size: int,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> GRUCell:
    c1, c2, c3 = (
        rnn_cell(
            in_size,
            state_size,
            act,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k,
        )
        for act, k in zip((nn.sigmoid, nn.sigmoid, nn.tanh), rnd.split(key, 3))
    )
    return GRUCell(c1, c2, c3)


class LSTMCell(PyTree):
    input_gate: RNNCell
    output_gate: RNNCell
    forget_gate: RNNCell
    cell_gate: RNNCell

    def __call__(
        self, x: Array, state: tp.Tuple[Array, Array], *args, **kwargs
    ) -> tp.Tuple[Array, tp.Tuple[Array, Array]]:
        del args, kwargs
        h, c = state
        inp, _ = self.input_gate(x, h)
        out, _ = self.output_gate(x, h)
        f, _ = self.forget_gate(x, h)
        c_hat, _ = self.cell_gate(x, h)
        c = f * c + inp * c_hat
        h = out * jnp.tanh(c)
        return h, (h, c)

    @property
    def hidden_size(self):
        return self.output_gate.hidden_size

    @property
    def initial_state(self):
        shape = (1, self.hidden_size)
        return jnp.zeros(shape), jnp.zeros(shape)


def lstm_cell(
    in_size: int,
    state_size: int,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> LSTMCell:
    c1, c2, c3, c4 = (
        rnn_cell(
            in_size,
            state_size,
            act,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k,
        )
        for act, k in zip(
            (nn.sigmoid, nn.sigmoid, nn.sigmoid, nn.tanh), rnd.split(key, 4)
        )
    )
    return LSTMCell(c1, c2, c3, c4)
