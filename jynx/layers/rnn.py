from collections.abc import Callable

import jax.nn.initializers as init
from jax import Array, nn
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer

from ..pytree import PyTree, static
from .linear import _maybe_add_bias


class RNNCell(PyTree):
    """A basic recurrent neural network (RNN) cell.

    This cell implements the simplest form of RNN, where the new state
    is computed based on the current input and the previous state,
    passed through an activation function.

    Attributes:
        weight_x (Array): Weights for input x.
        weight_h (Array): Weights for previous state.
        bias (Optional[Array]): Optional bias term.
        activation (Callable[[Array], Array]): Activation function applied to state update.

    The `initial_state` property returns the initial state of the RNN cell, typically zeros.

    """

    weight_x: Array
    weight_h: Array
    bias: Array | None
    activation: Callable[[Array], Array] = static(default=nn.sigmoid)

    def __call__(
        self,
        x: Array,
        state: Array,
        *args,
        **kwargs,
    ) -> tuple[Array, Array]:
        del args, kwargs
        state = self.activation(
            _maybe_add_bias(self.bias, x @ self.weight_x + state @ self.weight_h),
        )
        return state, state

    @property
    def hidden_size(self):
        return self.weight_h.shape[0]

    @property
    def initial_state(self):
        """Returns the initial state of the RNN cell, typically zeros."""
        return jnp.zeros((1, self.hidden_size))


def rnn_cell(
    in_size: int,
    state_size: int,
    activation: Callable[[Array], Array] = nn.sigmoid,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> RNNCell:
    """Constructs an RNNCell with specified dimensions, activation, and initializers.

    Args:
        in_size (int): The size of the input dimension.
        state_size (int): The size of the state dimension.
        activation (Callable[[Array], Array]): The activation function
            applied to the output state. Defaults to sigmoid.
        weight_init (Initializer): The initializer for the weight
            matrices. Defaults to kaiming_normal.
        bias_init (Initializer): The initializer for the bias
            vector. Defaults to normal.
        key (Array): A JAX random key used for initializing weights
            and biases.

    Returns:
        RNNCell: An instance of RNNCell initialized with the specified parameters.

    """
    k1, k2, k3 = rnd.split(key, 3)
    return RNNCell(
        weight_init(k1, (in_size, state_size)),
        weight_init(k2, (state_size, state_size)),
        bias_init(k3, (state_size,)) if bias_init is not None else None,
        activation,
    )


class GRUCell(PyTree):
    """A Gated Recurrent Unit (GRU) cell.

    GRU is an advanced RNN variant that includes update and reset gates,
    improving the ability to capture dependencies and mitigate vanishing
    gradient issues.

    Attributes:
        update_gate (RNNCell): The update gate cell.
        reset_gate (RNNCell): The reset gate cell.
        candidate_cell (RNNCell): The candidate state cell.

    The `initial_state` property returns the initial state of the GRU cell, typically zeros.

    """

    update_gate: RNNCell
    reset_gate: RNNCell
    candidate_cell: RNNCell

    def __call__(
        self,
        x: Array,
        state: Array,
        *args,
        **kwargs,
    ) -> tuple[Array, Array]:
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
        """Returns the initial state of the GRU cell, typically zeros."""
        return jnp.zeros((1, self.hidden_size))


def gru_cell(
    in_size: int,
    state_size: int,
    *,
    weight_init: Initializer = init.kaiming_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> GRUCell:
    """Constructs a GRUCell with specified dimensions and initializers.

    Args:
        in_size (int): The size of the input dimension.
        state_size (int): The size of the state dimension.
        weight_init (Initializer): The initializer for the weight matrices. Defaults to kaiming_normal.
        bias_init (Initializer): The initializer for the bias vector. Defaults to normal.
        key (Array): A JAX random key used for initializing the gates' weights and biases.

    Returns:
        GRUCell: An instance of GRUCell initialized with the specified
        parameters, comprising update, reset, and candidate cells.

    """
    c1, c2, c3 = (
        rnn_cell(
            in_size,
            state_size,
            act,
            weight_init=weight_init,
            bias_init=bias_init,
            key=k,
        )
        for act, k in zip(
            (nn.sigmoid, nn.sigmoid, nn.tanh),
            rnd.split(key, 3),
            strict=False,
        )
    )
    return GRUCell(c1, c2, c3)


class LSTMCell(PyTree):
    """A Long Short-Term Memory (LSTM) cell.

    LSTM is a type of RNN that includes input, output, and forget gates,
    significantly improving the network's ability to capture long-term
    dependencies and mitigate vanishing or exploding gradient issues.

    Attributes:
        input_gate (RNNCell): The input gate cell.
        output_gate (RNNCell): The output gate cell.
        forget_gate (RNNCell): The forget gate cell.
        cell_gate (RNNCell): The cell state update gate.

    The `initial_state` property returns the initial state of the LSTM
    cell, typically zeros for both the hidden state and the cell state.

    """

    input_gate: RNNCell
    output_gate: RNNCell
    forget_gate: RNNCell
    cell_gate: RNNCell

    def __call__(
        self,
        x: Array,
        state: tuple[Array, Array],
        *args,
        **kwargs,
    ) -> tuple[Array, tuple[Array, Array]]:
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
        """Returns the initial state of the LSTM cell, typically zeros
        for both the hidden state and the cell state.
        """
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
    """Constructs an LSTMCell with specified dimensions and initializers.

    Args:
        in_size (int): The size of the input dimension.
        state_size (int): The size of the state dimension.
        weight_init (Initializer): The initializer for the weight matrices. Defaults to kaiming_normal.
        bias_init (Initializer): The initializer for the bias vector. Defaults to normal.
        key (Array): A JAX random key used for initializing the gates' weights and biases.

    Returns:
        LSTMCell: An instance of LSTMCell initialized with the specified
        parameters, including input, output, forget, and cell gates.

    """
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
            (nn.sigmoid, nn.sigmoid, nn.sigmoid, nn.tanh),
            rnd.split(key, 4),
            strict=False,
        )
    )
    return LSTMCell(c1, c2, c3, c4)
