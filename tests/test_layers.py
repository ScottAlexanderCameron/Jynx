from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import jax
import jynx
import jynx.layers as nn
from jax import numpy as jnp
from jax import random as rnd
from jax import tree_util as tu

small_ints = st.integers(1, 64)
tiny_ints = st.integers(1, 3)
seeds = st.integers(0, 4096)
layer_sizes = st.lists(st.integers(1, 1024), min_size=3, max_size=10)

slow_settings = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)


@given(seeds, layer_sizes)
@slow_settings
def test_mlp_initial_forward_is_not_nan(seed, sizes):
    k1, k2 = rnd.split(rnd.PRNGKey(seed))
    net = nn.mlp(sizes, key=k1)

    x_shape = (32, sizes[0])

    zero = jnp.zeros(x_shape)
    assert not jnp.isnan(net(zero)).any()
    rand = rnd.normal(k2, x_shape)
    assert not jnp.isnan(net(rand)).any()


@given(
    seeds,
    layer_sizes,
    st.one_of([st.just(nn.rnn_cell), st.just(nn.gru_cell), st.just(nn.lstm_cell)]),
)
@slow_settings
def test_rnn_initial_forward_is_not_nan(seed, sizes, cell_type):
    key = jynx.key_seq(rnd.PRNGKey(seed))
    x_shape = (32, sizes[0])

    net = nn.Recurrent(
        cell_type(si, so, key=next(key)) for si, so in zip(sizes[:-1], sizes[1:])
    )
    state = net.initial_state

    zero = jnp.zeros(x_shape)
    assert not jnp.isnan(net(zero, state)[0]).any()
    rand = rnd.normal(next(key), x_shape)
    assert not jnp.isnan(net(rand, state)[0]).any()


def test_slice_module_list_preserves_type():
    key = jynx.key_seq(rnd.PRNGKey(0))
    net = nn.mlp([1, 32, 32, 1], key=next(key))

    assert isinstance(net, nn.Sequential)
    assert isinstance(net[::2], nn.Sequential)
    assert isinstance(net[1:-1], nn.Sequential)

    net = nn.parallel(
        nn.linear(1, 1, key=next(key)),
        jax.nn.relu,
        nn.linear(1, 1, key=next(key)),
        nn.conv(1, 1, (3, 3), key=next(key)),
    )

    assert isinstance(net, nn.Parallel)
    assert isinstance(net[::2], nn.Parallel)
    assert isinstance(net[1:-1], nn.Parallel)

    net = nn.recurrent(
        nn.lstm_cell(1, 1, key=next(key)),
        nn.gru_cell(1, 1, key=next(key)),
        jax.nn.relu,
        nn.rnn_cell(1, 1, key=next(key)),
    )

    assert isinstance(net, nn.Recurrent)
    assert isinstance(net[::2], nn.Recurrent)
    assert isinstance(net[1:-1], nn.Recurrent)


def test_rnn_state_is_none_for_stateless_layers():
    key = jynx.key_seq(rnd.PRNGKey(0))
    rnn = nn.recurrent(
        nn.conv(1, 8, (3, 3), key=next(key)),
        nn.Reshape((-1, 8)),
        nn.rnn_cell(8, 16, key=next(key)),
        jax.nn.relu,
        nn.gru_cell(16, 32, key=next(key)),
        nn.Dropout(0.1),
        nn.lstm_cell(32, 64, key=next(key)),
        nn.linear(64, 3 * 3, key=next(key)),
        nn.Reshape((-1, 1, 3, 3)),
    )

    state = rnn.initial_state
    x = jnp.zeros((32, 1, 3, 3))
    y = x
    for _ in range(3):
        y, state = rnn(y, state, key=next(key))

    assert all(state[i] is None for i in (0, 1, 3, 5, 7, 8))
    assert all(state[i] is not None for i in (2, 4, 6))
    assert y.shape == x.shape


def test_rnn_scan_eq_forward_in_loop():
    key = jynx.key_seq(rnd.PRNGKey(0))
    rnn = nn.recurrent(
        nn.linear(10, 32, key=next(key)),
        jax.nn.relu,
        nn.lstm_cell(32, 32, key=next(key)),
        jax.nn.relu,
        nn.gru_cell(32, 32, key=next(key)),
        jax.nn.relu,
        nn.linear(32, 10, key=next(key)),
    )

    states = [rnn.initial_state]
    xs = [rnd.normal(next(key), (10, 10))]

    for _ in range(5):
        x, s = rnn(xs[-1], states[-1])
        xs.append(x)
        states.append(s)

    xs = jnp.stack(xs)
    ys, last_state = rnn.scan(xs[:-1], states[0])

    assert jnp.allclose(xs[1:], ys)
    assert tu.tree_all(tu.tree_map(jnp.allclose, last_state, states[-1]))


@given(st.lists(small_ints, min_size=3, max_size=3), small_ints, tiny_ints, tiny_ints)
@slow_settings
def test_conv_shape_inverts_conv_transpose(in_shape, channels, kernel_size, stride):
    in_shape = (1,) + tuple(in_shape)
    key = jynx.key_seq(rnd.PRNGKey(0))
    conv = nn.conv(
        channels,
        in_shape[1],
        (kernel_size, kernel_size),
        (stride, stride),
        key=next(key),
    )
    deconv = nn.conv_transpose(
        in_shape[1],
        channels,
        (kernel_size, kernel_size),
        (stride, stride),
        key=next(key),
    )

    x = jnp.zeros(in_shape)
    assert x.shape == conv(deconv(x)).shape
