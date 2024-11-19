from functools import partial

import jax
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from jax import numpy as jnp
from jax import random as rnd
from jax import tree_util as tu

import jynx
import jynx.layers as nn

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
        cell_type(si, so, key=next(key))
        for si, so in zip(sizes[:-1], sizes[1:], strict=False)
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
        states.append(s)  # type: ignore

    xs = jnp.stack(xs)
    ys, last_state = rnn.scan(xs[:-1], states[0])  # type: ignore

    assert jnp.allclose(xs[1:], ys)  # type: ignore
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


@given(small_ints, small_ints)
@slow_settings
def test_attn_permutation_invariant(dhead, num_heads):
    # NB does not apply to sliding window
    key = partial(next, jynx.key_seq(rnd.PRNGKey(0)))
    d = dhead * num_heads
    attn = nn.attention(d, num_heads, key=key())
    q = rnd.normal(key(), (3, d))
    k = rnd.normal(key(), (10, d))
    p = rnd.permutation(key(), 10)
    x1 = attn(q, k)
    x2 = attn(q, k[p])

    assert jnp.allclose(x1, x2, rtol=1e-3, atol=1e-5)


@given(st.integers(16, 128), st.integers(32, 64), seeds, st.booleans())
@slow_settings
def test_sliced_attn_eq_full_attn(seq_len, dim, seed, use_mask):
    key = partial(next, jynx.key_seq(rnd.PRNGKey(seed)))
    q = rnd.normal(key(), (seq_len, dim))
    k = rnd.normal(key(), (seq_len, dim))
    v = rnd.normal(key(), (seq_len, dim))
    if use_mask:
        mask = rnd.bernoulli(key(), 0.1, (seq_len, seq_len))
        mask = mask.at[jnp.diag_indices_from(mask)].set(True)
    else:
        mask = None

    o1 = nn.scaled_dot_product_attention(q, k, v, mask)
    o2 = nn.sliced_attention(q, k, v, mask)

    assert jnp.all(jnp.isfinite(o1))
    assert jnp.all(jnp.isfinite(o2))

    assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-5)


def assert_tree_map_with_paths_preserves_order(obj):
    obj_with_names = tu.tree_map_with_path(lambda k, _: tu.keystr(k), obj)
    for k, name in tu.tree_leaves_with_path(obj_with_names):
        assert tu.keystr(k) == name


@given(layer_sizes)
@slow_settings
def test_mlp_tree_map_preserves_order(sizes):
    net = nn.mlp(sizes, key=rnd.PRNGKey(0))
    assert_tree_map_with_paths_preserves_order(net)


@given(
    layer_sizes,
    st.one_of([st.just(nn.rnn_cell), st.just(nn.gru_cell), st.just(nn.lstm_cell)]),
)
@slow_settings
def test_rnn_tree_map_preserves_order(sizes, cell_type):
    key = jynx.key_seq(rnd.PRNGKey(0))
    net = nn.Recurrent(
        cell_type(si, so, key=next(key))
        for si, so in zip(sizes[:-1], sizes[1:], strict=False)
    )
    assert_tree_map_with_paths_preserves_order(net)


@given(st.lists(small_ints, min_size=3, max_size=3), small_ints, tiny_ints, tiny_ints)
@slow_settings
def test_conv_tree_map_preserves_order(in_shape, channels, kernel_size, stride):
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
    assert_tree_map_with_paths_preserves_order((conv, deconv))


@given(small_ints, small_ints, tiny_ints)
@slow_settings
def test_transformer_tree_map_preserves_order(dhead, num_heads, layers):
    key = partial(next, jynx.key_seq(rnd.PRNGKey(0)))
    d = dhead * num_heads
    enc = nn.transformer_encoder(layers, d, num_heads, key=key())
    dec = nn.transformer_decoder(layers, d, num_heads, key=key())

    assert_tree_map_with_paths_preserves_order((enc, dec))


@given(tiny_ints, seeds)
@slow_settings
def test_unet_initial_forward_is_not_nan(depth, key):
    key = partial(next, jynx.key_seq(rnd.PRNGKey(key)))
    unet = nn.unet(depth, 1, 1, 16, key=key())
    x = rnd.normal(key(), (1, 1, 32, 32))

    assert jnp.all(jnp.isfinite(unet(x)))


@given(seeds)
def test_norm_output_is_normalized(seed):
    norm = nn.norm((), axis=-1, use_weight=False, use_bias=False)
    x = rnd.normal(rnd.PRNGKey(seed), (16, 16))
    y = norm(x)
    assert jnp.allclose(jnp.std(y, axis=-1), jnp.ones(y.shape[:-1]))
