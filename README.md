# Jynx

A straight forward neural network library written in jax. No hidden mechanisms, no black
magic. Requires only jax and optax.

This library provides 3 components: (1) standard neural network layers, (2) a
`fit` function to train models, and (3) a collection of basic callbacks for
checkpointing and logging. The `fit` function doesn't know anything about the
way layers are implemented, and can be used with other frameworks if desired. It
only relies on optax.

## TLDR;

```python
import jax
import jax.numpy as jnp
import jax.random as rnd
import optax

import jynx
import jynx.callbacks as cb
import jynx.layers as nn

from jax import Array


def data_iter(key: Array):
    from itertools import repeat
    x = jnp.linspace(-1, 1, 100).reshape(-1, 1)
    y = jnp.cos(x) + 0.05 * rnd.normal(key, x.shape)
    return repeat((x, y))


def loss_fn(
    net: nn.Module[[Array], Array],
    batch: tuple[Array, Array],  # any type you want
    key: Array
) -> Array:
    x, y = batch
    y_pred = net(x, key=key)
    return optax.l2_loss(y_pred, y).mean()


def make_model(key: Array) -> nn.Module[[Array], Array]:
    k1, k2, k3 = rnd.split(key, 3)
    net = nn.sequential(
        nn.linear(1, 32, key=k1),
        jax.nn.relu,
        nn.linear(32, 32, key=k2),
        jax.nn.relu,
        nn.linear(32, 1, key=k3),
    )
    # or use:
    # net = nn.mlp(
    #     [1, 32, 32, 1],
    #     activation=jax.nn.silu,
    #     key=key,
    # )
    return net


k1, k2, k3 = rnd.split(rnd.PRNGKey(0), 3)
res = jynx.fit(
    make_model(k1),
    loss_fn=loss_fn,
    data_iter=data_iter(k2),
    optimizer=optax.adam(1e-3),
    key=k3,
    callbacks=[
        cb.ConsoleLogger(metrics=["loss"]),
        cb.TensorBoardLogger("tboard"),  # requires tensorboardx
        cb.MlflowLogger(),  # requires mlflow
        cb.CheckPoint("latest.pk"),  # requires cloudpickle
        cb.EarlyStopping(
            monitor="loss",
            steps_without_improvement=500,
        ),
    ],
)
print("final loss", res.logs["loss"])
net = res.params
```

## Layers

Currently implemented modules:

- `Sequential`
- `Parallel`
- `Recurrent`: like `Sequential` but passes state, used for RNNs
- `DenselyConnected`: DenseNet
- `Linear`
- `Conv` and `ConvTranspose`
- `Embedding`
- `Fn`: activation function
- `StarFn`: equivalent of `fn(*x)`
- `Static`: wraps an object to be ignored by jax
- `Reshape`
- `Dropout`
- Pooling layers: `AvgPoolng`, `MaxPooling`, `MinPooling`
- `Norm`: layer norm, batch norm etc.
- `SkipConnection`
- RNN layers: `RNNCell`, `GRUCell`, `LSTMCell`
- `Attention`, `TransformerEncoderBlock`, `TransformerDecoderBlock`

Constructors for full networks:

- `mlp`
- `transformer_encoder`
- `transformer_decoder`
- `rnn`, `lstm`, and `gru`
- More to come...

### How layers work

Layers are simple pytree containers with a `__call__` method. To define new
modules easily, we provide a base `PyTree` class. Using this is not at all a
requirement, it just makes most definitions simpler. Layers that
don't require any static data can just as easily be defined as `NamedTuple`s.
```python
class MyLinear(NamedTuple):
    weight: Array
    bias: Array

    def __call__(self, x, *, key=None):
        return x @ self.weight + self.bias
```
We provide initialization with factory functions instead of in the
contructor. This makes flattening and unflattening pytrees much simpler. For
example:
```python
def my_linear(size_in, size_out, *, key):
    w_init = jax.nn.initializers.kaiming_normal()
    return MyLinear(
        w_init(key, (size_in, size_out)),
        jnp.zeros((size_out,)),
    )
```

So layers are just 'dumb' containers.
The `PyTree` base class converts the inheriting class to a dataclass and
registers the type as a `jax` pytree

```python
class MyDense(PyTree):
    weight: Array
    bias: Array
    activation: Callable[[Array], Array] = static(default=jax.nn.relu)

    def __call__(self, x, *, key=None):
        return self.activation(x @ self.weight + self.bias)
```

## The `fit` function
TODO
## Callbacks
TODO
