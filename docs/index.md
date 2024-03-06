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

