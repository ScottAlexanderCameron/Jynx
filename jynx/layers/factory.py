from functools import partial
from inspect import signature

import jax
import jax.random as rnd
from jax import Array

from .. import layers as nn


def namespace_entry(name: str, ns: tuple):
    for n in ns:
        if hasattr(n, name):
            return getattr(n, name)

    return None


def create_from_dict(dct: dict, *, key: Array, ns: tuple = (nn, jax.nn)):
    assert len(dct) == 1, "Only one module can be contructed at a time"

    [(name, args)] = dct.items()

    def cons_arg(item, key):
        if isinstance(item, (list, tuple)):
            return type(item)(map(cons_arg, item, rnd.split(key, len(item))))
        elif isinstance(item, dict):
            return create_from_dict(item, key=key, ns=ns)
        elif isinstance(item, str):
            return namespace_entry(item, ns) or item
        else:
            return item

    mod = namespace_entry(name, ns)
    assert mod is not None

    if "key" in signature(mod).parameters:
        key, k = rnd.split(key)
        mod = partial(mod, key=k)

    if isinstance(args, dict):
        ks = rnd.split(key, len(args) + 1)
        return mod(
            **{n: cons_arg(v, k) for (n, v), k in zip(args.items(), ks)},
        )
    elif isinstance(args, (list, tuple)):
        ks = rnd.split(key, len(args) + 1)
        return mod(
            *(cons_arg(v, k) for v, k in zip(args, ks)),
        )
    else:
        return mod(args)


if __name__ == "__main__":
    from pprint import pp

    mods = {
        "recurrent": [
            {"lstm_cell": (1, 12)},
            {"Dropout": 0.1},
            {"lstm_cell": (12, 12)},
            "sigmoid",
            {
                "parallel": [
                    {"linear": (12, 1)},
                    {"linear": (12, 1)},
                    {"linear": (12, 1)},
                ]
            },
        ]
    }

    mod = create_from_dict(mods, key=jax.random.PRNGKey(0))
    out = mod.scan(jax.numpy.zeros((5, 2, 1)), mod.initial_state)[0]
    print([o.shape for o in out])
    pp(jax.tree_util.tree_map(jax.numpy.shape, mod))
