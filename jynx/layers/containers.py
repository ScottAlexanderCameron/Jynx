import typing as tp
from dataclasses import MISSING, dataclass, fields

import jax.tree_util as tu
from jax import Array
from jax import numpy as jnp
from jax import random as rnd

from ..pytree import dataclass_flatten, static
from .module import Key, Module, RecurrentModule
from .static import Fn


class ModuleList[M: Module](list[M]):
    def __init_subclass__(cls):
        return tu.register_pytree_with_keys_class(
            dataclass(cls, frozen=True, init=False)  # type: ignore
        )

    def __init__(self, itr: tp.Iterable[M], **kwargs):
        super().__init__(itr)
        for f in fields(type(self)):  # type: ignore
            if f.name in kwargs:
                val = kwargs[f.name]
            elif f.default is not MISSING:
                val = f.default
            elif f.default_factory is not MISSING:
                val = f.default_factory()
            else:
                raise ValueError(f"Required parameter: {f.name} was not specified")
            object.__setattr__(self, f.name, val)

    def tree_flatten(self):
        (ch,), aux = dataclass_flatten(self)
        return (ch, tuple(self)), aux

    def tree_flatten_with_keys(self):
        (ch, itr), aux = self.tree_flatten()
        return (
            [(tu.GetAttrKey(k), v) for k, v in ch.items()]
            + [(tu.SequenceKey(i), v) for i, v in enumerate(itr)],
            aux,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        ch, itr = children
        return cls(itr, **ch, **aux)

    def split_keys(self, kwargs: dict[str, tp.Any]) -> tp.Sequence[dict[str, Key]]:
        if "key" in kwargs:
            return [{"key": k} for k in rnd.split(kwargs.pop("key"), len(self))]
        else:
            return [{}] * len(self)

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self)})"

    @tp.overload
    def __getitem__(self, idx: tp.SupportsIndex) -> M:
        ...

    @tp.overload
    def __getitem__(self, idx: slice) -> tp.Self:
        ...

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if isinstance(idx, slice):
            (ch,), aux = dataclass_flatten(self)
            return type(self)(item, **ch, **aux)
        else:
            return item


class Sequential[T, **P](ModuleList[Module[tp.Concatenate[T, P], T]]):
    def __call__(self, x: T, *args: P.args, **kwargs: P.kwargs) -> T:
        for layer, key in zip(self, self.split_keys(kwargs)):
            x = layer(x, *args, **kwargs, **key)

        return x


class Parallel[**T, U](ModuleList[Module[T, U]]):
    def __call__(self, *args: T.args, **kwargs: T.kwargs) -> tp.Sequence[U]:
        outputs = []
        for layer, key in zip(self, self.split_keys(kwargs)):
            outputs.append(layer(*args, **kwargs, **key))
        return outputs


type _RecLayer[T, S, **P] = (
    Module[tp.Concatenate[T, P], T] | RecurrentModule[T, S, T, P]
)


class Recurrent[T, S, **P](ModuleList[_RecLayer[T, S, P]]):
    type States = tp.Sequence[tp.Optional[S]]

    def __call__(
        self, x: T, state: States, *args: P.args, **kwargs: P.kwargs
    ) -> tp.Tuple[T, States]:
        from itertools import zip_longest

        new_states = []

        for layer, st, key in zip_longest(self, state, self.split_keys(kwargs)):
            if st is None:
                layer = tp.cast(tp.Callable[tp.Concatenate[T, P], T], layer)
                x = layer(x, *args, **kwargs, **key)
            else:
                layer = tp.cast(RecurrentModule, layer)
                x, st = layer(x, st, *args, **kwargs, **key)
            new_states.append(st)

        return x, new_states

    @property
    def initial_state(self):
        return [
            getattr(layer, "initial_state") if hasattr(layer, "initial_state") else None
            for layer in self
        ]

    def scan(
        self,
        xs: tp.Sequence[T],
        states: States,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tp.Tuple[T, States]:
        from jax import eval_shape, lax
        from jax.tree_util import tree_map

        _, state_shapes = eval_shape(self, xs[0], states, *args, **kwargs)
        states = tree_map(
            lambda s, shape_dtype: jnp.broadcast_to(s, shape_dtype.shape),
            states,
            state_shapes,
        )
        key = tp.cast(Array, kwargs.pop("key")) if "key" in kwargs else None

        def forward(carry, x):
            s, k = carry
            if k is not None:
                k1, k2 = rnd.split(k)
                kwds = {**kwargs, "key": k2}
            else:
                k1, k2 = None, None
                kwds = kwargs
            y, s = self(x, s, *args, **kwds)
            return (s, k1), y

        carry, y = lax.scan(forward, (states, key), xs)
        return y, carry[0]


class DenselyConnected[**P](ModuleList[Module[tp.Concatenate[Array, P], Array]]):
    concat_axis: int = static(default=-1)
    only_inputs: bool = static(default=False)

    def __call__(self, x: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        inp = x
        for layer, key in zip(self, self.split_keys(kwargs)):
            if not self.only_inputs:
                inp = x
            x = layer(x, *args, **kwargs, **key)
            x = jnp.concatenate((x, inp), axis=self.concat_axis)
        return x


def wrap_if_function(arg):
    import jax.tree_util as tu

    treedef = tu.tree_structure(arg)
    isleaf = tu.treedef_is_leaf(treedef)
    if isleaf and treedef.num_leaves != 0:
        assert callable(arg), "layers must be callable"
        return Fn(arg)
    else:
        return arg


def sequential(*layers):
    return Sequential(map(wrap_if_function, layers))


def parallel(*layers):
    return Parallel(map(wrap_if_function, layers))


def recurrent(*layers):
    return Recurrent(map(wrap_if_function, layers))


def densely_connected(*layers):
    return DenselyConnected(map(wrap_if_function, layers))
