import typing as tp
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as tu
from jax import Array

from ..pytree import PyTree, static
from .linear import linear
from .module import Module
from .nets import mlp


class MixtureOfExperts[**P](PyTree):
    type Gating = Module[tp.Concatenate[Array, P], tuple[Array, Array]]
    type Experts = Module[tp.Concatenate[Array, Array, P], Array]

    gate: Gating
    stacked_experts: Experts

    def __call__(self, x: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        *batch_shape, features_in = x.shape
        x = x.reshape((-1, features_in))

        weights, selected_experts = self.gate(x, *args, **kwargs)
        experts_out = self.stacked_experts(selected_experts, x, *args, **kwargs)
        experts_out = jnp.sum(experts_out * weights[:, :, None], axis=1)

        return experts_out.reshape((*batch_shape, -1))


class TopKGating[**P](PyTree):
    gate: Module[tp.Concatenate[Array, P], Array]
    top_k_experts: int = static()

    def __call__(
        self,
        x: Array,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Array, Array]:
        logits = self.gate(x, *args, **kwargs)
        return jax.lax.top_k(
            jax.nn.softmax(logits),
            self.top_k_experts,
        )


class StackedExperts[**P](PyTree):
    stacked_experts: Module[tp.Concatenate[Array, P], Array]

    def _call_expert(self, idx, x, *args: P.args, **kwargs: P.kwargs):
        return tu.tree_map(lambda w: w[idx], self.stacked_experts)(x, *args, **kwargs)


class VMappedExperts[**P](StackedExperts[P]):
    def __call__(
        self,
        idx: Array,
        x: Array,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Array:
        def apply_experts(idx, xi):
            return jax.vmap(lambda i: self._call_expert(i, xi, *args, **kwargs))(idx)

        return jax.vmap(apply_experts)(idx, x)


class ScannedExperts[**P](StackedExperts[P]):
    def __call__(
        self,
        idx: Array,
        x: Array,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Array:
        def apply_experts(idx, xi):
            return jax.lax.map(lambda i: self._call_expert(i, xi, *args, **kwargs), idx)

        return jax.lax.map(lambda a: apply_experts(*a), (idx, x))


def sparse_moe(
    num_experts: int,
    in_size: int,
    out_size: int,
    hidden_size: int,
    *,
    top_k_experts: int = 2,
    activation: Callable[[Array], Array] = jax.nn.relu,
    vmap: bool = True,
    key: Array,
) -> MixtureOfExperts:
    k1, k2 = rnd.split(key)
    return MixtureOfExperts(
        TopKGating(
            linear(in_size, num_experts, key=k1),
            top_k_experts=top_k_experts,
        ),
        (VMappedExperts if vmap else ScannedExperts)(
            jax.vmap(
                lambda k: mlp(
                    [in_size, hidden_size, out_size],
                    activation=activation,
                    key=k,
                ),
            )(rnd.split(k2, num_experts)),
        ),
    )
