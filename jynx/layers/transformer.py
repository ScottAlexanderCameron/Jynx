import typing as tp

import jax.nn.initializers as init
from jax import Array, nn
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer

from ..pytree import PyTree, static
from .containers import Sequential
from .linear import Linear
from .misc import Norm, layer_norm
from .module import Key
from .nets import mlp
from .static import Dropout


class Attention(PyTree):
    proj_q: Linear
    proj_k: Linear
    proj_v: Linear
    proj_o: Linear

    def __call__(
        self,
        q: Array,  # shape: (..., Tq, qdim)
        k: tp.Optional[Array] = None,  # shape: (..., Tk, kdim)
        v: tp.Optional[Array] = None,  # shape: (..., Tk, vdim)
        *,
        mask: tp.Optional[Array] = None,  # shape: (..., Tk) or (..., Tq, Tk)
        attention_distances: tp.Union[bool, Array] = False,
        # return_weights: bool = False,
        **kwargs,
    ) -> Array:
        del kwargs
        if k is None:
            k = q
        if v is None:
            v = k

        q = self.proj_q(jnp.expand_dims(q, -3))  # shape: (..., n_heads, Tq, d_head)
        k = self.proj_k(jnp.expand_dims(k, -3))  # shape: (..., n_heads, Tk, d_head)
        v = self.proj_v(jnp.expand_dims(v, -3))  # shape: (..., n_heads, Tk, d_head)

        d_k = k.shape[-1]
        logits = q @ k.swapaxes(-1, -2) / jnp.sqrt(d_k)  # shape: (..., n_head, Tq, Tk)

        if attention_distances is True:
            attention_distances = jnp.abs(
                jnp.arange(logits.shape[-2])[:, None] - jnp.arange(logits.shape[-1])
            )

        if attention_distances is not False:
            logits = logits - attention_distances

        if mask is not None:
            logits = jnp.where(mask, logits, -jnp.inf)

        weights = nn.softmax(logits, axis=-1)
        atten = weights @ v  # shape: (..., n_head, Tq, d_head)
        atten = atten.swapaxes(-2, -3)  # shape: (..., Tq, n_head, d_head)
        atten = atten.reshape(atten.shape[:-2] + (-1,))  # shape: (..., Tq, embed)
        out = self.proj_o(atten)

        # if return_weights:
        #     return out, weights
        # else:
        return out

    @classmethod
    def causal_mask(cls, seq_len: int) -> Array:
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


def attention(
    embed_dim: int,
    num_heads: int,
    *,
    qdim: tp.Optional[int] = None,
    kdim: tp.Optional[int] = None,
    vdim: tp.Optional[int] = None,
    out_dim: tp.Optional[int] = None,
    weight_init: Initializer = init.xavier_normal(),
    bias_init: Initializer = init.normal(),
    key: Array,
) -> Attention:
    """
    Constructs an Attention layer with the specified parameters.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.

    Keyword Args:
        qdim (int, optional): The dimension of the query vectors. Defaults to None.
        kdim (int, optional): The dimension of the key vectors. Defaults to None.
        vdim (int, optional): The dimension of the value vectors. Defaults to None.
        out_dim (int, optional): The dimension of the output vectors. Defaults to None.
        weight_init (Initializer, optional): The weight initializer for the linear
            layers. Defaults to init.xavier_normal().
        bias_init (Initializer, optional): The bias initializer for the linear layers.
            Defaults to init.normal().
        key (Array): The random key array for splitting.

    Returns:
        Attention: An instance of the Attention layer.

    Raises:
        AssertionError: If embed_dim is not divisible by num_heads.

    """
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num heads"

    dhead = embed_dim // num_heads
    qdim = qdim or embed_dim
    kdim = kdim or qdim
    vdim = vdim or kdim
    out_dim = out_dim or embed_dim
    bias_init = bias_init or tp.cast(Initializer, (lambda *_: None))
    wk1, wk2, wk3, wk4, bk1, bk2, bk3, bk4 = rnd.split(key, 8)

    return Attention(
        Linear(
            weight_init(wk1, (num_heads, qdim, dhead)),
            bias_init(bk1, (num_heads, 1, dhead)),
        ),
        Linear(
            weight_init(wk2, (num_heads, kdim, dhead)),
            bias_init(bk2, (num_heads, 1, dhead)),
        ),
        Linear(
            weight_init(wk3, (num_heads, vdim, dhead)),
            bias_init(bk3, (num_heads, 1, dhead)),
        ),
        Linear(
            weight_init(wk4, (embed_dim, out_dim)),
            bias_init(bk4, (num_heads, 1, dhead)),
        ),
    )


class TransformerEncoderBlock(PyTree):
    norm1: Norm
    norm2: Norm
    self_attention: Attention
    mlp: Sequential
    dropout: Dropout
    norm_first: bool = static(default=True)

    def __call__(
        self,
        x: Array,
        *,
        mask: tp.Optional[Array] = None,
        self_attention_distances: tp.Union[bool, Array] = False,
        key: Key = None,
    ) -> Array:
        if key is not None:
            k1, k2, k3 = rnd.split(key, 3)
        else:
            k1, k2, k3 = None, None, None

        if self.norm_first:
            x = self.norm1(x)

        x = x + self.dropout(
            self.self_attention(
                x, mask=mask, attention_distances=self_attention_distances
            ),
            key=k1,
        )
        x = self.norm2(x)
        x = x + self.dropout(self.mlp(x, key=k2), key=k3)

        if not self.norm_first:
            x = self.norm1(x)

        return x


def transformer_encoder_block(
    embed_dim: int,
    num_heads: int,
    *,
    dropout_prob: float = 0,
    activation: tp.Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    key: Array,
) -> TransformerEncoderBlock:
    k1, k2 = rnd.split(key)
    return TransformerEncoderBlock(
        layer_norm((embed_dim,)),
        layer_norm((embed_dim,)),
        attention(embed_dim, num_heads, key=k1),
        mlp(
            [embed_dim, ff_hidden_size_factor * embed_dim, embed_dim],
            activation=activation,
            key=k2,
        ),
        Dropout(dropout_prob),
        norm_first=norm_first,
    )


def transformer_encoder(
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    *,
    dropout_prob: float = 0,
    activation: tp.Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    key: Array,
) -> Sequential:
    return Sequential(
        [
            transformer_encoder_block(
                embed_dim,
                num_heads,
                dropout_prob=dropout_prob,
                activation=activation,
                ff_hidden_size_factor=ff_hidden_size_factor,
                norm_first=norm_first,
                key=k,
            )
            for k in rnd.split(key, num_layers)
        ]
    )


class TransformerDecoderBlock(PyTree):
    norm1: Norm
    norm2: Norm
    norm3: Norm
    self_attention: Attention
    cross_attention: Attention
    mlp: Sequential
    dropout: Dropout
    norm_first: bool = static(default=True)

    def __call__(
        self,
        x: Array,
        context: Array,
        *,
        mask: tp.Optional[Array] = None,
        self_attention_distances: bool | Array = False,
        context_mask: tp.Optional[Array] = None,
        cross_attention_distances: bool | Array = False,
        key: Key = None,
    ) -> Array:
        if key is not None:
            k1, k2, k3, k4 = rnd.split(key, 4)
        else:
            k1, k2, k3, k4 = None, None, None, None

        if self.norm_first:
            x = self.norm1(x)

        x = x + self.dropout(
            self.self_attention(
                x,
                mask=mask,
                attention_distances=self_attention_distances,
            ),
            key=k1,
        )
        x = self.norm2(x)
        x = x + self.dropout(
            self.cross_attention(
                x,
                context,
                mask=context_mask,
                attention_distances=cross_attention_distances,
            ),
            key=k2,
        )
        x = self.norm3(x)
        x = x + self.dropout(self.mlp(x, key=k3), key=k4)

        if not self.norm_first:
            x = self.norm1(x)

        return x


def transformer_decoder_block(
    embed_dim: int,
    num_heads: int,
    *,
    dropout_prob: float = 0,
    activation: tp.Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    key: Array,
) -> TransformerDecoderBlock:
    k1, k2, k3 = rnd.split(key, 3)
    return TransformerDecoderBlock(
        layer_norm((embed_dim,)),
        layer_norm((embed_dim,)),
        layer_norm((embed_dim,)),
        attention(embed_dim, num_heads, key=k1),
        attention(embed_dim, num_heads, key=k2),
        mlp(
            [embed_dim, ff_hidden_size_factor * embed_dim, embed_dim],
            activation=activation,
            key=k3,
        ),
        Dropout(dropout_prob),
        norm_first=norm_first,
    )


def transformer_decoder(
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    *,
    dropout_prob: float = 0,
    activation: tp.Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    key: Array,
) -> Sequential:
    return Sequential(
        [
            transformer_decoder_block(
                embed_dim,
                num_heads,
                dropout_prob=dropout_prob,
                activation=activation,
                ff_hidden_size_factor=ff_hidden_size_factor,
                norm_first=norm_first,
                key=k,
            )
            for k in rnd.split(key, num_layers)
        ]
    )