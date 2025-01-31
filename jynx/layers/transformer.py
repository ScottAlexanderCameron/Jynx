import typing as tp
from collections.abc import Callable

import jax
import jax.nn.initializers as init
from jax import Array, nn
from jax import numpy as jnp
from jax import random as rnd
from jax.nn.initializers import Initializer

from ..pytree import PyTree, static
from .containers import Sequential
from .linear import Linear
from .misc import Norm, layer_norm
from .nets import mlp
from .static import Dropout


class AttentionFn(tp.Protocol):
    """A protocol for attention functions used in the Attention module.

    Args:
        q: Array, the query tensor with shape (..., Tq, qdim).
        k: Array, the key tensor with shape (..., Tk, kdim).
        v: Array, the value tensor with shape (..., Tk, vdim).
        mask: Optional[Array], the mask tensor with shape (..., Tk) or
            (..., Tq, Tk), used to mask out certain positions.

    Returns:
        Array: The result of the attention mechanism, usually with shape (..., Tq, vdim).

    Example:
        ```python
        # Define a simple attention function
        def simple_attention(q, k, v, mask=None):
            scores = jnp.matmul(q, k.transpose(-2, -1))
            if mask is not None:
                scores = scores * mask
            p_attn = nn.softmax(scores)
            return jnp.matmul(p_attn, v)
        # Create query, key, value arrays
        q = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
        k = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
        v = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        # Apply simple attention
        simple_attention(q, k, v)
        ```

    """

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        ...


def scaled_dot_product_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: Array | None = None,
    **kwargs,
) -> Array:
    """Compute the scaled dot-product attention.

    Args:
        q: Array, the queries with shape (..., Tq, qdim).
        k: Array, the keys with shape (..., Tk, kdim).
        v: Array, the values with shape (..., Tk, vdim).
        mask: Optional[Array], the mask tensor with shape (..., Tq, Tk).

    Returns:
        Array: The result of the attention mechanism with shape (..., Tq, vdim).

    Example:
        ```python
        q = jnp.array([[[1.0, 2.0]]])  # shape: (1, 1, 2)
        k = jnp.array([[[3.0, 4.0], [1.0, 0.0]]])  # shape: (1, 2, 2)
        v = jnp.array([[[5.0, 6.0], [7.0, 8.0]]])  # shape: (1, 2, 2)
        o = scaled_dot_product_attention(q, k, v)  # shape: (1, 1, 2)
        ```

    """
    del kwargs
    d = q.shape[-1]
    logits = jnp.einsum("...qd,...kd->...qk", q / jnp.sqrt(d), k)
    weights = nn.softmax(logits, axis=-1, where=mask)
    return jnp.einsum("...qk,...kv->...qv", weights, v)


def sliced_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: Array | None = None,
    **kwargs,
) -> Array:
    """A memory efficient version of attention implemented by iterating over
    the queries and values. This is mathematically equivalent to
    `scaled_dot_product_attention`, although may produce slightly different
    results due to floating point error.

    Args:
        q: Array, the queries with shape (..., Tq, qdim).
        k: Array, the keys with shape (..., Tk, kdim).
        v: Array, the values with shape (..., Tk, vdim).
        mask: Optional[Array], the mask tensor with shape (..., Tq, Tk).
        chunk_size (int): how many key and value pairs to include in the
            attention calculation per iteration, default 1.
        logits_base (ArrayLike): numerical stability factor subtracted from
            the logits before exponentiating. The calculation is mathematically
            equivalent for any value, but setting it specifically may result
            in better or worse numerical accuracy. default log(Tk).

    Returns:
        Array: The result of the attention mechanism with shape (..., Tq, vdim).

    Example:
        ```python
        q = jnp.array([[[1.0, 2.0]]])  # shape: (1, 1, 2)
        k = jnp.ones((1, 6, 2))
        v = jnp.ones((1, 6, 2))
        o = sliced_attention(q, k, v, chunk_size=3)  # shape: (1, 1, 2)
        ```

    """
    base = kwargs.get("logits_base", jnp.log(k.shape[-2]))
    chunk_size = kwargs.get("chunk_size", 1)
    del kwargs
    q /= jnp.sqrt(q.shape[-1])

    def chunk(x):
        x = x.reshape(
            *x.shape[:-2],
            x.shape[-2] // chunk_size,
            chunk_size,
            x.shape[-1],
        )

        return jnp.moveaxis(x, -3, 0)

    k, v = chunk(k), chunk(v)

    if mask is not None:
        mask = mask.reshape(*mask.shape[:-1], mask.shape[-1] // chunk_size, chunk_size)
        mask = jnp.moveaxis(mask, -2, 0)

    def loop(carry, kvm):
        k, v, m = kvm
        w, out = carry
        logits = jnp.einsum("...qd,...kd->...qk", q, k)
        wi = jnp.exp(logits - base)
        if m is not None:
            wi = jnp.where(m, wi, 0)
        out += jnp.einsum("...qk,...kv->...qv", wi, v)
        w += wi.sum(axis=-1, keepdims=True)
        return (w, out), None

    (sum_w, out), _ = jax.lax.scan(
        loop,
        (jnp.zeros((*q.shape[:-1], 1)), jnp.zeros((*q.shape[:-1], v.shape[-1]))),
        (k, v, mask),
    )
    return out / sum_w


def sliding_window_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: Array | None = None,
    **kwargs,
) -> Array:
    del mask
    window_left = kwargs.get("window_left", 0)
    window_right = kwargs.get("window_right", 0)
    window_width = window_left + window_right + 1
    assert window_width > 1, "a valid window must be provided"
    N = q.shape[-2]
    idx = jnp.arange(N)

    def chunk(x):
        pad = [(0, 0)] * len(x.shape)
        pad[-2] = (window_left, window_right)
        x = jnp.pad(x, pad)
        x = x[..., idx : idx + window_width, :]
        return x

    k = chunk(k)
    v = chunk(v)
    logits = jnp.einsum("...qd,...qkd->...qk", q / jnp.sqrt(q.shape[-1]), k)
    i, j = jnp.indices(logits.shape[-2:])
    m = jnp.logical_and(i + j >= window_left, i + j <= N - window_left)
    m = jnp.broadcast_to(m, logits.shape)
    weights = jax.nn.softmax(logits, axis=-1, where=m)
    return jnp.einsum("...qk,...qkv->...qv", weights, v)


class Attention(PyTree):
    """A module implementing the multi-head attention mechanism.

    Attributes:
        proj_q: Linear, linear projection layer for queries.
        proj_k: Linear, linear projection layer for keys.
        proj_v: Linear, linear projection layer for values.
        proj_o: Linear, final linear projection layer for output.
        attn_fn: AttentionFn, the attention function to compute attention scores.

    Methods:
        __call__: Apply the attention mechanism to inputs.

    Example:
        ```python
        # Initialize an Attention instance with specific dimensions
        attention_layer = Attention(
            proj_q=linear(input_dim, output_dim, key=k1),
            proj_k=linear(input_dim, output_dim, key=k2),
            proj_v=linear(input_dim, output_dim, key=k3),
            proj_o=linear(output_dim, final_output_dim, key=k4),
        )

        # Generate random query, key, value tensors
        q = jnp.array([[[1.0, 2.0]]])  # Shape: (batch_size, seq_len, dim)
        k = jnp.array([[[3.0, 4.0], [1.0, 0.0]]])  # Shape: (batch_size, seq_len, dim)
        v = jnp.array([[[5.0, 6.0], [7.0, 8.0]]])  # Shape: (batch_size, seq_len, dim)

        # Apply attention
        output = attention_layer(q, k, v)
        ```

    """

    proj_q: Linear
    proj_k: Linear
    proj_v: Linear
    proj_o: Linear
    attn_fn: AttentionFn = static(default=scaled_dot_product_attention)

    def __call__(
        self,
        q: Array,  # shape: (..., Tq, qdim)
        k: Array | None = None,  # shape: (..., Tk, kdim)
        v: Array | None = None,  # shape: (..., Tk, vdim)
        *,
        mask: Array | None = None,  # shape: (..., Tk) or (..., Tq, Tk)
        **kwargs,
    ) -> Array:
        """Apply the attention mechanism to the input tensors.

        Args:
            q: Array, the query tensor with shape (..., Tq, qdim).
            k: Optional[Array], the key tensor with shape (..., Tk,
                kdim). If None, uses `q` as keys.
            v: Optional[Array], the value tensor with shape (..., Tk,
                vdim). If None, uses `k` as values.
            mask: Optional[Array], the mask tensor with shape (..., Tk)
                or (..., Tq, Tk), used to mask out certain positions.

        Returns:
            Array: The output tensor after applying attention and linear
                projections, with shape (..., Tq, final_output_dim).

        """
        if k is None:
            k = q
        if v is None:
            v = k

        # shape: (..., n_heads, Tq, d_head)
        q = self.proj_q(jnp.expand_dims(q, -3))
        # shape: (..., n_heads, Tk, d_head)
        k = self.proj_k(jnp.expand_dims(k, -3))
        # shape: (..., n_heads, Tk, d_head)
        v = self.proj_v(jnp.expand_dims(v, -3))

        atten = self.attn_fn(q, k, v, mask, **kwargs)
        atten = atten.swapaxes(-2, -3)  # shape: (..., Tq, n_head, d_head)
        # shape: (..., Tq, embed)
        atten = atten.reshape((*atten.shape[:-2], -1))
        out = self.proj_o(atten)

        return out

    @classmethod
    def causal_mask(cls, seq_len: int) -> Array:
        """Generate a causal mask for the self-attention mechanism.

        Args:
            seq_len: int, the sequence length.

        Returns:
            Array: A lower triangular matrix of shape (seq_len, seq_len)
                used as a causal mask.

        Example:
            ```python
            # Generate a causal mask for a sequence of length 5
            mask = Attention.causal_mask(5)
            ```

        """
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

    @classmethod
    def block_dense_mask(
        cls,
        key_block_sizes: tp.Sequence[int],
        query_block_sizes: tp.Sequence[int] | None = None,
    ) -> Array:
        if query_block_sizes is None:
            query_block_sizes = key_block_sizes

        kb_ids = jnp.arange(len(key_block_sizes))
        kb_ids = jnp.repeat(kb_ids, jnp.asarray(key_block_sizes))

        qb_ids = jnp.arange(len(query_block_sizes))
        qb_ids = jnp.repeat(qb_ids, jnp.asarray(query_block_sizes))
        return qb_ids[:, None] == kb_ids

    @classmethod
    def block_causal_mask(cls, block_sizes: tp.Sequence[int]) -> Array:
        return jnp.tril(cls.block_dense_mask(block_sizes))


def attention(
    embed_dim: int,
    num_heads: int,
    *,
    qdim: int | None = None,
    kdim: int | None = None,
    vdim: int | None = None,
    out_dim: int | None = None,
    weight_init: Initializer = init.xavier_normal(),
    bias_init: Initializer = init.normal(),
    attention_fn: AttentionFn = scaled_dot_product_attention,
    key: Array,
) -> Attention:
    """Constructs an Attention layer with the specified parameters.

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
        key (Array): The random key array for initialization.

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
            bias_init(bk4, (out_dim,)),
        ),
        attention_fn,
    )


class TransformerEncoderBlock(PyTree):
    """A Transformer Encoder Block that applies self-attention, followed by a feed-forward network.

    Attributes:
        norm1: Norm, the first normalization layer applied before
            self-attention or after depending on `norm_first`.
        norm2: Norm, the second normalization layer applied before the
            feed-forward network or after depending on `norm_first`.
        self_attention: Attention, the self-attention mechanism within
            the encoder block.
        mlp: Sequential, a feed-forward network applied after
            self-attention.
        dropout: Dropout, a dropout layer applied after self-attention
            and the feed-forward network.
        norm_first: bool, if True, normalization is applied before
            self-attention and feed-forward network, otherwise after.

    Methods:
        __call__: Apply the encoder block operations to the input.

    Example:
        ```python

        # Initialize an encoder block with specific configurations
        encoder_block = TransformerEncoderBlock(
            norm1=NormLayer(embed_dim),
            norm2=NormLayer(embed_dim),
            self_attention=AttentionLayer(...),
            mlp=FeedForwardNetwork(...),
            dropout=DropoutLayer(dropout_prob),
            norm_first=True
        )

        # Apply the encoder block to an input tensor
        output = encoder_block(x, mask=attention_mask)
        ```
    """

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
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """Apply the operations of the Transformer encoder block to the input tensor.

        Args:
            x: Array, the input tensor with shape (batch_size, seq_len,
                embed_dim).
            mask: Optional[Array], the mask tensor for self-attention,
                usually with shape (batch_size, 1, seq_len, seq_len).

        Returns:
            Array: The output tensor after applying the encoder block
                operations, with the same shape as the input.

        Example:
            ```python
            # Assuming encoder_block is an instance of TransformerEncoderBlock and x is the input tensor
            output = encoder_block(x, mask=attention_mask)
            ```

        """
        if self.norm_first:
            h = self.norm1(x)
        else:
            h = x

        x = x + self.dropout(
            self.self_attention(h, mask=mask, **kwargs),
            **kwargs,
        )

        if self.norm_first:
            h = self.norm2(x)
        else:
            h = x = self.norm1(x)

        x = x + self.dropout(self.mlp(h, **kwargs), **kwargs)

        if not self.norm_first:
            x = self.norm2(x)

        return x


def transformer_encoder_block(
    embed_dim: int,
    num_heads: int,
    *,
    dropout_prob: float = 0,
    activation: Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    attention_fn: AttentionFn = scaled_dot_product_attention,
    key: Array,
) -> TransformerEncoderBlock:
    """Constructs a Transformer encoder block.

    Args:
        embed_dim: int, dimensionality of the input embeddings.
            num_heads: int, number of attention heads.
        dropout_prob: float, dropout probability.
            activation: Callable, the activation function to use in the
            feed-forward network.
        ff_hidden_size_factor: int, size factor for the feed-forward
            hidden layer.
        norm_first: bool, whether to apply normalization before other
            operations.
        attention_fn: AttentionFn, the attention function to be used.
        key: Array, a random key for initializing weights.

    Returns:
        TransformerEncoderBlock: An instance of TransformerEncoderBlock.

    Example:
        ```python
        key = jax.random.PRNGKey(0)
        encoder_block = transformer_encoder_block(
            embed_dim=128,
            num_heads=4,
            dropout_prob=0.1,
            activation=jax.nn.relu,
            ff_hidden_size_factor=4,
            norm_first=True,
            attention_fn=scaled_dot_product_attention,
            key=key,
        )
        ```

    """
    k1, k2 = rnd.split(key)
    return TransformerEncoderBlock(
        layer_norm((embed_dim,)),
        layer_norm((embed_dim,)),
        attention(embed_dim, num_heads, attention_fn=attention_fn, key=k1),
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
    activation: Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    attention_fn: AttentionFn = scaled_dot_product_attention,
    key: Array,
) -> Sequential:
    """Create a Transformer encoder consisting of a sequence of Transformer encoder blocks.

    Args:
        num_layers: int, the number of encoder blocks in the encoder.
        embed_dim: int, the dimensionality of the input embeddings.
        num_heads: int, the number of attention heads in each encoder block.
        dropout_prob: float, the dropout probability.
        activation: Callable, the activation function used in the feed-forward
            networks.
        ff_hidden_size_factor: int, a factor for the hidden layer size of
            the feed-forward networks.
        norm_first: bool, if True, apply normalization before self-attention
            and feed-forward networks, otherwise after.
        attention_fn: AttentionFn, the attention function used in the
            self-attention mechanism.
        key: Array, a key for random number generation used in dropout layers.

    Returns:
        Sequential: A Sequential container of Transformer encoder blocks.

    Example:
        ```python
        # Create a Transformer encoder
        encoder = transformer_encoder(
            num_layers=6,
            embed_dim=512,
            num_heads=8,
            dropout_prob=0.1,
            activation=jax.nn.relu,
            ff_hidden_size_factor=4,
            norm_first=True,
            attention_fn=my_custom_attention_fn,
            key=random.PRNGKey(0)
        )

        # Apply the encoder to an input tensor
        output = encoder(x, mask=attention_mask)
        ```

    """
    return Sequential(
        [
            transformer_encoder_block(
                embed_dim,
                num_heads,
                dropout_prob=dropout_prob,
                activation=activation,
                ff_hidden_size_factor=ff_hidden_size_factor,
                norm_first=norm_first,
                attention_fn=attention_fn,
                key=k,
            )
            for k in rnd.split(key, num_layers)
        ],
    )


class TransformerDecoderBlock(PyTree):
    """Represents a single block of a Transformer decoder.

    Attributes:
        norm1, norm2, norm3: Norm, normalization layers.
        self_attention, cross_attention: Attention, self-attention and
            cross-attention mechanisms.
        mlp: Sequential, feed-forward neural network.
        dropout: Dropout, dropout layer.
        norm_first: bool, whether to apply normalization before other
            operations.

    Methods:
        __call__: Applies the decoder block to inputs.

    Example:
        ```python
        decoder_block = TransformerDecoderBlock(
            norm1=layer_norm((embed_dim,)),
            norm2=layer_norm((embed_dim,)),
            norm3=layer_norm((embed_dim,)),
            self_attention=attention1,
            cross_attention=attention2,
            mlp=sequential_mlp_instance,
            dropout=Dropout(0.1),
            norm_first=True,
        )
        output = decoder_block(x, context, mask=mask, context_mask=context_mask, key=key)
        ```

    """

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
        mask: Array | None = None,
        context_mask: Array | None = None,
        **kwargs,
    ) -> Array:
        if self.norm_first:
            h = self.norm1(x)
        else:
            h = x

        x = x + self.dropout(
            self.self_attention(h, mask=mask, **kwargs),
            **kwargs,
        )

        if self.norm_first:
            h = self.norm2(x)
        else:
            h = x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(h, context, mask=context_mask, **kwargs),
            **kwargs,
        )

        if self.norm_first:
            h = self.norm3(x)
        else:
            h = x = self.norm2(x)

        x = x + self.dropout(self.mlp(h, **kwargs), **kwargs)

        if not self.norm_first:
            x = self.norm3(x)

        return x


def transformer_decoder_block(
    embed_dim: int,
    num_heads: int,
    *,
    dropout_prob: float = 0,
    activation: Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    attention_fn: AttentionFn = scaled_dot_product_attention,
    key: Array,
) -> TransformerDecoderBlock:
    """Constructs a Transformer decoder block.

    Args:
        embed_dim: int, dimensionality of the input embeddings.
        num_heads: int, number of attention heads.
        dropout_prob: float, dropout probability.
        activation: Callable, the activation function to use in the
            feed-forward network.
        ff_hidden_size_factor: int, size factor for the feed-forward
            hidden layer.
        norm_first: bool, whether to apply normalization before other
            operations.
        attention_fn: AttentionFn, the attention function to be used.
        key: Array, a random key for initializing weights.

    Returns:
        TransformerDecoderBlock: An instance of TransformerDecoderBlock.

    Example:
        ```python
        key = jax.random.PRNGKey(0)
        decoder_block = transformer_decoder_block(
            embed_dim=128,
            num_heads=4,
            dropout_prob=0.1,
            activation=jax.nn.relu,
            ff_hidden_size_factor=4,
            norm_first=True,
            attention_fn=scaled_dot_product_attention,
            key=key,
        )
        ```

    """
    k1, k2, k3 = rnd.split(key, 3)
    return TransformerDecoderBlock(
        layer_norm((embed_dim,)),
        layer_norm((embed_dim,)),
        layer_norm((embed_dim,)),
        attention(embed_dim, num_heads, attention_fn=attention_fn, key=k1),
        attention(embed_dim, num_heads, attention_fn=attention_fn, key=k2),
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
    activation: Callable[[Array], Array] = nn.relu,
    ff_hidden_size_factor: int = 4,
    norm_first: bool = True,
    attention_fn: AttentionFn = scaled_dot_product_attention,
    key: Array,
) -> Sequential:
    """Constructs a Transformer decoder composed of multiple
    TransformerDecoderBlock instances.

    Args:
        num_layers: int, number of decoder blocks.
        embed_dim: int, dimensionality of the input embeddings.
        num_heads: int, number of attention heads.
        dropout_prob: float, dropout probability.
        activation: Callable, the activation function to use in the feed-forward network.
        ff_hidden_size_factor: int, size factor for the feed-forward hidden layer.
        norm_first: bool, whether to apply normalization before other operations.
        attention_fn: AttentionFn, the attention function to be used.
        key: Array, a random key for initializing weights.

    Returns:
        Sequential: A Sequential container of TransformerDecoderBlock instances.

    Example:
        ```python
        key = jax.random.PRNGKey(0)
        decoder = transformer_decoder(
            num_layers=6,
            embed_dim=128,
            num_heads=4,
            dropout_prob=0.1,
            activation=jax.nn.relu,
            ff_hidden_size_factor=4,
            norm_first=True,
            attention_fn=scaled_dot_product_attention,
            key=key,
        )
        ```

    """
    return Sequential(
        [
            transformer_decoder_block(
                embed_dim,
                num_heads,
                dropout_prob=dropout_prob,
                activation=activation,
                ff_hidden_size_factor=ff_hidden_size_factor,
                norm_first=norm_first,
                attention_fn=attention_fn,
                key=k,
            )
            for k in rnd.split(key, num_layers)
        ],
    )
