import typing as tp
from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass, fields

import jax.tree_util as tu
from jax import Array
from jax import numpy as jnp
from jax import random as rnd

from ..pytree import dataclass_flatten, static
from .module import Module, RecurrentModule
from .static import Fn


class ModuleList[M: Module](list[M]):
    """A list-like container for neural network modules, treating them
    as a collective module.

    This class provides a way to group multiple neural network modules
    together, allowing operations and transformations to be applied
    to the group as a whole. It is particularly useful for defining
    sequences of layers or components that should be processed in order.

    Inherits from Python's native list, providing all standard list functionalities.
    """

    def __init_subclass__(cls):
        return tu.register_pytree_with_keys_class(
            dataclass(cls, frozen=True, init=False),  # type: ignore
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
                raise ValueError(
                    f"Required parameter: {
                                 f.name} was not specified",
                )
            object.__setattr__(self, f.name, val)

    def tree_flatten(self):
        ch, aux = dataclass_flatten(self)
        return (*ch, *self), aux

    def tree_flatten_with_keys(self):
        ch, (keys, static) = dataclass_flatten(self)
        return (
            [(tu.GetAttrKey(k), v) for k, v in zip(keys, ch, strict=True)]
            + [(tu.SequenceKey(i), v) for i, v in enumerate(self)],
            (keys, static),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, static = aux
        ch = dict(zip(keys, children[: len(keys)], strict=True))
        return cls(children[len(keys) :], **ch, **static)

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self)})"

    @tp.overload
    def __getitem__(self, idx: tp.SupportsIndex) -> M: ...

    @tp.overload
    def __getitem__(self, idx: slice) -> tp.Self: ...

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if isinstance(idx, slice):
            ch, (keys, static) = dataclass_flatten(self)
            ch = dict(zip(keys, ch, strict=True))
            return type(self)(item, **ch, **static)
        else:
            return item


class Sequential[T, **P](ModuleList[Module[tp.Concatenate[T, P], T]]):
    """A container for sequencing neural network layers or modules linearly.

    `Sequential` allows for the creation of a composite model by stacking
    layers or modules where the output of one layer is the input to the
    next. This is particularly useful for creating feedforward neural
    networks where data flows linearly through layers.

    Example:
        ```python
        model = Sequential([
            linear(in_size=128, out_size=256, key=k1),
            Fn(jax.nn.relu),
            linear(in_size=256, out_size=10, key=k2),
        ])
        output = model(input_data)
        ```

        The example above creates a simple feedforward network with two
        dense layers and a ReLU activation function in between.

    """

    def __call__(self, x: T, *args: P.args, **kwargs: P.kwargs) -> T:
        for layer in self:
            x = layer(x, *args, **kwargs)

        return x


class Parallel[**T, U](ModuleList[Module[T, U]]):
    """A container for applying multiple neural network layers or modules
    in parallel to the same input.

    `Parallel` facilitates the construction of neural network
    architectures where different operations need to be performed
    on the same input simultaneously. The outputs of these parallel
    layers can then be aggregated or processed further. This is useful
    in complex architectures that require multi-pathway processing or
    feature extraction.

    Example:
        ```python
        model = Parallel([
            conv(in_channels=3, out_channels=64, kernel_shape=(3, 3), key=k1),
            MaxPoolingLayer(window=(2, 2), strides=(2, 2),
            conv_transpose(in_channels=3, out_channels=64, kernel_shape=(3, 3), key=k2)
        ])
        combined_output = model(input_data)
        ```

        In the example above, a convolutional layer, a max pooling layer,
        and a conv transpose layer are applied in parallel to the input
        data. The outputs from these layers can then be combined or processed
        according to specific architectural needs.

    """

    def __call__(self, *args: T.args, **kwargs: T.kwargs) -> Sequence[U]:
        outputs = []
        for layer in self:
            outputs.append(layer(*args, **kwargs))
        return outputs


type _RecLayer[T, S, **P] = (
    Module[tp.Concatenate[T, P], T] | RecurrentModule[T, S, T, P]
)


class Recurrent[T, S, **P](ModuleList[_RecLayer[T, S, P]]):
    """A container for managing a sequence of layers where some may have
    recurrent states.

    `Recurrent` behaves similarly to `Sequential` but is specifically
    tailored for layers that can maintain a hidden state, making it
    suitable for recurrent neural network (RNN) architectures. It allows
    for a mix of both recurrent and non-recurrent layers. Recurrent layers
    should conform to the `RecurrentModule` Protocol, accepting a state
    argument and returning a tuple of their output and updated state. They
    should also define an `initial_state` property. Non-recurrent layers
    should conform to the `Module` protocol.

    When iterating through the layers, `Recurrent` passes the output of
    the previous layer as the input to the next. For recurrent layers,
    it also manages the passing and updating of their states. The final
    output is the output of the last layer, and the method returns a
    list of updated states for all recurrent layers.

    Example:
        ```python
        rnn = Recurrent([
            rnn_cell(10, 256, key=k1),      # Recurrent layer
            Fn(jax.nn.relu),                # Non-recurrent layer
            lstm_cell(256, 256, key=k2),    # Recurrent layer
            Fn(jax.nn.sigmoid),             # Non-recurrent layer
            linear(256, 1, key=k3),         # Non-recurrent layer
        ])
        state = rnn.initial_state  # Initial states for recurrent layers: [Array, None, (Array, Array), None, None]
        y1, state = rnn(x1, state)  # Process input x1, update states
        ```

        In this example, `rnn` is a `Recurrent` container with a mix of
        recurrent (e.g., `rnn_cell`, `lstm_cell`) and non-recurrent layers
        (e.g., `Fn(jax.nn.relu)`, `linear`). The `initial_state` provides
        initial states for the recurrent layers, while `None` is used for
        non-recurrent layers. The `rnn` object processes an input `x1`,
        updating the states of the recurrent layers and returning the final
        output `y1` and the updated states.

    """

    type States = Sequence[S | None]

    def __call__(
        self,
        x: T,
        state: States,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[T, States]:
        from itertools import zip_longest

        new_states = []

        for layer, st in zip_longest(self, state):
            if st is None:
                layer = tp.cast(Callable[tp.Concatenate[T, P], T], layer)
                x = layer(x, *args, **kwargs)
            else:
                layer = tp.cast(RecurrentModule, layer)
                x, st = layer(x, st, *args, **kwargs)
            new_states.append(st)

        return x, new_states

    @property
    def initial_state(self):
        """Provides the default initial states for all recurrent layers
        within the Recurrent container.

        This property iterates through each layer in the Recurrent
        container, checking if the layer conforms to the RecurrentModule
        Protocol and thus has a defined initial state. It compiles a
        list of initial states for all layers, where layers that are not
        recurrent or do not require an initial state are represented by
        `None`.

        The returned list of states is aligned with the order of the
        layers, allowing each layer's initial state to be easily accessed
        and managed during the recurrent processing of inputs.

        Returns:
            List[Optional[S]]: A list of initial states for the recurrent
                layers, where non-recurrent layers have `None` as
                their state.

        """
        return [
            layer.initial_state if hasattr(layer, "initial_state") else None  # type: ignore
            for layer in self
        ]

    def scan(
        self,
        xs: Sequence[T],
        state: States,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Sequence[T], States]:
        """Efficiently processes a sequence of inputs through the
        recurrent model using JAX's lax.scan.

        This method takes a sequence of inputs (`xs`) and an initial
        state, and iterates over the sequence, applying the model to
        each element. For recurrent layers, the method passes along the
        updated state at each step. This allows for efficient processing
        of sequences with shared model parameters across time steps,
        which is typical in recurrent neural networks.

        The method is designed to be used with layers that conform to
        the RecurrentModule Protocol, accepting a state argument and
        returning a tuple of their output and updated state. It also
        supports layers conforming to the Module protocol, assuming no
        state update for these layers.

        Args:
            xs (Sequence[T]): A sequence of inputs to be processed by the model.
            state (Sequence[Optional[S]]): The initial states for all
                layers within the Recurrent container. Each element in
                the list corresponds to the initial state of a layer,
                with `None` for non-recurrent layers.

        Returns:
            Tuple[Sequence[T], List[Optional[S]]]: A tuple containing
            the final output after processing the entire sequence and
            a list of updated recurrent states for each layer, aligned
            with the order of the layers.

        Example:
            ```python
            rnn = Recurrent([ ... ])
            state = rnn.initial_state  # [Array, None, None, (Array, Array), None, None]
            ys, state = rnn.scan(xs, state)
            ```

        """
        from jax import eval_shape, lax
        from jax.tree_util import tree_map

        _, state_shapes = eval_shape(self, xs[0], state, *args, **kwargs)
        state = tree_map(
            lambda s, shape_dtype: jnp.broadcast_to(s, shape_dtype.shape),
            state,
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

        carry, y = lax.scan(forward, (state, key), xs)
        return y, carry[0]  # type: ignore


class DenselyConnected[**P](ModuleList[Module[tp.Concatenate[Array, P], Array]]):
    """A container that creates a densely connected block of layers,
    where each layer receives the concatenation of the outputs from all
    previous layers as its input.

    In a `DenselyConnected` block, the output from each layer is
    concatenated with the original input and the outputs of all preceding
    layers.

    ```
       ↓
    ╭─────╮┌───────────┬────────┐
    │block│┤╭─────╮┌───│──────┐ │
    ╰─────╯└│block│┤╭─────╮   │ │
            ╰─────╯└│block│┐╭─────╮
                    ╰─────╯└│block│
                            ╰─────╯
                               ↓
    ```

    Attributes:
        concat_axis (int): The axis along which to concatenate the
            outputs. Typically, this is the feature/channel axis.
        only_inputs (bool): If True, only the inputs to the block are
            concatenated with each layer's output. If False, the output
            of each layer is concatenated with the combined input and
            outputs of all preceding layers.

    Example:
        ```python
        # Creating a densely connected block with convolutional layers
        def conv_block(in_channels, out_channels, key):
            k1, k2 = jax.random.split(key)
            return sequential(
                conv(in_channels, out_channels, kernel_shape=(3, 3), padding="SAME", key=k1),
                jax.nn.silu,
                conv(out_channels, out_channels, kernel_shape=(3, 3), padding="SAME", key=k2),
            )

        dense_block = DenselyConnected([
            conv_block(in_channels=32, out_channels=32, key=k1),
            conv_block(in_channels=64, out_channels=32, key=k2),
            conv_block(in_channels=96, out_channels=32, key=k3),
        ], concat_axis=1)  # Assuming channel-first data format

        # Forward pass through the dense block
        output = dense_block(input_data)
        # output.shape[1] == 128
        ```

        In the example above, each `conv_block` in the `DenselyConnected`
        block receives a concatenation of the input data and the outputs
        from all preceding layers. The `concat_axis=1` indicates that the
        concatenation is performed along the channel axis, which is typical
        for convolutional layers in a channel-first data format.

    """

    concat_axis: int = static(default=-1)
    only_inputs: bool = static(default=False)

    def __call__(self, x: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        inp = x
        for layer in self:
            if not self.only_inputs:
                inp = x
            x = layer(x, *args, **kwargs)
            x = jnp.concatenate((x, inp), axis=self.concat_axis)
        return x


def wrap_if_function(arg):
    """Wraps a callable argument in a Fn object if it is a function,
    otherwise returns it as is.

    This function checks if the provided argument is a pytree structure.
    If it is (ie a neural network layer) then it returns the object
    as is.  otherwise we assume that the passed object is a function
    (eg jax.nn.relu) but it is not a valid pytree structure, in which
    case it is wrapped in an Fn obect, This utility is used to ensure
    that functions can be seamlessly included in model architectures
    alongside other neural network layers.

    Args:
        arg: The argument to be potentially wrapped in an Fn object.

    Returns:
        The argument wrapped in an Fn object if it is a function, or the
        argument itself if it is a valid pytree.

    """
    import jax.tree_util as tu

    treedef = tu.tree_structure(arg)
    isleaf = tu.treedef_is_leaf(treedef)
    if isleaf and treedef.num_leaves != 0:
        assert callable(arg), "layers must be callable"
        return Fn(arg)
    else:
        return arg


def sequential(*layers):
    """Constructs a sequential container from a list of layers and
    functions.

    This convenience function allows for the easy creation of a neural
    network where each layer or function is applied in sequence, with
    the output of one being the input to the next. Raw functions, such
    as activation functions from jax.nn, are automatically wrapped in
    an Fn object, allowing them to be included directly.

    Args:
        *layers: A list of layers or functions to be applied
            sequentially. Can include raw functions, or any callable
            that conforms to the Module protocol.

    Returns:
        Sequential: A Sequential container with the given layers.

    Example:
        ```python
        net = sequential(
            linear(10, 256, key=k1),
            jax.nn.relu,
            linear(256, 1, key=k2),
        )
        ```

    """
    return Sequential(map(wrap_if_function, layers))


def parallel(*layers):
    """Constructs a parallel container from a list of layers and
    functions.

    This function creates a model where each layer or function is applied
    in parallel to the same input, and their outputs can be combined
    or processed further. Similar to `sequential`, raw functions are
    automatically wrapped in an Fn object.

    Args:
        *layers: A list of layers or functions to be applied in
            parallel. Includes raw functions, or any callable that
            conforms to the Module protocol.

    Returns:
        Parallel: A Parallel container with the given layers.

    Example:
        ```python
        net = parallel(
            conv(in_channels=3, out_channels=64, kernel_shape=(3, 3), key=k1),
            MaxPooling(window=(2, 2), strides=(2, 2)),
            jax.nn.relu,
        )
        ```

    """
    return Parallel(map(wrap_if_function, layers))


def recurrent(*layers):
    """Constructs a recurrent container from a list of layers and
    functions.

    This function is designed for creating recurrent neural network
    architectures, allowing layers or functions that manage recurrent
    states to be sequenced. Raw functions are automatically wrapped in
    an Fn object.

    Args:
        *layers: A list of layers, functions, or any callables that
            either conform to the RecurrentModule protocol (accepting a
            state argument and returning a tuple of output and updated
            state) or the Module protocol.

    Returns:
        Recurrent: A Recurrent container with the given layers.

    Example:
        ```python
        net = recurrent(
            rnn_cell(10, 256, key=k1),
            jax.nn.relu,
            lstm_cell(256, 256, key=k2),
        )
        ```

    """
    return Recurrent(map(wrap_if_function, layers))


def densely_connected(*layers):
    """Constructs a densely connected container from a list of layers
    and functions.

    This function facilitates the creation of dense blocks where the
    output from each layer is concatenated with the outputs of all
    previous layers. It automatically wraps raw functions in an Fn object.

    Args:
        *layers: A list of layers or functions to be densely
            connected. Includes Module instances, raw functions,
            or any callable that conforms to the Module protocol.

    Returns:
        DenselyConnected: A DenselyConnected container with the given layers.

    Example:
        ```python
        net = densely_connected(
            conv(in_channels=16, out_channels=16, kernel_shape=(3, 3), padding="SAME", key=k1),
            jax.nn.relu,
            conv(in_channels=64, out_channels=32, kernel_shape=(3, 3), padding="SAME", key=k2),
        )
        ```

    """
    return DenselyConnected(map(wrap_if_function, layers))
