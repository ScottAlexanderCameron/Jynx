from collections.abc import Callable
from typing import Protocol

type Module[**T, U] = Callable[T, U]


class RecurrentModule[T, S, U, **P](Protocol):
    """A protocol class for recurrent neural network layers.

    This class is used to annotate types that behave as recurrent layers within neural networks, such as GRU or LSTM layers.
    It defines a recurrent layer's expected interface, including the call method for processing inputs with a hidden state
    and a property for obtaining the default initial state of the hidden layer.

    Type Parameters:
        T: The type of input to the recurrent layer.
        S: The type of the hidden state for the recurrent layer.
        U: The type of output from the recurrent layer.
        **P: A parameter specification for additional arguments the layer might accept.

    The `RecurrentModule` interface expects a callable implementation that takes an input of type `T`, a hidden state of type `S`,
    and returns a tuple of the output of type `U` and the updated hidden state of type `S`.

    The `initial_state` property should provide the default initial value for the layer's hidden state, facilitating the initialization
    of the recurrent model's state.

    Example Usage:
        ```python
        type RNN = RecurrentModule[Array, Array, Array, []]
        model: RNN = create_model()  # some initialization logic
        state = model.initial_state  # Obtain the default initial state
        for x in input_data:
            y, state = model(x, state)  # Process input `x` with current state, producing output `y` and updated state
        ```
    """

    def __call__(
        self,
        x: T,
        state: S,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[U, S]:
        """Processes an input `x` with the current hidden state `state`, producing an output and updating the state.

        Args:
            x (T): The input to the recurrent layer.
            state (S): The current hidden state of the recurrent layer.
            *args: Additional positional arguments as specified by `P.args`.
            **kwargs: Additional keyword arguments as specified by `P.kwargs`.

        Returns:
            tuple[U, S]: A tuple containing the output of the recurrent layer and the updated hidden state.

        """
        ...

    @property
    def initial_state(self) -> S:
        """Provides the default initial value for the recurrent layer's hidden state.

        This property should be implemented to return the initial state of the hidden layer, allowing the recurrent model
        to be properly initialized before processing any input data.

        Returns:
            S: The default initial state of the recurrent layer's hidden state.

        """
        ...
