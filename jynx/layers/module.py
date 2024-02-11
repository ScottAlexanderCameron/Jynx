import typing as tp

from jax import Array

type Key = tp.Optional[Array]
type Module[**T, U] = tp.Callable[T, U]


class RecurrentModule[T, S, U, **P](tp.Protocol):
    def __call__(
        self, x: T, state: S, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[U, S]:
        ...

    @property
    def initial_state(self) -> S:
        ...
