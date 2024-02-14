import typing as tp

import optax

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax import tree_util

type Logs = tp.Dict[str, tp.Any]


class TrainState[Params](tp.NamedTuple):
    params: Params
    grads: Params
    opt_state: optax.OptState


class OptimizationResult[Params](tp.NamedTuple):
    params: Params
    logs: Logs


type TrainStep[Params, Batch, Extras] = tp.Callable[
    [TrainState[Params], Batch, Array], tp.Tuple[TrainState[Params], Array, Extras]
]


def predict_on_batch(loss_fn):
    def fn(model, batch, key=None):
        return loss_fn(model(batch[0], key=key), *batch[1:]).mean()

    return fn


def make_train_step[Params, Batch](
    loss_fn: tp.Callable[[Params, Batch, Array], Array],
    optimizer: optax.GradientTransformation,
    loss_has_aux: bool = False,
    use_pmap: bool = False,
    donate_args: bool = False,
) -> TrainStep[Params, Batch, tp.Any]:
    def train_step(
        state: TrainState,
        batch: Batch,
        key: Array,
    ) -> tp.Tuple[TrainState, float, tp.Any]:
        loss, grads = jax.value_and_grad(loss_fn, has_aux=loss_has_aux)(
            state.params, batch, key
        )
        if loss_has_aux:
            loss, aux = loss
        else:
            aux = None
        if use_pmap:
            grads = lax.pmean(grads, axis_name="device")

        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return (
            TrainState(params, grads, opt_state),
            loss,
            aux,
        )

    donate = (1,) if donate_args else ()
    if use_pmap:
        return jax.pmap(train_step, axis_name="device", donate_argnums=donate)
    else:
        return jax.jit(train_step, donate_argnums=donate)


def fit[Params, Batch](
    params: Params,
    *,
    loss_fn: tp.Callable[[Params, Batch, Array], Array],
    data_iter: tp.Iterable[Batch],
    optimizer: optax.GradientTransformation,
    max_steps: tp.Optional[int] = None,
    callbacks: tp.Sequence[tp.Callable[[TrainState[Params], Logs], None]] = (),
    n_steps_between_calls: int = 100,
    loss_has_aux: bool = False,
    use_pmap: bool = False,
    donate_args: bool = False,
    train_step: tp.Optional[TrainStep[Params, Batch, tp.Any]] = None,
    state: tp.Optional[TrainState[Params]] = None,
    key: Array,
) -> OptimizationResult[Params]:
    if train_step is None:
        train_step = make_train_step(
            loss_fn,
            optimizer,
            loss_has_aux,
            use_pmap,
            donate_args,
        )

    if state is None:
        state = TrainState(
            params,
            tree_util.tree_map(jnp.zeros_like, params),
            optimizer.init(params),  # type: ignore
        )

    cum_loss = 0.0
    rng = key_seq(key)
    logs = {}

    for step, batch in enumerate(data_iter):
        state, loss, *_ = train_step(state, batch, next(rng))
        cum_loss += float(loss)

        if (step + 1) % n_steps_between_calls == 0:
            logs = {"loss": cum_loss / n_steps_between_calls, "step": step + 1}
            cum_loss = 0.0
            try:
                for callback in callbacks:
                    callback(state, logs)
            except StopIteration:
                break

        if max_steps and step >= max_steps:
            break

    return OptimizationResult(state.params, logs)  # type: ignore


def key_seq(key: Array, split: int = 16):
    import jax.random as rnd

    while True:
        key, *others = rnd.split(key, split)
        yield from others
