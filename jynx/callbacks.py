import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

type Logs = dict[str, tp.Any]
type TrainState = tp.Any


def StopIfDryRun(dry_run: bool) -> tp.Callable[..., None]:
    def fun(*args):
        del args
        if dry_run:
            raise StopIteration

    return fun


class _Monitor:
    def __init__(self, name: str, mode: tp.Literal["min", "max"] = "min"):
        from numpy import inf

        self.name = name
        self.is_better = (
            (lambda new: new < self.best)
            if mode == "min"
            else (lambda new: new > self.best)
        )
        self.best: float = inf if mode == "min" else -inf


class EarlyStopping:
    def __init__(
        self,
        monitor: str,
        steps_without_improvement: int = 100,
        mode: tp.Literal["min", "max"] = "min",
    ):
        self.monitor = _Monitor(monitor, mode)
        self.steps_without_improvement = steps_without_improvement
        self.last_improvement = 0

    def __call__(self, state: TrainState, logs: Logs):
        step, value = logs["step"], logs[self.monitor.name]
        if self.monitor.is_better(value):
            self.monitor.best = value
            self.last_improvement = step
        if step - self.last_improvement >= self.steps_without_improvement:
            raise StopIteration


class CheckPoint:
    def __init__(
        self,
        path: Path | str,
        monitor: str | None = None,
        mode: tp.Literal["min", "max"] = "min",
    ):
        from os import makedirs
        from os.path import exists

        self.path = Path(path)
        self.monitor = _Monitor(monitor, mode) if monitor is not None else None
        self.best = None
        if not exists(self.path.parent):
            makedirs(self.path.parent)

    def __call__(self, state: TrainState, logs: Logs):
        import cloudpickle as pickle

        if self.monitor is not None:
            value = logs[self.monitor.name]
            if self.monitor.is_better(value):
                self.monitor.best = value
            else:
                return

        with self.path.open("wb") as stream:
            pickle.dump(
                {
                    "state": state,
                    "logs": logs,
                },
                stream,
            )

    def load(self) -> tuple[TrainState, Logs]:
        import cloudpickle as pickle

        with self.path.open("rb") as stream:
            obj = pickle.load(stream)

        return obj["state"], obj["logs"]


class Logger:
    def __init__(self, metrics: tp.Sequence[str] | None = None):
        self.metrics = metrics

    def __call__(self, state: TrainState, logs: Logs):
        if self.metrics is not None:
            metrics = self.metrics
        else:
            metrics = [k for k in logs.keys() if k != "step"]

        for m in metrics:
            self.log(m, logs[m], logs["step"])

    def log_hparams(self, hparams: Logs):
        raise NotImplementedError

    def log(self, metric_name: str, value: tp.Any, step: int):
        ...


class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str, metrics: tp.Sequence[str] | None = None):
        super().__init__(metrics)
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def log_hparams(self, hparams, metrics={}):
        self.writer.add_hparams(hparams, metrics)

    def log(self, metric_name, value, step):
        self.writer.add_scalar(metric_name, value, global_step=step)


class MlflowLogger(Logger):
    def log_hparams(self, hparams):
        import mlflow

        mlflow.log_params(hparams)

    def log(self, metric_name, value, step):
        import mlflow

        mlflow.log_metric(metric_name, value, step)


class ConsoleLogger(Logger):
    def __init__(
        self,
        name: str = __name__,
        metrics: tp.Sequence[str] | None = None,
    ):
        super().__init__(metrics)
        import logging

        self.logger = logging.getLogger(name)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def log_hparams(self, hparams):
        self.logger.info("Hparams: %s", hparams)

    def log(self, metric_name, value, step):
        self.logger.info("Step [%s], %s = %s", step, metric_name, value)


@dataclass(frozen=True)
class BatchMetric[B, T]:
    compute_on_batch: tp.Callable[[TrainState, B], T]
    aggregate: tp.Callable[[list[T]], T] = np.mean  # type: ignore


type Metric[B, T] = BatchMetric[B, T] | tp.Callable[[TrainState], T]


class ComputeMetrics[B]:
    def __init__(
        self,
        data_iter: tp.Iterable[B],
        metrics: tp.Mapping[str, Metric[B, tp.Any]],
        max_batches: int | None = None,
    ):
        self.data_iter = data_iter
        self.metrics = metrics
        self.max_batches = max_batches

    def __call__(self, state: TrainState, logs: Logs):
        metrics: dict = {name: [] for name in self.metrics.keys()}

        for i, batch in enumerate(self.data_iter):
            if self.max_batches is not None and i >= self.max_batches:
                break
            for name, metric in self.metrics.items():
                if isinstance(metric, BatchMetric):
                    metrics[name].append(metric.compute_on_batch(state, batch))

        for name, metric in self.metrics.items():
            if isinstance(metric, BatchMetric):
                logs[name] = metric.aggregate(metrics[name])
            else:
                logs[name] = metric(state)
