"""Metric-logging seam decoupling the training loop from any backend.

Named ``metric_logger`` (not ``logging``) to avoid shadowing the stdlib
``logging`` module for intra-package imports.
"""

from typing import Protocol


class MetricLogger(Protocol):
    """Minimal metric sink the training loop logs scalars through.

    Any object exposing ``log_scalar`` and ``close`` with these signatures
    satisfies the contract; the loop never imports a concrete backend.
    """

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Record a single scalar ``value`` under ``tag`` at ``step``."""
        ...

    def close(self) -> None:
        """Flush and release any backend resources."""
        ...


class NullLogger:
    """No-op :class:`MetricLogger` — the default when no backend is injected.

    Every method is a no-op so the loop can call ``logger.log_scalar(...)``
    unconditionally without guarding on ``logger is None``.
    """

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Discard the scalar."""

    def close(self) -> None:
        """Do nothing."""


class WandbLogger:
    """A :class:`MetricLogger` backed by Weights & Biases.

    ``wandb`` is imported lazily inside ``__init__`` so that ``NullLogger``
    paths and test collection never pay the heavy import cost.
    """

    def __init__(self, wandb_cfg, config: dict) -> None:
        """Start a wandb run from ``wandb_cfg`` and log ``config`` as hyperparameters.

        Args:
            wandb_cfg: A config object exposing ``project``, ``name``, and
                ``mode`` attributes describing the run.
            config: A plain dict of resolved hyperparameters. The caller is
                responsible for converting an OmegaConf config via
                ``OmegaConf.to_container(cfg, resolve=True)`` before passing it
                here — a raw ``DictConfig`` must not be handed to wandb.
        """
        import wandb

        self._run = wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            mode=wandb_cfg.mode,
            config=config,
        )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log ``value`` under ``tag`` at ``step``.

        ``step`` must be non-decreasing across calls;
        the caller's ``alphago_step`` axis guarantees this.
        """
        self._run.log({tag: value}, step=step)

    def close(self) -> None:
        """Finish the wandb run."""
        self._run.finish()
