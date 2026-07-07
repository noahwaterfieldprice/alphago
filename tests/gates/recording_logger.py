"""Shared ``RecordingLogger`` for the gate tests.

Lives in a plain importable module rather than ``conftest.py`` so both the
session fixture in ``conftest.py`` and the determinism gate can import it by
name. Conftest files are pytest plugins and must never be imported by name.
"""


class RecordingLogger:
    """A :class:`~alphago.metric_logger.MetricLogger` that captures every scalar.

    Mirrors ``NullLogger``'s two-method shape but records each logged scalar so
    the E2E gate can assert over the full ``loss/total`` and ``eval/success_rate``
    sequences after a single training run.

    Attributes:
        records: Maps each tag to the ``(step, value)`` pairs logged under it.
    """

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[int, float]]] = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Record ``value`` under ``tag`` at ``step``."""
        self.records.setdefault(tag, []).append((step, value))

    def close(self) -> None:
        """No-op — nothing to flush or release."""
