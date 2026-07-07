"""Tests for the metric-logging seam."""

from alphago.metric_logger import MetricLogger, NullLogger, WandbLogger


class FakeLogger:
    """A duck-typed :class:`MetricLogger` recording calls, for tests.

    Shaped like ``tests/unit/mock_estimator.py`` — a tiny in-test fake that
    conforms to the protocol structurally without inheriting from it.
    """

    def __init__(self):
        self.scalars = []
        self.closed = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def close(self) -> None:
        self.closed = True


def test_null_logger_log_scalar_returns_none():
    logger = NullLogger()
    assert logger.log_scalar("loss", 1.0, 0) is None


def test_null_logger_close_returns_none():
    logger = NullLogger()
    assert logger.close() is None


def test_null_logger_conforms_to_protocol():
    logger: MetricLogger = NullLogger()
    logger.log_scalar("loss", 0.5, 3)
    logger.close()


def test_fake_logger_conforms_to_protocol_and_records():
    logger: MetricLogger = FakeLogger()
    logger.log_scalar("loss/total", 2.0, 0)
    logger.log_scalar("loss/value", 0.7, 1)
    logger.close()

    assert logger.scalars == [("loss/total", 2.0, 0), ("loss/value", 0.7, 1)]
    assert logger.closed is True


class _StubWandbCfg:
    project = "alphago-test"
    name = "unit"
    mode = "disabled"


def test_wandb_logger_logs_without_network(monkeypatch):
    """WandbLogger records a scalar hermetically via a monkeypatched wandb."""
    import wandb

    logged = []
    finished = []

    class _StubRun:
        def log(self, data, step):
            logged.append((data, step))

        def finish(self):
            finished.append(True)

    init_calls = []

    def _fake_init(project, name, mode, config):
        init_calls.append((project, name, mode, config))
        return _StubRun()

    monkeypatch.setattr(wandb, "init", _fake_init)

    logger = WandbLogger(_StubWandbCfg(), config={"lr": 0.01})
    logger.log_scalar("loss/total", 1.5, 0)
    logger.close()

    assert init_calls == [("alphago-test", "unit", "disabled", {"lr": 0.01})]
    assert logged == [({"loss/total": 1.5}, 0)]
    assert finished == [True]
