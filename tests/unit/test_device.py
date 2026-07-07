"""CPU-runnable tests for device resolution and CPU-only determinism.

Covers the probe behavior, the float32 boundary cast, the CPU default, the
fail-loud MPS raise, and the determinism gate. All tests run on CPU /
CI: the MPS-unavailable path is driven via monkeypatch rather than hardware.
"""

import numpy as np
import pytest
import torch

from alphago.device import enable_cpu_determinism, get_device
from alphago.estimator import NACNetEstimator
from alphago.games import NoughtsAndCrosses


def test_get_device_none_is_cpu_default():
    """None resolves to CPU (the deterministic default)."""
    assert get_device(None).type == "cpu"


def test_get_device_cpu_is_cpu_default():
    """The explicit 'cpu' string resolves to CPU (the deterministic default)."""
    assert get_device("cpu").type == "cpu"


def test_get_device_unknown_raises():
    """An unknown device string is rejected with ValueError (input validation)."""
    with pytest.raises(ValueError):
        get_device("cuda")


def test_get_device_mps_unavailable_raises(monkeypatch):
    """device='mps' fails loud with RuntimeError when MPS is unavailable.

    Driven via monkeypatch so it runs on CI/CPU and on the dev machine alike.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        get_device("mps")


def test_vectors_to_input_casts_float32():
    """_vectors_to_input produces float32 even from float64 input.

    The NAC board is 3x3 with a single channel, so a flat length-9 vector is
    the correct shape; a deliberately float64 array must come back as float32.
    """
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(
        learning_rate=0.01, l2_weight=0.1, action_indices=nac.action_indices
    )
    vectors = np.ones((1, 9), dtype=np.float64)
    result = nnet._vectors_to_input(vectors)
    assert result.dtype == torch.float32


def test_estimator_default_device_is_cpu():
    """An estimator built with no device= resolves to CPU."""
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(action_indices=nac.action_indices)
    assert nnet.device.type == "cpu"


def test_estimator_net_on_resolved_device():
    """The net parameters live on the estimator's resolved device."""
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(action_indices=nac.action_indices)
    assert next(nnet.net.parameters()).device.type == nnet.device.type


def test_vectors_to_input_on_device_float32_preserved():
    """_vectors_to_input lands on self.device with the float32 cast intact."""
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(action_indices=nac.action_indices)
    result = nnet._vectors_to_input(np.ones((1, 9), dtype=np.float64))
    assert result.dtype == torch.float32
    assert result.device.type == nnet.device.type


def test_batch_to_tensors_on_device():
    """_batch_to_tensors places x, pi and z on the resolved device."""
    nac = NoughtsAndCrosses()
    nnet = NACNetEstimator(action_indices=nac.action_indices)
    batch = [((0,) * 9, np.ones(9) / 9, 1.0)]
    x, pi, z = nnet._batch_to_tensors(batch)
    assert x.device.type == nnet.device.type
    assert pi.device.type == nnet.device.type
    assert z.device.type == nnet.device.type


@pytest.fixture
def restore_determinism_flag():
    """Save/restore the global deterministic-algorithms flag around a test.

    enable_cpu_determinism sets torch.use_deterministic_algorithms(True)
    process-wide; without this teardown the flag leaks into later suites
    (e.g. the MPS smoke test, which deterministic algorithms can destabilise).
    """
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def test_enable_cpu_determinism_sets_flag(restore_determinism_flag):
    """enable_cpu_determinism turns on deterministic algorithms."""
    enable_cpu_determinism(0)
    assert torch.are_deterministic_algorithms_enabled()


def test_enable_cpu_determinism_reproduces_draws(restore_determinism_flag):
    """Seeding via enable_cpu_determinism makes torch draws reproducible."""
    enable_cpu_determinism(123)
    first = torch.rand(4)
    enable_cpu_determinism(123)
    second = torch.rand(4)
    assert torch.equal(first, second)
