"""Compute-device resolution and CPU-only determinism helpers."""

import random

import numpy as np
import torch


def get_device(device: str | None = None) -> torch.device:
    """Resolve a compute device, failing loud on an unavailable MPS request.

    CPU is the deterministic default: ``None`` and ``"cpu"`` both resolve to a
    CPU device. ``"mps"`` resolves to Apple MPS only when it is actually
    available at runtime, raising otherwise rather than silently falling back.

    Args:
        device: ``None`` or ``"cpu"`` resolves to CPU (the deterministic
            default). ``"mps"`` resolves to Apple MPS, or raises if MPS is
            unavailable. Any other value is rejected.

    Returns:
        The resolved ``torch.device``.

    Raises:
        RuntimeError: ``device="mps"`` was requested but MPS is unavailable
            (not an Apple Silicon machine, or this torch build lacks MPS).
        ValueError: an unrecognized device string was passed.
    """
    if device is None or device == "cpu":
        return torch.device("cpu")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "device='mps' requested but MPS is unavailable "
                "(not an Apple Silicon machine, or this torch build lacks MPS). "
                "Use device=None / 'cpu', or run on Apple Silicon."
            )
        return torch.device("mps")
    raise ValueError(f"unsupported device {device!r}; expected None, 'cpu', or 'mps'")


def enable_cpu_determinism(seed: int = 0) -> None:
    """Seed RNGs and enable deterministic algorithms -- CPU only.

    Centralizes the ``random`` / ``numpy`` / ``torch`` seed triple plus
    ``torch.use_deterministic_algorithms(True)``. MPS has ops with no
    deterministic implementation and degrades severely under deterministic
    algorithms, so callers must gate on ``get_device(...).type == "cpu"`` and
    never invoke this on the MPS path.

    Args:
        seed: The seed applied to ``random``, ``numpy``, and ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
