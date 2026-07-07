"""SL-parity gate: the ported net still learns the solver targets.

A trimmed, fully seeded supervised retrain of ``ConnectFourNet`` on the Connect
Four perfect-solver data must clear the pinned parity bar on a held-out dev
slice: top-1 policy move-match accuracy >= 0.50, value sign-agreement >= 0.85,
and value MSE <= 0.35. The net is retrained
IN-GATE from the solver data on every run — the bar can never pass on a
stale or committed checkpoint. This is the independent behavioral signal that
the migrated code path, including the move-parsing label-inversion fix,
genuinely matches the solver.

The trimmed config below (fewer epochs, 20k rows vs the 50k-row / 30-epoch
baseline) was empirically tuned DOWN within the runtime budget until the seeded
run cleared the bar with margin; it is pinned as explicit constants.
The bar is defended: a post-tuning failure is a discovered learning defect,
never a threshold to relax.

The test is ``@pytest.mark.gate`` so the default fast suite (``-m 'not gate'``)
skips it; run explicitly with ``uv run pytest -m gate``.
"""

from pathlib import Path

import numpy as np
import pytest

from alphago.connect_four_data import load_solved_states
from alphago.device import enable_cpu_determinism
from alphago.estimator import ConnectFourNet
from alphago.games.connect_four import ConnectFour

# --- Pinned trimmed SL config (empirically tuned DOWN within the runtime
# budget). This exact seeded config clears the pinned bar with margin.
# Retrained in-gate every run — never evaluated from a committed checkpoint.
SEED = 0
MAX_LINES = 20000
EPOCHS = 24
LEARNING_RATE = 0.02
L2_WEIGHT = 1e-4
VALUE_WEIGHT = 0.5
BATCH_SIZE = 32
DEV_FRACTION = 0.02

# --- The SL parity bar — floors the ported net must clear, never knobs
# to loosen.
TOP1_BAR = 0.50
SIGN_AGREEMENT_BAR = 0.85
VALUE_MSE_BAR = 0.35

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "connect_four_data.txt"


def _sign(x: float) -> int:
    """Return the sign of ``x`` as -1, 0, or 1."""
    return (x > 0) - (x < 0)


def _dev_metrics(estimator, dev_data):
    """Compute (top-1 accuracy, value sign-agreement, value MSE) on ``dev_data``.

    Definitions:
    top-1 is the fraction of dev states whose argmax policy column
    (``max(probs, key=probs.get) + 1``, the ``compute_accuracy`` convention)
    equals the solver's optimal column; sign-agreement is measured only over
    non-draw states (``z != 0``); value MSE is over the full dev slice.

    Args:
        estimator: A trained ``ConnectFourNet`` (callable state -> (probs, value)).
        dev_data: Held-out ``(state, probs_vector, z)`` tuples.

    Returns:
        A ``(top1, sign_agreement, value_mse)`` tuple of floats.
    """
    top1_hits = 0
    sign_hits = 0
    sign_total = 0
    squared_error = 0.0
    for state, probs_vector, z in dev_data:
        probs, value = estimator(state)
        predicted_column = max(probs, key=probs.get) + 1
        optimal_columns = [i + 1 for i, p in enumerate(probs_vector) if p > 0]
        top1_hits += predicted_column in optimal_columns
        squared_error += (value - z) ** 2
        if z != 0:
            sign_total += 1
            sign_hits += _sign(value) == _sign(z)

    top1 = top1_hits / len(dev_data)
    sign_agreement = sign_hits / sign_total if sign_total else float("nan")
    value_mse = squared_error / len(dev_data)
    return top1, sign_agreement, value_mse


@pytest.mark.gate
def test_sl_parity_gate():
    """A trimmed seeded SL retrain clears the pinned parity bar.

    Seeds all RNGs via ``enable_cpu_determinism(SEED)`` before building the net,
    retrains ``ConnectFourNet`` in-gate from a trimmed slice of the solver data,
    then asserts top-1 >= 0.50, sign-agreement >= 0.85, and value MSE <= 0.35 on
    the held-out dev slice.
    """
    assert DATA_PATH.exists(), (
        f"solver data not found at {DATA_PATH}; the SL-parity gate retrains "
        "in-gate from it and cannot run without it."
    )

    enable_cpu_determinism(SEED)
    training_data = load_solved_states(DATA_PATH, max_lines=MAX_LINES)

    # Shuffle then split off the held-out dev slice, mirroring the num_dev =
    # max(1, ...) guard from examples/train_connect_four_sl.py.
    np.random.shuffle(training_data)
    num_dev = max(1, int(DEV_FRACTION * len(training_data)))
    dev_data = training_data[:num_dev]
    train_data = training_data[num_dev:]

    game = ConnectFour()
    estimator = ConnectFourNet(
        learning_rate=LEARNING_RATE,
        l2_weight=L2_WEIGHT,
        value_weight=VALUE_WEIGHT,
        action_indices=game.action_indices,
        device="cpu",
    )

    # Retrain in-gate: one supervised epoch per iteration (training_iters=-1).
    for _ in range(EPOCHS):
        estimator.train(train_data, BATCH_SIZE, -1, mode="supervised", verbose=False)

    top1, sign_agreement, value_mse = _dev_metrics(estimator, dev_data)

    assert top1 >= TOP1_BAR, (
        f"top-1 policy accuracy {top1:.3f} < {TOP1_BAR} — the ported net no "
        "longer learns the solver's optimal move (fix the pipeline, do "
        "not relax the bar)."
    )
    assert sign_agreement >= SIGN_AGREEMENT_BAR, (
        f"value sign-agreement {sign_agreement:.3f} < {SIGN_AGREEMENT_BAR} — the "
        "value head no longer predicts the outcome sign."
    )
    assert value_mse <= VALUE_MSE_BAR, (
        f"value MSE {value_mse:.3f} > {VALUE_MSE_BAR} — the value head no longer "
        "fits the solver outcomes."
    )
