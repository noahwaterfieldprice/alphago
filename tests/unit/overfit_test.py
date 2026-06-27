"""Learning proof: the ported ConnectFourNet overfits a tiny batch.

Before any self-play wiring, prove the torch-ported net actually learns.
Training many steps on a fixed, tiny batch
 (with ``l2_weight=0`` so L2 cannot floor the loss) must drive the
combined loss down and top-1 policy move-match accuracy to 1.0 (the net memorizes
the tiny set, which proves gradients, the loss, and the wiring are correct).

The test reads only the 64-line fixture (``tests/fixtures/connect_four_slice.txt``),
never the 95 MB solver file, and seeds torch/numpy/random for determinism.
"""

import random
from pathlib import Path

import numpy as np
import torch

from alphago.connect_four_data import load_solved_states
from alphago.estimator import ConnectFourNet
from alphago.games.connect_four import ConnectFour

# Resolve the fixture relative to this file so the test is cwd-independent and
# can never accidentally read the full solver file.
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "connect_four_slice.txt"


def _seed_everything(seed: int = 0) -> None:
    """Seed torch, numpy and random so the overfit run is deterministic."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _top1_accuracy(estimator, batch) -> float:
    """Top-1 policy move-match accuracy on the batch.

    Mirrors ``compute_accuracy`` (examples/train_connect_four_sl.py): the
    estimator's argmax column (1-indexed) must equal the batch's optimal column,
    recovered as ``argmax(probs_vector) + 1``.
    """
    correct = 0
    for state, probs_vector, _ in batch:
        probs_dict, _ = estimator(state)
        predicted = max(probs_dict, key=probs_dict.get) + 1
        optimal = int(np.argmax(probs_vector)) + 1
        correct += int(predicted == optimal)
    return correct / len(batch)


def test_overfit_tiny_batch_connect_four():
    """The ported net memorizes a tiny batch: loss collapses, top-1 acc -> 1.0."""
    _seed_everything(0)

    batch = load_solved_states(FIXTURE, max_lines=32)
    assert len(batch) == 32

    game = ConnectFour()
    # learning_rate=5e-2 (tuned up from the ~1e-2 starting point the plan allows):
    # with full-batch SGD on 32 samples, 1e-2 plateaus for hundreds of steps before
    # descending, whereas 5e-2 drives the combined loss to ~1e-3 well inside 1000
    # steps while staying fast (CPU, seconds) and deterministic.
    estimator = ConnectFourNet(
        learning_rate=5e-2,
        l2_weight=0.0,  # L2 must not floor the loss for the overfit proof.
        value_weight=1.0,
        action_indices=game.action_indices,
    )

    loss = None
    for _ in range(1000):
        loss = estimator.train_step(batch, return_summary=True)

    # Threshold tuned empirically for this seed/config and recorded here. The hard
    # requirement is top-1 accuracy == 1.0; the loss bound proves the loss collapses.
    assert loss < 0.05, f"overfit loss did not collapse: {loss}"
    assert _top1_accuracy(estimator, batch) == 1.0
