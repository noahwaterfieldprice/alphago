"""Determinism gate: two same-seed CPU runs are identical.

Reproducibility is the guarantee that makes every other gate debuggable: a red
gate must fail the same way twice. This gate runs a SHORT seeded CPU training
twice and asserts (a) the two per-step ``loss/total`` sequences are exactly
equal and (b) the final network ``state_dict``s are tensor-equal
(``torch.equal`` over every parameter). Each run re-seeds via
``enable_cpu_determinism(cfg_seed)`` BEFORE building the estimator
(re-seed before EACH run, so weight init is reproducible), and both
runs are forced onto the CPU (``device="cpu"``).

Determinism is guaranteed on CPU ONLY. MPS has ops with no deterministic
implementation, so ``enable_cpu_determinism`` must never be called on the MPS
path and this gate never touches it. Determinism here does not depend on
hash ordering: the seeded RNG triple and the fixed batch order fully determine
each run, so no interpreter hash-seed pinning is required.

The test is ``@pytest.mark.gate`` so the default fast suite (``-m 'not gate'``)
skips it; run explicitly with ``uv run pytest -m gate``.
"""

from pathlib import Path

import pytest
import torch
from recording_logger import RecordingLogger

from alphago.connect_four_data import load_solved_states
from alphago.device import enable_cpu_determinism
from alphago.estimator import ConnectFourNet
from alphago.games.connect_four import ConnectFour

# --- Pinned short determinism config (seconds-scale but still gate-marked).
SEED = 0
MAX_LINES = 512
STEPS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.01
L2_WEIGHT = 1e-4
VALUE_WEIGHT = 0.5

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "connect_four_data.txt"


def _seeded_cpu_run():
    """Run ONE short seeded CPU training and return (loss/total seq, state_dict).

    Re-seeds via ``enable_cpu_determinism(SEED)`` BEFORE building the estimator,
    forces the CPU device, trains ``STEPS`` fixed-order
    ``train_step``s while logging each step's total loss into a
    :class:`RecordingLogger`, and returns the recorded ``loss/total`` value
    sequence alongside the final network ``state_dict``.

    Returns:
        A ``(losses, state_dict)`` tuple where ``losses`` is the list of per-step
        ``loss/total`` values and ``state_dict`` is the trained ``net`` state.
    """
    # The training path below depends only on the seeded RNG triple and a fixed
    # batch order, not on dict hash ordering, so determinism holds without any
    # interpreter hash-seed pinning.
    enable_cpu_determinism(SEED)
    data = load_solved_states(DATA_PATH, max_lines=MAX_LINES)

    game = ConnectFour()
    estimator = ConnectFourNet(
        learning_rate=LEARNING_RATE,
        l2_weight=L2_WEIGHT,
        value_weight=VALUE_WEIGHT,
        action_indices=game.action_indices,
        device="cpu",
    )

    logger = RecordingLogger()
    for step in range(STEPS):
        start = (step * BATCH_SIZE) % len(data)
        batch = data[start : start + BATCH_SIZE]
        summary = estimator.train_step(batch, return_summary=True)
        logger.log_scalar("loss/total", summary["total"], step)

    losses = [value for _, value in logger.records["loss/total"]]
    return losses, estimator.net.state_dict()


@pytest.mark.gate
def test_determinism_gate():
    """Two same-seed CPU runs yield equal losses + tensor-equal state_dicts.

    CPU-only determinism: MPS is explicitly not guaranteed and is never
    touched here.
    """
    assert DATA_PATH.exists(), (
        f"solver data not found at {DATA_PATH}; the determinism gate trains a "
        "short seeded run from it and cannot run without it."
    )

    losses_a, state_dict_a = _seeded_cpu_run()
    losses_b, state_dict_b = _seeded_cpu_run()

    assert losses_a == losses_b, (
        "per-step loss/total sequences diverged across two same-seed CPU runs: "
        f"{losses_a} != {losses_b} — the pipeline is non-reproducible."
    )
    assert state_dict_a.keys() == state_dict_b.keys(), (
        "final state_dict keys differ across two same-seed CPU runs."
    )
    assert all(torch.equal(state_dict_a[k], state_dict_b[k]) for k in state_dict_a), (
        "final state_dict tensors differ across two same-seed CPU runs."
    )
