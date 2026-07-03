"""MPS op-coverage smoke test.

Proves both estimators run their forward / ``__call__`` inference path and one
SGD ``train_step`` on the MPS device with finite outputs. The module-level skip
gate keeps the suite green-and-skipped on CI/Linux (no MPS) and runs real MPS op
coverage on the Apple Silicon dev machine.
"""

import math

import numpy as np
import pytest
import torch

from alphago.estimator import ConnectFourNet, NACNetEstimator
from alphago.games import ConnectFour, NoughtsAndCrosses

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


def _empty_board_state(est):
    """Empty-board grid the estimator's net consumes via ``_state_to_vector``.

    The net takes a flat ``in_channels * board_h * board_w`` board grid;
    ConnectFour's game state already IS this flat board, while NAC's game state
    is a bitboard, so we build the net's grid representation of the empty board
    (matching ``tests/unit/estimator_test.py``). The input flows through the
    real ``__call__`` / ``_state_to_vector`` path, not a fabricated raw tensor
   .
    """
    cfg = est.cfg
    flat = cfg.in_channels * cfg.board_h * cfg.board_w
    return np.zeros(flat, dtype=np.float32)


@pytest.mark.parametrize(
    "net_cls, game_cls",
    [
        (ConnectFourNet, ConnectFour),
        (NACNetEstimator, NoughtsAndCrosses),
    ],
)
def test_mps_op_coverage_finite_outputs(net_cls, game_cls):
    """Forward + __call__ + SGD train_step run on MPS with finite outputs."""
    game = game_cls()
    est = net_cls(action_indices=game.action_indices, device="mps")
    assert est.device.type == "mps"

    # (a) forward via the __call__ inference path on the empty-board state.
    state = _empty_board_state(est)
    probs_dict, value = est(state)
    assert all(math.isfinite(p) for p in probs_dict.values())
    assert math.isfinite(value)

    # (b) one SGD train_step (exercises the SGD optimizer's step, not an
    #     Adam-family optimizer) over a synthesized (state, pi=uniform, z=0.0)
    #     batch.
    pi = np.ones(len(game.action_space)) / len(game.action_space)
    batch = [(state, pi, 0.0) for _ in range(32)]
    loss = est.train_step(batch, return_summary=True)
    assert math.isfinite(loss)
