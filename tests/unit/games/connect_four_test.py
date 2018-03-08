import numpy as np
import pytest

from alphago.games import connect_four as cf


def test_connect_four_initial_state():
    assert np.shape(cf.INITIAL_STATE) == (6, 7)
    assert np.isnan(cf.INITIAL_STATE).all()


grids_4_4 = [
    np.array([[1, 1, 0, 0],
              [0, 1, -1, 0],
              [np.nan, 1, 0, 1],
              [1, np.nan, 0, 1]]),
    np.full((4, 4), np.nan)
]

expected_line_sums = [
    [2, 0, 2, 2, 2, 3, -1, 2, 3, 1],
    [0 for i in range(10)]
]


@pytest.mark.parametrize("grid, expected_line_sums", zip(grids_4_4,
                         expected_line_sums))
def test_connect_four_line_sums_4_by_4(grid, expected_line_sums):
    line_sums = cf._calculate_line_sums_4_by_4(grid)
    assert (line_sums == expected_line_sums).all()


terminal_states = [
    np.full((6, 7), 1),      # All 1s
    np.full((6, 7), -1),     # All -1s
    np.array([(-1)**i for i in range(6*7)]).reshape(6, 7),
    np.array([[np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan],
              [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan],
              [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan],
              [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
    np.array([[-1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, -1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, -1.0, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, -1.0, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
    np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan]]),
]


@pytest.mark.parametrize("state", terminal_states)
def test_is_terminal_returns_true_for_terminal_states(state):
    assert cf.is_terminal(state) is True
