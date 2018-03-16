import numpy as np
import pytest

from alphago.games import connect_four as cf


def test_connect_four_initial_state():
    assert np.shape(cf.INITIAL_STATE) == (42,)
    assert np.isnan(cf.INITIAL_STATE).all()


sub_grids = [(1, 1, 0, 0, 0, 1, -1, 0, np.nan, 1, 0, 1, 1, np.nan, 0, 1),
             (np.nan,) * 16]

expected_line_sums = [
    [2, 0, 2, 2, 2, 3, -1, 2, 3, 1],
    [0 for i in range(10)]
]


@pytest.mark.parametrize("grid, expected_line_sums", zip(sub_grids,
                         expected_line_sums))
def test_connect_four_line_sums_4_by_4(grid, expected_line_sums):
    line_sums = cf._calculate_line_sums_4_by_4(grid)
    assert (line_sums == expected_line_sums).all()


terminal_states = [
    (1,) * 42,      # All 1s
    (-1,) * 42,     # All -1s
    tuple(-1 ** i for i in range(42)),

    (np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan,
     np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan,
     np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan,
     np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),

    (-1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, -1.0, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, -1.0, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, -1.0, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),

    (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan),
]


@pytest.mark.parametrize("state", terminal_states)
def test_is_terminal_returns_true_for_terminal_states(state):
    assert cf.is_terminal(state) is True


states = [
    (-1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, -1.0, np.nan, np.nan, np.nan, np.nan, 1.0,
     np.nan, np.nan, -1.0, np.nan, np.nan, np.nan, np.nan,
     1.0, np.nan, np.nan, -1.0, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan)]

div = "---+---+---+---+---+---+---"
# additional newline character accounts for the one added to the output
# by the print function itself
outputs = [
    "\n".join((" o |   |   |   |   |   |   ", div,
               "   | o |   |   |   |   | x ", div,
               "   |   | o |   |   |   |   ", div,
               " x |   |   | o |   |   |   ", div,
               "   |   |   |   |   |   |   ", div,
               "   |   |   |   |   | x |   ")) + "\n",
]


@pytest.mark.parametrize("state, expected_output", zip(states, outputs))
def test_display_function_outputs_correct_strings(state, expected_output, capsys):
    cf.display(state)
    output = capsys.readouterr().out
    assert output == expected_output
