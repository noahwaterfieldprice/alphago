import numpy as np
import pytest

from alphago.games import connect_four as cf


def test_connect_four_initial_state():
    assert np.shape(cf.INITIAL_STATE) == (42,)
    assert np.isnan(cf.INITIAL_STATE).all()


def test_compute_next_states_on_full_column():
    # Full first column
    state = tuple(1 if i % 7 == 0 else np.nan for i in range(42))

    expected_next_actions = list(range(1, 7))
    computed = cf.compute_next_states(state)
    assert expected_next_actions == list(computed.keys())


SUB_GRIDS = [(1, 1, 0, 0, 0, 1, -1, 0, np.nan, 1, 0, 1, 1, np.nan, 0, 1),
             (np.nan,) * 16]

EXPECTED_LINE_SUMS = [
    [2, 0, 2, 2, 2, 3, -1, 2, 3, 1],
    [0 for i in range(10)]
]


@pytest.mark.parametrize("grid, expected_line_sums", zip(SUB_GRIDS,
                                                         EXPECTED_LINE_SUMS))
def test_connect_four_line_sums_4_by_4(grid, expected_line_sums):
    line_sums = cf._calculate_line_sums_4_by_4(grid)
    assert (line_sums == expected_line_sums).all()


TERMINAL_STATES = [
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


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_is_terminal_returns_true_for_terminal_states(state):
    assert cf.is_terminal(state) is True


STATES = [
    (-1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, -1.0, np.nan, np.nan, np.nan, np.nan, 1.0,
     np.nan, np.nan, -1.0, np.nan, np.nan, np.nan, np.nan,
     1.0, np.nan, np.nan, -1.0, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan)]

div = "---+---+---+---+---+---+---"
# additional newline character accounts for the one added to the output
# by the print function itself
OUTPUTS = [
    "\n".join((" o |   |   |   |   |   |   ", div,
               "   | o |   |   |   |   | x ", div,
               "   |   | o |   |   |   |   ", div,
               " x |   |   | o |   |   |   ", div,
               "   |   |   |   |   |   |   ", div,
               "   |   |   |   |   | x |   ")) + "\n",
]


@pytest.mark.parametrize("state, expected_output", zip(STATES, OUTPUTS))
def test_display_function_outputs_correct_strings(state, expected_output, capsys):
    cf.display(state)
    output = capsys.readouterr().out
    assert output == expected_output
