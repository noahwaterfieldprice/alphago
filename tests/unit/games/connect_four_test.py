import numpy as np
import pytest

from alphago.games.connect_four import ConnectFour


def test_connect_four_initial_state(mocker):
    mock = mocker.MagicMock()
    ConnectFour.__init__(mock)

    assert np.shape(mock.initial_state) == (42,)
    assert ~np.any(mock.initial_state)


def test_compute_next_states_on_full_column(mocker):
    # Full first column
    state = tuple(1 if i % 7 == 0 else 0 for i in range(42))

    expected_next_actions = list(range(1, 7))
    mock = mocker.MagicMock()
    computed = ConnectFour.compute_next_states(mock, state)
    assert expected_next_actions == list(computed.keys())


NEXT_STATES_STATES = [
    (0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, -1, 0, 0,
     0, 1, -1, 0, 1, 0, 0)
]


NEXT_NEXT_STATES_STATES = [
    {0: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, -1, 0, 0,
         1, 1, -1, 0, 1, 0, 0),
     1: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, -1, 0, 0,
         0, 1, -1, 0, 1, 0, 0),
     2: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, -1, 0, 0,
         0, 1, -1, 0, 1, 0, 0),
     3: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, -1, 0, 0,
         0, 1, -1, 1, 1, 0, 0),
     4: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, -1, 0, 0,
         0, 1, -1, 0, 1, 0, 0),
     5: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, -1, 0, 0,
         0, 1, -1, 0, 1, 1, 0),
     6: (0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, -1, 0, 0,
         0, 1, -1, 0, 1, 0, 1)}
]


@pytest.mark.parametrize("state, expected_next_states",
                         zip(NEXT_STATES_STATES, NEXT_NEXT_STATES_STATES))
def test_compute_next_states(state, expected_next_states, mocker):
    cf = ConnectFour()
    next_states = cf.compute_next_states(state)
    assert next_states == expected_next_states


SUB_GRIDS = [(1, 1, 0, 0, 0, 1, -1, 0, 0, 1, 0, 1, 1, 0, 0, 1),
             (0,) * 16]

EXPECTED_LINE_SUMS = [
    [2, 0, 2, 2, 2, 3, -1, 2, 3, 1],
    [0 for i in range(10)]
]


@pytest.mark.parametrize("grid, expected_line_sums", zip(SUB_GRIDS,
                                                         EXPECTED_LINE_SUMS))
def test_connect_four_line_sums_4_by_4(grid, expected_line_sums, mocker):
    mock = mocker.MagicMock()
    line_sums = ConnectFour._calculate_line_sums_4_by_4(mock, grid)
    assert np.all(line_sums == expected_line_sums)


TERMINAL_STATES = [
    (1,) * 42,      # All 1s
    (-1,) * 42,     # All -1s
    tuple(-1 ** i for i in range(42)),

    (0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0, 0,
     0, 0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0),

    (-1, 0, 0, 0, 0, 0, 0,
     0, -1, 0, 0, 0, 0, 0,
     0, 0, -1, 0, 0, 0, 0,
     0, 0, 0, -1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0),

    (0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 0, 0, 0),
]


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_is_terminal_returns_true_for_terminal_states(state, mocker):
    cf = ConnectFour()
    assert cf.is_terminal(state)


UTILITY_STATES = [
    (0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0, 0,
     0, 0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0),

    (-1, 0, 0, 0, 0, 0, 0,
     0, -1, 0, 0, 0, 0, 0,
     0, 0, -1, 0, 0, 0, 0,
     0, 0, 0, -1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0),

    (0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 0, 0, 0),
]

EXPECTED_UTILITIES = [
    {1: 1, 2: -1},
    {1: -1, 2: 1},
    {1: 1, 2: -1}
]


@pytest.mark.parametrize("state, expected_utility",
                         zip(UTILITY_STATES, EXPECTED_UTILITIES))
def test_utility(state, expected_utility, mocker):
    cf = ConnectFour()
    assert cf.utility(state) == expected_utility


STATES = [
    (-1, 0, 0, 0, 0, 0, 0,
     0, -1, 0, 0, 0, 0, 1,
     0, 0, -1, 0, 0, 0, 0,
     1, 0, 0, -1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0)]

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
    ConnectFour.display(state)
    output = capsys.readouterr().out
    assert output == expected_output
