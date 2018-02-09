import numpy as np
import pytest

from alphago import noughts_and_crosses as nac

terminal_states = [
    (1, 1, 1, 1, -1, -1, -1, -1, 1),  # 1s top line
    (1, -1, -1, -1, 1, 1, -1, 1, 1),  # 1s negative diagonal
    (1, 1, 1, 1, -1, -1, 1, -1, -1),  # 1s top line and left side
    (1, -1, 1, -1, 1, -1, 1, -1, 1),  # 1s both diagonals
    (-1, 1, np.nan, -1, 1, 1, -1, -1, 1),  # -1s left side
    (1, np.nan, -1, -1, -1, 1, -1, 1, 1),  # -1s positive diagonal
    (1, 1, -1, -1, -1, 1, 1, -1, 1),  # draw
]

outcomes = ([nac.Outcome(1, -1)] * 4 +
            [nac.Outcome(-1, 1)] * 2 +
            [nac.Outcome(0, 0)])

line_sums_list = [
    (3, -1, -1, 1, -1, 1, 1, -1),
    (-1, 1, 1, -1, 1, 1, 3, -1),
    (3, -1, -1, 3, -1, -1, -1, 1),
    (1, -1, 1, 1, -1, 1, 3, 3),
    (0, 1, -1, -3, 1, 2, 1, 0),
    (0, -1, 1, -1, 0, 1, 1, -3),
    (1, -1, 1, 1, -1, 1, 1, -1),
]

non_terminal_states = [
    (1, -1, np.nan, np.nan, 1, np.nan, 1, np.nan, -1),
    (np.nan, 1, -1, -1, 1, -1, 1, np.nan, 1),
    (1, np.nan, 1, np.nan, -1, np.nan, 1, np.nan, -1),
]


def test_noughts_and_crosses_initial_state():
    assert nac.INITIAL_STATE == (np.nan,) * 9


@pytest.mark.parametrize("state", terminal_states)
def test_is_terminal_returns_true_for_terminal_states(state):
    assert nac.is_terminal(state)


@pytest.mark.parametrize("state, outcome", zip(terminal_states, outcomes))
def test_utility_function_returns_correct_outcomes(state, outcome):
    assert nac.utility(state) == outcome


@pytest.mark.parametrize("state, line_sums",
                         zip(terminal_states, line_sums_list))
def test_line_sums_are_calculated_correctly(state, line_sums):
    assert tuple(nac._calculate_line_sums(state)) == line_sums


@pytest.mark.parametrize("state", non_terminal_states)
def test_utility_raises_exception_on_non_terminal_input_state(state):
    with pytest.raises(ValueError) as exception_info:
        nac.utility(state)
    assert str(exception_info.value) == ("Utility can not be calculated "
                                         "for a non-terminal state.")
