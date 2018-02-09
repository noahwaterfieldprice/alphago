import numpy as np
import pytest

from alphago.noughts_and_crosses import NoughtsAndCrossesState


def test_can_create_initial_state():
    assert NoughtsAndCrossesState.initial_state == (np.nan, ) * 9


def test_correctly_identifies_state_terminality():
    terminal_state = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    non_terminal_state = (1, -1, np.nan, np.nan, 1, -1, np.nan, -1, 1)

    assert NoughtsAndCrossesState.is_terminal(terminal_state)
    assert not NoughtsAndCrossesState.is_terminal(non_terminal_state)


def test_exception_raised_when_utility_called_for_non_terminal_state():
    non_terminal_state = (1, -1, np.nan, np.nan, 1, -1, np.nan, -1, 1)

    with pytest.raises(ValueError):
        NoughtsAndCrossesState.utility(non_terminal_state)


def test_calculating_utility():
    terminal_state_p1_wins = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    terminal_state_p2_wins = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    pass
