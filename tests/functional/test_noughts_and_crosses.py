import numpy as np
import pytest

import alphago.noughts_and_crosses as nac


def test_can_create_initial_state():
    assert nac.INITIAL_STATE == (np.nan,) * 9


def test_correctly_identifies_state_terminality():
    terminal_state = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    non_terminal_state = (1, -1, np.nan, np.nan, -1, 1, np.nan, 1, -1)

    assert nac.is_terminal(terminal_state)
    assert not nac.is_terminal(non_terminal_state)


def test_exception_raised_when_utility_called_for_non_terminal_state():
    non_terminal_state = (1, -1, np.nan, np.nan, -1, 1, np.nan, 1, -1)

    with pytest.raises(ValueError) as exception_info:
        nac.utility(non_terminal_state)

    assert str(exception_info.value) == ("Utility can not be calculated "
                                         "for a non-terminal state.")


def test_calculating_utility_of_terminal_state():
    terminal_state_p1_wins = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    terminal_state_p2_wins = (-1, -1, -1, 1, -1, 1, -1, 1, 1)
    terminal_state_draw = (1, 1, -1, -1, -1, 1, 1, -1, 1)

    assert nac.utility(terminal_state_p1_wins) == nac.Outcome(1, -1)
    assert nac.utility(terminal_state_p2_wins) == nac.Outcome(-1, 1)
    assert nac.utility(terminal_state_draw) == nac.Outcome(0, 0)
