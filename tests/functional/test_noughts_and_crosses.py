import numpy as np
import pytest

import alphago.noughts_and_crosses as nac

TERMINAL_STATE = (1, 1, 1, -1, 1, -1, 1, -1, -1)
NON_TERMINAL_STATE = (1, -1, np.nan, np.nan, -1, 1, np.nan, 1, -1)


def test_can_create_initial_state():
    assert nac.INITIAL_STATE == (np.nan,) * 9


def test_correctly_identifies_state_terminality():
    assert nac.is_terminal(TERMINAL_STATE)
    assert not nac.is_terminal(NON_TERMINAL_STATE)


def test_exception_raised_when_utility_called_for_non_terminal_state():
    with pytest.raises(ValueError) as exception_info:
        nac.utility(NON_TERMINAL_STATE)

    assert str(exception_info.value) == ("Utility can not be calculated "
                                         "for a non-terminal state.")


def test_calculating_utility_of_terminal_state():
    # player 1 wins
    assert nac.utility(TERMINAL_STATE) == nac.Outcome(1, -1)
    # draw
    terminal_state_draw = (1, 1, -1, -1, -1, 1, 1, -1, 1)
    assert nac.utility(terminal_state_draw) == nac.Outcome(0, 0)


def test_generating_possible_next_states():
    penultimate_state = (1, 1, np.nan, -1, 1, -1, 1, -1, -1)

    expected_next_states = {(0, 2): TERMINAL_STATE}

    assert nac.next_states(penultimate_state) == expected_next_states


def test_displaying_a_game_in_ascii_format(capsys):
    expected_output = (" x | o |   \n"
                       "---+---+---\n"
                       "   | o | x \n"
                       "---+---+---\n"
                       "   | x | o \n")

    nac.display(NON_TERMINAL_STATE)
    output = capsys.readouterr().out
    assert output == expected_output
