import pytest
from alphago.noughts_and_crosses import NoughtsAndCrossesState


def test_can_create_initial_state():
    assert NoughtsAndCrossesState.initial_state == (None, ) * 9


def test_correctly_identifies_terminal_state():
    terminal_state = (1, 1, 1, 0, 1, 0, 1, 0, 0)

    assert NoughtsAndCrossesState.is_terminal(terminal_state)
