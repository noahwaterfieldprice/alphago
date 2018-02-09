import pytest

from alphago import noughts_and_crosses as nac

terminal_states = [
    (1, 1, 1, 1, 0, 1, 0, 0, 0),  # 1s top line
    (1, 0, 1, 1, 0, 0, 1, 0, 0),  # 1s left side
    (1, 0, 0, 0, 1, 1, 0, 1, 1),  # 1s negative diagonal
    (1, 1, 1, 1, 0, 0, 1, 0, 0),  # 1s top line and left side
    (1, 0, 1, 0, 1, 0, 1, 0, 1)   # 1s both diagonals
]


def test_abstract_state():
    abstract_state = nac.AbstractState

    assert abstract_state.initial_state == None  # TODO: Maybe this is a bad idea

    with pytest.raises(NotImplementedError):
        abstract_state.vector()

    with pytest.raises(NotImplementedError):
        abstract_state.available_actions()

    with pytest.raises(NotImplementedError):
        abstract_state.next_state()

    with pytest.raises(NotImplementedError):
        abstract_state.is_terminal()

    with pytest.raises(NotImplementedError):
        abstract_state.utility()


def test_noughts_and_crosses_initial_state():
    assert nac.initial_state == (None, ) * 9


@pytest.mark.parametrize("terminal_state", terminal_states)
def test_is_terminal_returns_true_for_terminal_states(terminal_state):
    assert nac.NoughtsAndCrosses.is_terminal(terminal_state)
