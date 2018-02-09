import numpy as np
import pytest

from alphago import noughts_and_crosses as nac

terminal_states = [
    (1, 1, 1, 1, -1, 1, -1, -1, -1),  # 1s top line
    (1, -1, 1, 1, -1, -1, 1, -1, -1),  # 1s left side
    (1, -1, -1, -1, 1, 1, -1, 1, 1),  # 1s negative diagonal
    (1, 1, 1, 1, -1, -1, 1, -1, -1),  # 1s top line and left side
    (1, -1, 1, -1, 1, -1, 1, -1, 1)   # 1s both diagonals
]

def test_abstract_state():
    abstract_state = nac.AbstractState

    assert abstract_state.initial_state is None  # TODO: Maybe this is a bad idea

    with pytest.raises(NotImplementedError):
        abstract_state.vector(None)

    with pytest.raises(NotImplementedError):
        abstract_state.available_actions(None)

    with pytest.raises(NotImplementedError):
        abstract_state.next_state(None, None)

    with pytest.raises(NotImplementedError):
        abstract_state.is_terminal(None)

    with pytest.raises(NotImplementedError):
        abstract_state.utility(None)


def test_noughts_and_crosses_initial_state():
    assert nac.NoughtsAndCrossesState.initial_state == (np.nan, ) * 9


@pytest.mark.parametrize("terminal_state", terminal_states)
def test_is_terminal_returns_true_for_terminal_states(terminal_state):
    assert nac.NoughtsAndCrossesState.is_terminal(terminal_state)
