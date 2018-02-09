import pytest

from alphago import noughts_and_crosses


def test_abstract_state():
    abstract_state = noughts_and_crosses.AbstractState

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
