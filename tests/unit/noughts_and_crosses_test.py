import pytest

from alphago import noughts_and_crosses as nac


def test_abstract_state():
    abstract_state = nac.AbstractState

    assert abstract_state.initial_state == None  # TODO: Maybe this is a bad idea

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
