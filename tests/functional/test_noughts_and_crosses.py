import pytest
from alphago import noughts_and_crosses


def test_can_create_initial_state():

    assert noughts_and_crosses.initial_state == (None, ) * 9

