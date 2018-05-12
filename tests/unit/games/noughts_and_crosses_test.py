import pytest

from alphago.games import NoughtsAndCrosses


# TODO: mock this file


def test_noughts_and_crosses_initial_state():
    nac = NoughtsAndCrosses()
    assert nac.initial_state == (0,) * 9


terminal_states = [
    (1, 1, 1, 1, -1, -1, -1, -1, 1),  # 1s top line
    (1, -1, -1, -1, 1, 1, -1, 1, 1),  # 1s negative diagonal
    (1, 1, 1, 1, -1, -1, 1, -1, -1),  # 1s top line and left side
    (1, -1, 1, -1, 1, -1, 1, -1, 1),  # 1s both diagonals
    (-1, 1, 0, -1, 1, 1, -1, -1, 1),  # -1s left side
    (1, 0, -1, -1, -1, 1, -1, 1, 1),  # -1s positive diagonal
    (1, 1, -1, -1, -1, 1, 1, -1, 1),  # draw
]


@pytest.mark.parametrize("state", terminal_states)
def test_is_terminal_returns_true_for_terminal_states(state):
    nac = NoughtsAndCrosses()
    assert nac.is_terminal(state) is True


non_terminal_states = [
    (1, -1, 0, 0, 1, 0, 1, 0, -1),
    (0, 1, -1, -1, 1, -1, 1, 0, 1),
    (1, 0, 1, 0, -1, 0, 1, 0, -1),
]


@pytest.mark.parametrize("state", non_terminal_states)
def test_is_terminal_returns_false_for_non_terminal_states(state):
    nac = NoughtsAndCrosses()
    assert nac.is_terminal(state) is False


players = (1, 2, 2, 2)


@pytest.mark.parametrize("player, state",
                         zip(players, [(0,) * 9] + non_terminal_states))
def test_which_player_returns_correct_player(player, state, mocker):
    mock = mocker.MagicMock()
    assert NoughtsAndCrosses.which_player(mock, state) == player


line_sums_list = [
    (3, -1, -1, 1, -1, 1, 1, -1),
    (-1, 1, 1, -1, 1, 1, 3, -1),
    (3, -1, -1, 3, -1, -1, -1, 1),
    (1, -1, 1, 1, -1, 1, 3, 3),
    (0, 1, -1, -3, 1, 2, 1, 0),
    (0, -1, 1, -1, 0, 1, 1, -3),
    (1, -1, 1, 1, -1, 1, 1, -1),
]


@pytest.mark.parametrize("state, line_sums",
                         zip(terminal_states, line_sums_list))
def test_line_sums_are_calculated_correctly(state, line_sums, mocker):
    nac = NoughtsAndCrosses()
    assert tuple(nac._calculate_line_sums(state)) == line_sums


@pytest.mark.parametrize("state", non_terminal_states)
def test_utility_raises_exception_on_non_terminal_input_state(state, mocker):
    nac = NoughtsAndCrosses()
    with pytest.raises(ValueError) as exception_info:
        nac.utility(state)
    assert str(exception_info.value) == ("Utility can not be calculated "
                                         "for a non-terminal state.")


outcomes = ([{1: 1, 2: -1}] * 4 +  # player 1 wins
            [{1: -1, 2: 1}] * 2 +  # player 2 wins
            [{1: 0, 2: 0}])  # draw


@pytest.mark.parametrize("state, outcome", zip(terminal_states, outcomes))
def test_utility_function_returns_correct_outcomes(state, outcome):
    nac = NoughtsAndCrosses()
    assert nac.utility(state) == outcome


@pytest.mark.parametrize("state", terminal_states)
def test_next_state_raises_exception_on_terminal_input_state(state):
    nac = NoughtsAndCrosses()
    with pytest.raises(ValueError) as exception_info:
        nac.compute_next_states(state)
    assert str(exception_info.value) == ("Next states can not be generated "
                                         "for a terminal state.")


states = [
    (1, 0, -1, 0, 1, 0, 1, 0, -1),
    (1, 0, -1, -1, 1, 0, 1, 0, -1),
    (1, 0, -1, -1, 1, 1, 1, 0, -1),
    (1, 0, -1, -1, 1, 1, 1, -1, -1),
]
# player 2s turn
next_states_move6 = {
    (0, 1): (1, -1, -1, 0, 1, 0, 1, 0, -1),
    (1, 0): (1, 0, -1, -1, 1, 0, 1, 0, -1),
    (1, 2): (1, 0, -1, 0, 1, -1, 1, 0, -1),
    (2, 1): (1, 0, -1, 0, 1, 0, 1, -1, -1),
}
# player 2 placed 'o' at (1, 0)
# player 1s turn
next_states_move7 = {
    (0, 1): (1, 1, -1, -1, 1, 0, 1, 0, -1),
    (1, 2): (1, 0, -1, -1, 1, 1, 1, 0, -1),
    (2, 1): (1, 0, -1, -1, 1, 0, 1, 1, -1),
}
# player 1 placed 'x' at (1, 2)
# player 2 to move
next_states_move8 = {
    (0, 1): (1, -1, -1, -1, 1, 1, 1, 0, -1),
    (2, 1): (1, 0, -1, -1, 1, 1, 1, -1, -1),
}
# player 2 placed 'o' at (2, 1)
# player 1 to move
next_states_move9 = {
    (0, 1): (1, 1, -1, -1, 1, 1, 1, -1, -1),
}

expected_next_states_list = (next_states_move6, next_states_move7,
                             next_states_move8, next_states_move9)


@pytest.mark.parametrize("state, expected_next_states",
                         zip(states, expected_next_states_list))
def test_generating_a_dict_of_all_possible_next_states(state,
                                                       expected_next_states):
    nac = NoughtsAndCrosses()
    assert nac.compute_next_states(state) == expected_next_states


states = [
    (0,) * 9,
    (1, -1, 0, 0, 1, 0, 1, 0, -1),
    (1, 1, 1, 1, -1, -1, -1, -1, 1),
]
div = "---+---+---"
# additional newline character accounts for the one added to the output
# by the print function itself
outputs = [
    "\n".join(("   |   |   ", div, "   |   |   ", div, "   |   |   ")) + "\n",
    "\n".join((" x | o |   ", div, "   | x |   ", div, " x |   | o ")) + "\n",
    "\n".join((" x | x | x ", div, " x | o | o ", div, " o | o | x ")) + "\n",
]


@pytest.mark.parametrize("state, expected_output", zip(states, outputs))
def test_display_function_outputs_correct_strings(state, expected_output, capsys):
    NoughtsAndCrosses.display(state)
    output = capsys.readouterr().out
    assert output == expected_output
