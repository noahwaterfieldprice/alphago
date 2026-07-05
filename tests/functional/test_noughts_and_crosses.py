import pytest

from alphago.games import NoughtsAndCrosses
from alphago.games.noughts_and_crosses import (
    Action,
    GameState,
)


class TestBasic3x3NoughtsAndCrosses:
    terminal_state = GameState(0b111010100, 0b000101011, 2)
    non_terminal_state = GameState(0b100001010, 0b010010001, 1)
    terminal_state_draw = GameState(0b110001101, 0b001110010, 2)
    penultimate_state = GameState(0b110010100, 0b000101011, 1)

    def test_can_create_initial_state(self):
        nac = NoughtsAndCrosses()
        assert nac.initial_state == GameState(0, 0, 1)

    def test_correctly_identifies_state_terminality(self):
        nac = NoughtsAndCrosses()
        assert nac.is_terminal(self.terminal_state) is True
        assert nac.is_terminal(self.terminal_state_draw) is True
        assert nac.is_terminal(self.non_terminal_state) is False

    def test_exception_raised_when_utility_called_for_non_terminal_state(self):
        nac = NoughtsAndCrosses()
        with pytest.raises(ValueError) as exception_info:
            nac.utility(self.non_terminal_state)

        assert str(exception_info.value) == (
            "Utility can not be calculated for a non-terminal state."
        )

    def test_calculating_utility_of_terminal_state(self):
        nac = NoughtsAndCrosses()
        # player 1 wins
        assert nac.utility(self.terminal_state) == {1: 1, 2: -1}
        # draw

        assert nac.utility(self.terminal_state_draw) == {1: 0, 2: 0}

    def test_current_player_returns_correct_player(self):
        nac = NoughtsAndCrosses()

        assert nac.current_player(self.non_terminal_state) == 1

    def test_exception_raised_when_computing_legal_actions_for_terminal_state(self):
        nac = NoughtsAndCrosses()
        with pytest.raises(ValueError) as exception_info:
            nac.legal_actions(self.terminal_state)

        assert str(exception_info.value) == (
            "Legal actions can not be computed for a terminal state."
        )

    def test_computing_possible_legal_actions(self):
        nac = NoughtsAndCrosses()
        expected_next_states = {Action(2, 0): self.terminal_state}

        assert nac.legal_actions(self.penultimate_state) == expected_next_states

    def test_displaying_a_game_in_ascii_format(self, capsys):
        nac = NoughtsAndCrosses()
        expected_output = (
            " o | x |   \n---+---+---\n x | o |   \n---+---+---\n   | o | x \n"
        )

        nac.display(self.non_terminal_state)
        output = capsys.readouterr().out
        assert output == expected_output


class TestMxNNoughtsAndCrosses:
    # terminal state for 4x7 game - O's 2nd major diagonal
    terminal_state = GameState(31457280, 33686018, 1)
    # non-terminal state for 4x7 game - X played top left
    non_terminal_state = GameState(1, 0, 2)

    def test_can_create_instance_of_mxn_game(self):
        nac_4x7 = NoughtsAndCrosses(rows=4, columns=7)
        assert nac_4x7.rows == 4
        assert nac_4x7.columns == 7
        assert nac_4x7.initial_state == GameState(0, 0, 1)

    def test_correctly_identifies_state_terminality(self):
        nac_4x7 = NoughtsAndCrosses(rows=4, columns=7)
        assert nac_4x7.is_terminal(self.terminal_state) is True
        assert nac_4x7.is_terminal(self.non_terminal_state) is False
