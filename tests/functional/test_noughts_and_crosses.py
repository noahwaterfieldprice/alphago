import pytest

from alphago.games import NoughtsAndCrosses, UltimateNoughtsAndCrosses


class TestBasic3x3NoughtsAndCrosses:

    terminal_state = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    non_terminal_state = (1, -1, 0, 0, -1, 1, 0, 1, -1)

    def test_can_create_initial_state(self):
        nac = NoughtsAndCrosses()
        assert nac.initial_state == (0,) * 9

    def test_correctly_identifies_state_terminality(self):
        nac = NoughtsAndCrosses()
        assert nac.is_terminal(self.terminal_state)
        assert not nac.is_terminal(self.non_terminal_state)

    def test_exception_raised_when_utility_called_for_non_terminal_state(self):
        nac = NoughtsAndCrosses()
        with pytest.raises(ValueError) as exception_info:
            nac.utility(self.non_terminal_state)

        assert str(exception_info.value) == ("Utility can not be calculated "
                                             "for a non-terminal state.")

    def test_calculating_utility_of_terminal_state(self):
        nac = NoughtsAndCrosses()
        # player 1 wins
        assert nac.utility(self.terminal_state) == {1: 1, 2: -1}
        # draw
        terminal_state_draw = (1, 1, -1, -1, -1, 1, 1, -1, 1)
        assert nac.utility(terminal_state_draw) == {1: 0, 2: 0}

    def test_generating_possible_next_states(self):
        nac = NoughtsAndCrosses()
        penultimate_state = (1, 1, 0, -1, 1, -1, 1, -1, -1)

        expected_next_states = {(0, 2): self.terminal_state}

        assert nac.compute_next_states(penultimate_state) == expected_next_states

    def test_displaying_a_game_in_ascii_format(self, capsys):
        nac = NoughtsAndCrosses()
        expected_output = (" x | o |   \n"
                           "---+---+---\n"
                           "   | o | x \n"
                           "---+---+---\n"
                           "   | x | o \n")

        nac.display(self.non_terminal_state)
        output = capsys.readouterr().out
        assert output == expected_output


class TestMxNNoughtsAndCrosses:

    def test_can_create_instance_of_mxn_game(self):
        nac_4x7 = NoughtsAndCrosses(rows=4, columns=7)
        assert nac_4x7.rows == 4
        assert nac_4x7.columns == 7
        assert nac_4x7.initial_state == (0,) * 4 * 7


class TestUltimateNoughtsAndCrosses:
    def test_initial_state_is_correct(self):
        unac = UltimateNoughtsAndCrosses()
        assert unac.initial_state == (-1,) + (0,) * 81
