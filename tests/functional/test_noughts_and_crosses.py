import alphago.games.noughts_and_crosses_mxn as nacmn
import numpy as np
import pytest

import alphago.games.noughts_and_crosses as nac


class TestBasic3x3NoughtsAndCrosses:

    terminal_state = (1, 1, 1, -1, 1, -1, 1, -1, -1)
    non_terminal_state = (1, -1, np.nan, np.nan, -1, 1, np.nan, 1, -1)

    def test_can_create_initial_state(self):
        assert nac.INITIAL_STATE == (np.nan,) * 9

    def test_correctly_identifies_state_terminality(self):
        assert nac.is_terminal(self.terminal_state)
        assert not nac.is_terminal(self.non_terminal_state)

    def test_exception_raised_when_utility_called_for_non_terminal_state(self):
        with pytest.raises(ValueError) as exception_info:
            nac.utility(self.non_terminal_state)

        assert str(exception_info.value) == ("Utility can not be calculated "
                                             "for a non-terminal state.")

    def test_calculating_utility_of_terminal_state(self):
        # player 1 wins
        assert nac.utility(self.terminal_state) == {1: 1, 2: -1}
        # draw
        terminal_state_draw = (1, 1, -1, -1, -1, 1, 1, -1, 1)
        assert nac.utility(terminal_state_draw) == {1: 0, 2: 0}

    def test_generating_possible_next_states(self):
        penultimate_state = (1, 1, np.nan, -1, 1, -1, 1, -1, -1)

        expected_next_states = {(0, 2): self.terminal_state}

        assert nac.compute_next_states(penultimate_state) == expected_next_states

    def test_displaying_a_game_in_ascii_format(self, capsys):
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
        nac_4x7 = nacmn.NoughtsAndCrosses(4, 7)
        assert nac_4x7.rows == 4
        assert nac_4x7.columns == 7
        assert nac_4x7.initial_state == (np.nan,) * 4 * 7
