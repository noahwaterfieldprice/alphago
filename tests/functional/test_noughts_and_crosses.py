import itertools

import pytest

from alphago.games import NoughtsAndCrosses, UltimateNoughtsAndCrosses
from alphago.games.noughts_and_crosses import (GameState, Action, UltimateAction,
                                               UltimateGameState)


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

        assert str(exception_info.value) == ("Utility can not be calculated "
                                             "for a non-terminal state.")

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

        assert str(exception_info.value) == ("Legal actions can not be computed "
                                             "for a terminal state.")

    def test_computing_possible_legal_actions(self):
        nac = NoughtsAndCrosses()
        expected_next_states = {Action(2, 0): self.terminal_state}

        assert nac.legal_actions(self.penultimate_state) == expected_next_states

    def test_displaying_a_game_in_ascii_format(self, capsys):
        nac = NoughtsAndCrosses()
        expected_output = (" o | x |   \n"
                           "---+---+---\n"
                           " x | o |   \n"
                           "---+---+---\n"
                           "   | o | x \n")

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


@pytest.mark.skip(reason="Mid migration to using a bit board.")
class TestUltimateNoughtsAndCrosses:

    initial_state = UltimateGameState(last_sub_action=(0, 0),
                                      board=(0,) * 81)
    initial_metaboard = (0,) * 9

    non_terminal_state = UltimateGameState(
        last_sub_action=(2, 2),
        board=(
            0, 1, 0, 0, -1, 0, -1, -1, 1,
            0, 1, 0, 0, 1, 0, 0, -1, 0,
            -1, 1, 0, 0, -1, 1, 0, 1, -1,
            -1, 0, 1, 0, 0, -1, 1, 0, 0,
            -1, 0, 0, 0, 1, -1, 1, 0, 0,
            -1, 0, 0, 1, 0, -1, 1, 0, 0,
            1, 1, 1, 0, 0, -1, -1, 0, 0,
            0, 0, 0, 1, 0, -1, -1, 1, 0,
            -1, 0, -1, 0, 0, 0, 0, 0, 1)
    )
    non_terminal_state_meta_board = (1, 0, -1, -1, -1, 1, 1, 0, 0)

    terminal_state = UltimateGameState(
        last_sub_action=(0, 2),
        board=(
            1, 0, 1, -1, 0, 0, 0, 1, 0,
            -1, 1, 1, 1, -1, 0, 0, -1, 0,
            1, 0, -1, 0, 0, -1, 0, 1, 0,
            0, 1, 0, 1, 1, 1, 0, 0, -1,
            0, -1, 0, -1, -1, 1, 0, -1, 0,
            -1, 1, 0, 0, 0, -1, -1, 0, 0,
            0, -1, 0, -1, -1, -1, 0, 1, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0)
    )
    terminal_state_meta_board = (1, -1, 0, 0, 1, -1, 0, -1, 1)

    terminal_state_draw = UltimateGameState(
        last_sub_action=(0, 1),
        board=(
            1, -1, 1, 0, 1, -1, 1, -1, -1,
            0, -1, 1, 1, 1, 1, 0, 1, -1,
            -1, 0, 1, -1, 1, 0, 0, 0, -1,
            -1, -1, -1, -1, 0, 0, 0, 1, 1,
            1, 1, -1, -1, 1, 0, -1, 1, -1,
            1, -1, 0, -1, 0, 0, 0, 1, 0,
            0, 1, 1, 0, -1, 1, 0, 0, 0,
            1, 1, 0, 0, -1, 0, -1, -1, -1,
            -1, 1, -1, 0, -1, 1, 0, 0, 1)
    )
    terminal_state_draw_meta_board = (1, 1, -1, -1, -1, 1, 1, -1, -1)

    states = (initial_state, non_terminal_state, terminal_state, terminal_state_draw)
    meta_boards = (initial_metaboard, non_terminal_state_meta_board,
                   terminal_state_meta_board, terminal_state_draw_meta_board)

    def test_initial_state_is_correct(self):
        unac = UltimateNoughtsAndCrosses()
        assert unac.initial_state == self.initial_state

    @pytest.mark.parametrize("state, terminality", zip(states, [False, False, True, True]))
    def test_correctly_identifies_state_terminality(self, state, terminality):
        unac = UltimateNoughtsAndCrosses()

        assert unac.is_terminal(state) is terminality

    @pytest.mark.parametrize("state, metaboard", zip(states, meta_boards))
    def test_meta_board_is_calculated_correctly(self, state, metaboard):
        unac = UltimateNoughtsAndCrosses()

        assert unac._compute_meta_board(state) == metaboard

    def test_exception_raised_when_utility_called_non_terminal_state(self):
        unac = UltimateNoughtsAndCrosses()
        with pytest.raises(ValueError) as exception_info:
            unac.utility(self.non_terminal_state)

        assert str(exception_info.value) == ("Utility can not be calculated "
                                             "for a non-terminal state.")

    def test_calculating_utility_of_terminal_state(self):
        unac = UltimateNoughtsAndCrosses()
        # player 1 wins
        assert unac.utility(self.terminal_state) == {1: 1, 2: -1}
        # draw
        assert unac.utility(self.terminal_state_draw) == {1: 0, 2: 0}

    def test_which_player_returns_correct_player(self):
        unac = UltimateNoughtsAndCrosses()

        assert unac.which_player(self.non_terminal_state) == 1

    def test_generating_next_possible_states(self):
        unac = UltimateNoughtsAndCrosses()
        sub_game_possible_actions = (0, 1), (0, 2), (1, 2), (2, 0), (2, 1)
        possible_actions = (UltimateAction(sub_board=(2, 2), sub_action=action)
                            for action in sub_game_possible_actions)
        expected_next_states = {}
        for action in possible_actions:
            (sub_board_row, sub_board_col), (sub_row, sub_col) = action
            board_index = sub_board_row * 27 + sub_board_col * 3 + sub_row * 9 + sub_col
            next_board = list(self.non_terminal_state.board)
            next_board[board_index] = 1
            expected_next_states[action] = UltimateGameState(last_sub_action=action.sub_action,
                                                             board=tuple(next_board))

        assert unac.compute_next_states(self.non_terminal_state) == expected_next_states
