import numpy as np
import pytest

from alphago.games.noughts_and_crosses import (NoughtsAndCrosses, UltimateNoughtsAndCrosses,
                                               UltimateGameState, UltimateAction)
from .constants import (terminal_states, terminal_line_sums_arrays, outcomes,
                        non_terminal_states, non_terminal_line_sums_arrays,
                        expected_next_states_list)


class TestMByNNoughtsAndCrosses:
    sizes = [(3, 3), (3, 5), (4, 7), (5, 6), (8, 8), (9, 9)]

    @pytest.mark.parametrize("size", sizes)
    def test_noughts_and_crosses_instances_have_correct_size(self, size, mocker):
        mock_game = mocker.MagicMock()
        rows, columns = size
        NoughtsAndCrosses.__init__(mock_game, *size)
        assert mock_game.rows == rows
        assert mock_game.columns == columns

    @pytest.mark.parametrize("size", sizes)
    def test_initial_state_is_correct(self, size, mocker):
        rows, columns = size
        mock_game = mocker.MagicMock()
        NoughtsAndCrosses.__init__(mock_game, *size)
        assert mock_game.initial_state == (0,) * rows * columns

    @pytest.mark.parametrize("size, state, line_sums",
                             zip(sizes, terminal_states, terminal_line_sums_arrays))
    def test_line_sums_are_calculated_correctly(self, size, state, line_sums, mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        calculated_line_sums = NoughtsAndCrosses._calculate_line_sums(mock_game, state)

        np.testing.assert_array_equal(np.concatenate(calculated_line_sums),
                                      np.concatenate(line_sums))

    @pytest.mark.parametrize("size, state, line_sums",
                             zip(sizes, terminal_states, terminal_line_sums_arrays))
    def test_is_terminal_returns_true_for_terminal_states(self, size, state,
                                                          line_sums, mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        mock_calculate_line_sums = mocker.MagicMock(return_value=line_sums)
        mock_game._calculate_line_sums = mock_calculate_line_sums

        assert NoughtsAndCrosses.is_terminal(mock_game, state) is True

    @pytest.mark.parametrize("size, state, line_sums",
                             zip(sizes, non_terminal_states, non_terminal_line_sums_arrays))
    def test_is_terminal_returns_false_for_non_terminal_states(self, size, state, line_sums,
                                                               mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        mock_game._calculate_line_sums = mocker.MagicMock(return_value=line_sums)
        assert NoughtsAndCrosses.is_terminal(mock_game, state) is False

    players = (2, 1, 1, 2, 2, 1)

    @pytest.mark.parametrize("player, state",
                             zip(players, non_terminal_states))
    def test_which_player_returns_correct_player(self, player, state, mocker):
        mock_game = mocker.MagicMock()
        assert NoughtsAndCrosses.which_player(mock_game, state) == player

    @pytest.mark.parametrize("size, state, line_sums",
                             zip(sizes, non_terminal_states, non_terminal_line_sums_arrays))
    def test_utility_raises_exception_on_non_terminal_input_state(self, size, state, line_sums,
                                                                  mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        mock_game._calculate_line_sums = mocker.MagicMock(return_value=line_sums)
        with pytest.raises(ValueError) as exception_info:
            NoughtsAndCrosses.utility(mock_game, state)
        assert str(exception_info.value) == ("Utility can not be calculated "
                                             "for a non-terminal state.")

    @pytest.mark.parametrize("size, state, line_sums, outcome",
                             zip(sizes, terminal_states, terminal_line_sums_arrays, outcomes))
    def test_utility_function_returns_correct_outcomes(self, size, state, line_sums, outcome,
                                                       mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        mock_game._calculate_line_sums = mocker.MagicMock(return_value=line_sums)

        assert NoughtsAndCrosses.utility(mock_game, state) == outcome

    def test_next_state_raises_exception_on_terminal_input_state(self, mocker):
        mock_game = mocker.MagicMock()
        mock_game.is_terminal = mocker.MagicMock(return_value=True)
        mock_state = mocker.MagicMock()
        with pytest.raises(ValueError) as exception_info:
            NoughtsAndCrosses.compute_next_states(mock_game, mock_state)
        assert str(exception_info.value) == ("Next states can not be generated "
                                             "for a terminal state.")

    @pytest.mark.parametrize("size, state, player, expected_states",
                             zip(sizes, non_terminal_states, players, expected_next_states_list))
    def test_generating_a_dict_of_all_possible_next_states(self, size, state, player,
                                                           expected_states, mocker):
        mock_game = mocker.MagicMock()
        mock_game.rows, mock_game.columns = size
        mock_game.is_terminal = mocker.MagicMock(return_value=False)
        mock_game.which_player = mocker.MagicMock(return_value=player)

        assert NoughtsAndCrosses.compute_next_states(mock_game, state) == expected_states

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
    def test_display_function_outputs_correct_string_for_3x3(self, state, expected_output,
                                                             capsys, mocker):
        mock_game = mocker.MagicMock(rows=3, columns=3)
        NoughtsAndCrosses.display(mock_game, state)

        output = capsys.readouterr().out
        assert output == expected_output


class TestUltimateNoughtsAndCrosses:
    def test_initial_state_is_correct(self, mocker):
        mock_game = mocker.MagicMock()
        UltimateNoughtsAndCrosses.__init__(mock_game)

        assert mock_game.initial_state == UltimateGameState(last_action=(0, 0), board=(0,) * 81)

    terminal_board = [0] * 81
    terminal_board[18:27] = [1] * 9
    terminal_board[54:56] = [-1] * 2
    terminal_board[57:63] = [-1] * 6
    terminal_state1 = UltimateGameState(last_action=(0, 0), board=tuple(terminal_board))
    meta_board1 = (1, 1, 1, 0, 0, 0, 0, -1, -1)
    utilities1 = ({1: 1, 2: -1}, {1: 1, 2: -1}, {1: 1, 2: -1},
                  ValueError, ValueError, ValueError,
                  ValueError, {1: -1, 2: 1}, {1: -1, 2: 1})

    terminal_state2 = UltimateGameState(
        last_action=(0, 2),
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
    meta_board2 = (1, -1, 0, 0, 1, -1, 0, -1, 1)
    utilities2 = ({1: 1, 2: -1}, {1: -1, 2: 1}, ValueError,
                  ValueError, {1: 1, 2: -1}, {1: -1, 2: 1},
                  ValueError, {1: -1, 2: 1}, {1: 1, 2: -1})

    terminal_state3 = UltimateGameState(
        last_action=(0, 0),
        board=(
            0, -1, 0, -1, 0, 1, -1, 0, 0,
            1, 0, 1, 0, 1, 0, -1, 1, 0,
            0, 1, 1, 1, 0, 0, 1, 0, 0,
            -1, 0, 0, 0, 1, 0, 0, -1, 1,
            -1, 0, 1, 0, 1, 0, 0, 0, 0,
            -1, 0, 1, -1, 1, -1, -1, 0, 0,
            1, -1, 0, 1, 0, -1, 0, 0, -1,
            0, -1, 0, 1, 0, 0, 0, -1, -1,
            0, -1, 0, 1, -1, 0, 0, 0, 1)
    )
    meta_board3 = (0, 1, 0, -1, 1, 0, -1, 1, 0)
    utilities3 = (ValueError, {1: 1, 2: -1}, ValueError,
                  {1: -1, 2: 1}, {1: 1, 2: -1}, ValueError,
                  {1: -1, 2: 1}, {1: 1, 2: -1}, ValueError)

    terminal_state4 = UltimateGameState(
        last_action=(0, 1),
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
    meta_board4 = (1, 1, -1, -1, -1, 1, 1, -1, -1)
    utilities4 = ({1: 1, 2: -1}, {1: 1, 2: -1}, {1: -1, 2: 1},
                  {1: -1, 2: 1}, {1: -1, 2: 1}, {1: 1, 2: -1},
                  {1: 1, 2: -1}, {1: -1, 2: 1}, {1: -1, 2: 1})

    terminal_states = (terminal_state1, terminal_state2, terminal_state3, terminal_state4)
    meta_boards = (meta_board1, meta_board2, meta_board3, meta_board4)
    utilities_list = (utilities1, utilities2, utilities3, utilities4)

    @pytest.mark.parametrize("state, meta_board, utilities",
                             zip(terminal_states, meta_boards, utilities_list))
    def test_meta_board_is_calculated_correctly(self, state, meta_board, utilities, mocker):
        mock_utility = mocker.MagicMock(side_effect=utilities)
        mock_nac = mocker.MagicMock(utility=mock_utility)
        mock_game = mocker.MagicMock(sub_game=mock_nac)
        # split state into sub boards
        board = np.array(state.board).reshape(9, 9)
        sub_boards = [board[i * 3:(i + 1) * 3, j * 3:(j * 3) + 3]
                      for i in range(3) for j in range(3)]

        expected_calls = [mocker.call(tuple(sub_board.ravel()))
                          for sub_board in sub_boards]

        assert UltimateNoughtsAndCrosses._compute_meta_board(mock_game, state) == meta_board
        mock_utility.assert_has_calls(expected_calls)

    def test_meta_board_delegates_to_sub_game_is_terminal_method(self, mocker):
        mock_is_terminal = mocker.MagicMock()
        mock_game = mocker.MagicMock(
            sub_game=mocker.MagicMock(is_terminal=mock_is_terminal),
            _compute_meta_board=mocker.MagicMock(return_value="some_meta_board"))
        mock_state = mocker.MagicMock()
        UltimateNoughtsAndCrosses.is_terminal(mock_game, mock_state)
        mock_game._compute_meta_board.assert_called_once_with(mock_state)
        mock_is_terminal.assert_called_once_with("some_meta_board")

    def test_meta_board_delegates_to_sub_game_utility_method(self, mocker):
        mock_utility = mocker.MagicMock()
        mock_game = mocker.MagicMock(
            sub_game=mocker.MagicMock(utility=mock_utility),
            _compute_meta_board=mocker.MagicMock(return_value="some_meta_board"))
        mock_state = mocker.MagicMock()
        UltimateNoughtsAndCrosses.utility(mock_game, mock_state)

        mock_game._compute_meta_board.assert_called_once_with(mock_state)
        mock_utility.assert_called_once_with("some_meta_board")

    def test_which_player_delegates_to_sub_game_which_player(self, mocker):
        mock_which_player = mocker.MagicMock()
        mock_game = mocker.MagicMock(sub_game=mocker.MagicMock(which_player=mock_which_player))
        mock_state = mocker.MagicMock()
        UltimateNoughtsAndCrosses.which_player(mock_game, mock_state)
        mock_which_player.assert_called_once_with(mock_state.board)
