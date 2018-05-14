import itertools

import numpy as np
import pytest

from alphago.games import NoughtsAndCrosses, UltimateNoughtsAndCrosses


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

    # terminal state for 3x3 game - -1s minor diagonal
    terminal_state_3x3 = (1, 0, -1, -1, -1, 1, -1, 1, 1)
    # terminal state for 3x5 game - 1s top row
    terminal_state_3x5 = (1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1)
    # terminal state for 4x7 game - -1s 2nd major diagonal
    terminal_state_4x7 = np.zeros((4, 7))
    for i in range(4):
        terminal_state_4x7[i, i + 1] = -1
        terminal_state_4x7[3, :4] = 1
    terminal_state_4x7 = tuple(terminal_state_4x7.ravel())
    # terminal state for 5x6 game - -1s  2nd left column
    terminal_state_5x6 = np.zeros((5, 6))
    terminal_state_5x6[0, 1:] = 1
    terminal_state_5x6[:, 0] = -1
    terminal_state_5x6 = tuple(terminal_state_5x6.ravel())
    # terminal state for 8x8 game - draw
    doubled_even_row = np.array((1, 1, -1, -1) * 2)
    doubled_odd_row = doubled_even_row * -1
    terminal_state_8x8 = np.row_stack((doubled_even_row, doubled_odd_row) * 4)
    terminal_state_8x8 = tuple(terminal_state_8x8.ravel())
    # terminal state for 9x9 game - 1s both diagonals
    even_row = np.array((1, -1) * 4 + (1,))
    odd_row = even_row * -1
    terminal_state_9x9 = np.row_stack((even_row, odd_row) * 4 + (even_row,))
    terminal_state_9x9 = tuple(terminal_state_9x9.ravel())

    terminal_states = (terminal_state_3x3, terminal_state_3x5, terminal_state_4x7,
                       terminal_state_5x6, terminal_state_8x8, terminal_state_9x9)

    terminal_line_sums_list = [
        ((0, -1, 1), (-1, 0, 1), (1, -3)),
        ((5, -1, -3), (1, -1, 1, -1, 1), (1, -1, -1, 1, -1, -1)),
        ((-1, -1, -1, 3), (1, 0, 0, 0, -1, 0, 0), (1, -4, 0, 0, 1, 0, 1, 0)),
        ((4, -1, -1, -1, -1), (-5, 1, 1, 1, 1, 1), (-1, 1, 1, 0)),
        ((0,) * 8, (0,) * 8, (0, 0)),
        ((1, -1, 1, -1, 1, -1, 1, -1, 1), (1, -1, 1, -1, 1, -1, 1, -1, 1), (9, 9)),
    ]

    terminal_line_sums_list_arrays = [
        tuple(np.array(line_sums_i) for line_sums_i in line_sums)
        for line_sums in terminal_line_sums_list]

    @pytest.mark.parametrize("size, state, line_sums",
                             zip(sizes, terminal_states, terminal_line_sums_list_arrays))
    def test_line_sums_are_calculated_correctly(self, size, state, line_sums, mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        calculated_line_sums = NoughtsAndCrosses._calculate_line_sums(mock_game, state)

        np.testing.assert_array_equal(np.concatenate(calculated_line_sums),
                                      np.concatenate(line_sums))

    @pytest.mark.parametrize("size, state, line_sums",
                             zip(sizes, terminal_states, terminal_line_sums_list_arrays))
    def test_is_terminal_returns_true_for_terminal_states(self, size, state,
                                                          line_sums, mocker):
        rows, columns = size
        mock_game = mocker.MagicMock(rows=rows, columns=columns)
        mock_calculate_line_sums = mocker.MagicMock(return_value=line_sums)
        mock_game._calculate_line_sums = mock_calculate_line_sums

        assert NoughtsAndCrosses.is_terminal(mock_game, state) is True

    # non-terminal state for 3x3 game
    non_terminal_state_3x3 = (1, 0, -1, 0, 1, 0, 1, 0, -1)
    # non-terminal state for 3x5 game
    non_terminal_state_3x5 = (1, 1, 1, 1, 0, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1)
    # non-terminal state for 4x7 game - no moves played yet
    non_terminal_state_4x7 = (0,) * 28
    # non-terminal state for 5x6 game - almost all 1s in left column
    non_terminal_state_5x6 = np.zeros((5, 6))
    non_terminal_state_5x6[1:, 0] = 1
    non_terminal_state_5x6[2:, 1] = -1
    non_terminal_state_5x6 = tuple(non_terminal_state_5x6.ravel())
    # non-terminal state for 8x8 game - checkerboard apart from top right corner
    non_terminal_state_8x8 = np.copy(terminal_state_8x8)
    non_terminal_state_8x8[7] = 0
    non_terminal_state_8x8 = tuple(non_terminal_state_8x8.ravel())
    # non-terminal state for 9x9 game - 1s both diagonals except middle
    non_terminal_state_9x9 = np.copy(terminal_state_9x9)
    non_terminal_state_9x9[40] = 0
    non_terminal_state_9x9 = tuple(non_terminal_state_9x9.ravel())

    non_terminal_states = (
        non_terminal_state_3x3, non_terminal_state_3x5, non_terminal_state_4x7,
        non_terminal_state_5x6, non_terminal_state_8x8, non_terminal_state_9x9
    )

    non_terminal_line_sums_list = [
        ((0, 1, 0), (2, 1, -2), (1, 1)),
        ((4, -1, -3), (1, -1, 1, -1, 0), (1, -1, -1, 0, -1, -1)),
        ((0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0)),
        ((0., 1., 0., 0., 0.), (4., -3., 0., 0., 0., 0.), (0., 0., -1., 0.)),
        ((1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 1), (0, 1)),
        ((1, -1, 1, -1, 0, -1, 1, -1, 1), (1, -1, 1, -1, 0, -1, 1, -1, 1), (8, 8))

    ]

    non_terminal_line_sums_arrays = [
        tuple(np.array(line_sums_i) for line_sums_i in line_sums)
        for line_sums in non_terminal_line_sums_list]

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

    outcomes = ([{1: -1, 2: 1},  # player 2 wins
                 {1: 1, 2: -1},  # player 1 wins
                 {1: -1, 2: 1},  # player 1 wins
                 {1: -1, 2: 1},  # player 2 wins
                 {1: 0, 2: 0},  # draw
                 {1: 1, 2: -1}])  # player 1 wins

    @pytest.mark.parametrize("size, state, line_sums, outcome",
                             zip(sizes, terminal_states, terminal_line_sums_list_arrays, outcomes))
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

    states = [
        non_terminal_state_3x3,
        non_terminal_state_3x5,
        non_terminal_state_4x7,
        non_terminal_state_5x6,
        non_terminal_state_8x8,
        non_terminal_state_9x9,
    ]

    # player 2s turn
    next_states_3x3 = {
        (0, 1): (1, -1, -1, 0, 1, 0, 1, 0, -1),
        (1, 0): (1, 0, -1, -1, 1, 0, 1, 0, -1),
        (1, 2): (1, 0, -1, 0, 1, -1, 1, 0, -1),
        (2, 1): (1, 0, -1, 0, 1, 0, 1, -1, -1),
    }
    # player 1s turn - complete a row across first row
    next_states_3x5 = {
        (0, 4): (1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1)
    }
    # player 1s turn - all states possible
    next_states_4x7 = {}
    for action in itertools.product(range(4), range(7)):
        row, col = action
        next_state = list(non_terminal_state_4x7)
        next_state[row * 7 + col] = 1
        next_states_4x7[action] = tuple(next_state)
    # player 2s turn - 1st two columns almost full
    next_states_5x6 = {}
    actions_5x6 = list(itertools.product(range(5), range(2, 6)))
    actions_5x6.extend([(0, 0), (0, 1), (1, 1)])
    for action in actions_5x6:
        row, col = action
        next_state = list(non_terminal_state_5x6)
        next_state[row * 6 + col] = -1
        next_states_5x6[action] = tuple(next_state)
    # player 2s turn - checkerboard except for top right corner
    next_states_8x8 = {(0, 7): terminal_state_8x8}
    # player 1s turn
    next_states_9x9 = {(4, 4): terminal_state_9x9}

    expected_next_states_list = (next_states_3x3, next_states_3x5, next_states_4x7,
                                 next_states_5x6, next_states_8x8, next_states_9x9)

    @pytest.mark.parametrize("size, state, player, expected_states",
                             zip(sizes, states, players, expected_next_states_list))
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

        assert mock_game.initial_state == (0,) * 82

    terminal_board = [0] * 81
    terminal_board[18:27] = [1] * 9
    terminal_board[54:56] = [-1] * 2
    terminal_board[57:63] = [-1] * 6
    terminal_state1 = (0,) + tuple(terminal_board)
    meta_board1 = (1, 1, 1, 0, 0, 0, 0, -1, -1)
    utilities1 = ({1: 1, 2: -1}, {1: 1, 2: -1}, {1: 1, 2: -1},
                  ValueError, ValueError, ValueError,
                  ValueError, {1: -1, 2: 1}, {1: -1, 2: 1})

    terminal_state2 = (2,) + (
        1, 0, 1, -1, 0, 0, 0, 1, 0,
        -1, 1, 1, 1, -1, 0, 0, -1, 0,
        1, 0, -1, 0, 0, -1, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 0, 0, -1,
        0, -1, 0, -1, -1, 1, 0, -1, 0,
        -1, 1, 0, 0, 0, -1, -1, 0, 0,
        0, -1, 0, -1, -1, -1, 0, 1, 0,
        0, 1, 0,   0, 0, 0,   0, 1, 0,
        0, 1, 0,   0, 0, 0,   0, 1, 0,
    )
    meta_board2 = (1, -1, 0, 0, 1, -1, 0, -1, 1)
    utilities2 = ({1: 1, 2: -1}, {1: -1, 2: 1}, ValueError,
                  ValueError, {1: 1, 2: -1}, {1: -1, 2: 1},
                  ValueError, {1: -1, 2: 1}, {1: 1, 2: -1})

    terminal_states = (terminal_state1, terminal_state2)
    meta_boards = (meta_board1, meta_board2)
    utilities_list = (utilities1, utilities2)

    @pytest.mark.parametrize("state, meta_board, utilities",
                             zip(terminal_states, meta_boards, utilities_list))
    def test_meta_board_is_calculated_correctly(self, state, meta_board, utilities, mocker):
        mock_game = mocker.MagicMock()
        utility_mock = mocker.MagicMock(side_effect=utilities)
        mock_nac = mocker.MagicMock(utility=utility_mock)

        # split non_terminal_state into sub boards
        board = np.array(state[1:]).reshape(9, 9)
        sub_boards = [board[i * 3:(i + 1) * 3, j * 3:(j * 3) + 3]
                      for i in range(3) for j in range(3)]

        mocker.patch("alphago.games.noughts_and_crosses.super", return_value=mock_nac)

        expected_calls = [mocker.call(tuple(sub_board.ravel()))
                          for sub_board in sub_boards]

        assert UltimateNoughtsAndCrosses._compute_meta_board(mock_game, state) == meta_board
        utility_mock.assert_has_calls(expected_calls)

    def test_meta_board_is_passed_to_super_is_terminal_method(self, mocker):
        mock_game = mocker.MagicMock()
        mock_meta_board = mocker.MagicMock()
        mock_game._compute_meta_board = mocker.MagicMock(return_value=mock_meta_board)
        mock_state = mocker.MagicMock()
        mock_is_terminal = mocker.MagicMock()
        mock_nac = mocker.MagicMock(is_terminal=mock_is_terminal)
        mocker.patch("alphago.games.noughts_and_crosses.super", return_value=mock_nac)

        UltimateNoughtsAndCrosses.is_terminal(mock_game, mock_state)
        mock_is_terminal.assert_called_once_with(mock_meta_board)
