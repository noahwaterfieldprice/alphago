import pytest

from alphago.games.noughts_and_crosses import GameState, NoughtsAndCrosses

from .constants import (
    actions_to_binary_list,
    expected_next_states_list,
    flat_win_bitmasks_list,
    non_terminal_states,
    outcomes,
    terminal_states,
    win_bitmasks_list,
)


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
        mock_game = mocker.MagicMock()
        NoughtsAndCrosses.__init__(mock_game, *size)
        assert mock_game.initial_state == (0, 0, 1)

    @pytest.mark.parametrize(
        "size, actions_to_binary",
        list(zip(sizes, actions_to_binary_list, strict=True)),
    )
    def test_action_to_binary_is_correct(self, size, actions_to_binary, mocker):
        mock_game = mocker.MagicMock()
        NoughtsAndCrosses.__init__(mock_game, *size)
        assert mock_game._actions_to_binary == actions_to_binary

    @pytest.mark.parametrize(
        "size, actions_to_binary, win_bitmasks",
        list(zip(sizes, actions_to_binary_list, win_bitmasks_list, strict=True)),
    )
    def test_calculating_row_win_bitmasks(
        self, size, actions_to_binary, win_bitmasks, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, _actions_to_binary=actions_to_binary
        )
        row_win_bitmasks = NoughtsAndCrosses._calculate_row_bitmasks(mock_game)
        assert row_win_bitmasks == win_bitmasks["row"]

    @pytest.mark.parametrize(
        "size, actions_to_binary, win_bitmasks",
        list(zip(sizes, actions_to_binary_list, win_bitmasks_list, strict=True)),
    )
    def test_calculating_column_win_bitmasks(
        self, size, actions_to_binary, win_bitmasks, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, _actions_to_binary=actions_to_binary
        )
        column_win_bitmasks = NoughtsAndCrosses._calculate_column_bitmasks(mock_game)
        assert column_win_bitmasks == win_bitmasks["column"]

    @pytest.mark.parametrize(
        "size, actions_to_binary, win_bitmasks",
        list(zip(sizes, actions_to_binary_list, win_bitmasks_list, strict=True)),
    )
    def test_calculating_major_diagonal_win_bitmasks(
        self, size, actions_to_binary, win_bitmasks, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, _actions_to_binary=actions_to_binary
        )
        major_diagonal_win_bitmasks = (
            NoughtsAndCrosses._calculate_major_diagonal_bitmasks(mock_game)
        )
        assert major_diagonal_win_bitmasks == win_bitmasks["major_diagonal"]

    @pytest.mark.parametrize(
        "size, actions_to_binary, win_bitmasks",
        list(zip(sizes, actions_to_binary_list, win_bitmasks_list, strict=True)),
    )
    def test_calculating_minor_diagonal_win_bitmasks(
        self, size, actions_to_binary, win_bitmasks, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, _actions_to_binary=actions_to_binary
        )
        minor_diagonal_win_bitmasks = (
            NoughtsAndCrosses._calculate_minor_diagonal_bitmasks(mock_game)
        )
        assert minor_diagonal_win_bitmasks == win_bitmasks["minor_diagonal"]

    @pytest.mark.parametrize(
        "size, actions_to_binary, flat_win_bitmasks, state",
        list(
            zip(
                sizes,
                actions_to_binary_list,
                flat_win_bitmasks_list,
                terminal_states,
                strict=True,
            )
        ),
    )
    def test_is_terminal_returns_true_for_terminal_states(
        self, size, actions_to_binary, flat_win_bitmasks, state, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows,
            columns=columns,
            _actions_to_binary=actions_to_binary,
            _win_bitmasks=flat_win_bitmasks,
        )
        assert NoughtsAndCrosses.is_terminal(mock_game, state) is True

    @pytest.mark.parametrize(
        "size, actions_to_binary, flat_win_bitmasks, state",
        list(
            zip(
                sizes,
                actions_to_binary_list,
                flat_win_bitmasks_list,
                non_terminal_states,
                strict=True,
            )
        ),
    )
    def test_is_terminal_returns_false_for_non_terminal_states(
        self, size, actions_to_binary, flat_win_bitmasks, state, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows,
            columns=columns,
            _actions_to_binary=actions_to_binary,
            _win_bitmasks=flat_win_bitmasks,
        )
        assert NoughtsAndCrosses.is_terminal(mock_game, state) is False

    players = (2, 1, 1, 1, 1, 2)

    @pytest.mark.parametrize(
        "player, state", list(zip(players, non_terminal_states, strict=True))
    )
    def test_current_player_returns_correct_player(self, player, state, mocker):
        mock_game = mocker.MagicMock()
        assert NoughtsAndCrosses.current_player(mock_game, state) == player

    @pytest.mark.parametrize(
        "size, state", list(zip(sizes, non_terminal_states, strict=True))
    )
    def test_utility_raises_exception_on_non_terminal_input_state(
        self, size, state, mocker
    ):
        rows, columns = size
        mock_is_terminal = mocker.MagicMock(return_value=False)
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, is_terminal=mock_is_terminal
        )
        with pytest.raises(ValueError) as exception_info:
            NoughtsAndCrosses.utility(mock_game, state)
        assert str(exception_info.value) == (
            "Utility can not be calculated for a non-terminal state."
        )
        mock_is_terminal.assert_called_once_with(state)

    @pytest.mark.parametrize(
        "size, flat_win_bitmasks, state, outcome",
        list(
            zip(sizes, flat_win_bitmasks_list, terminal_states, outcomes, strict=True)
        ),
    )
    def test_utility_function_returns_correct_outcomes(
        self, size, flat_win_bitmasks, state, outcome, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, _win_bitmasks=flat_win_bitmasks
        )

        assert NoughtsAndCrosses.utility(mock_game, state) == outcome

    def test_legal_actions_raises_exception_on_terminal_input_state(self, mocker):
        mock_game = mocker.MagicMock()
        mock_game.is_terminal = mocker.MagicMock(return_value=True)
        mock_state = mocker.MagicMock()
        with pytest.raises(ValueError) as exception_info:
            NoughtsAndCrosses.legal_actions(mock_game, mock_state)
        assert str(exception_info.value) == (
            "Legal actions can not be computed for a terminal state."
        )

    @pytest.mark.parametrize(
        "state, action, expected_next_state",
        [
            # player 1 plays the top-left corner of an empty 3x3 board
            (GameState(0, 0, 1), (0, 0), GameState(0b000000001, 0, 2)),
            # player 2 replies in the top-middle square
            (GameState(0b000000001, 0, 2), (0, 1), GameState(0b000000001, 0b10, 1)),
        ],
    )
    def test_generating_next_state_from_given_action_and_state(
        self, state, action, expected_next_state, mocker
    ):
        # _next_state only needs the action->binary map; drive it in isolation.
        mock_game = mocker.MagicMock(_actions_to_binary=actions_to_binary_list[0])
        next_state = NoughtsAndCrosses._next_state(mock_game, state, action)
        assert next_state == expected_next_state

    @pytest.mark.parametrize(
        "size, actions_to_binary, state, expected_states",
        list(
            zip(
                sizes,
                actions_to_binary_list,
                non_terminal_states,
                expected_next_states_list,
                strict=True,
            )
        ),
    )
    def test_generating_a_dict_of_all_possible_next_states(
        self, size, actions_to_binary, state, expected_states, mocker
    ):
        rows, columns = size
        mock_game = mocker.MagicMock(
            rows=rows, columns=columns, _actions_to_binary=actions_to_binary
        )
        mock_game.is_terminal = mocker.MagicMock(return_value=False)
        # wire _next_state to the real implementation so legal_actions returns
        # concrete GameStates rather than MagicMock placeholders
        mock_game._next_state = lambda s, a: NoughtsAndCrosses._next_state(
            mock_game, s, a
        )
        assert NoughtsAndCrosses.legal_actions(mock_game, state) == expected_states

    states = [
        GameState(0b000000000, 0b000000000, 1),
        GameState(0b001010001, 0b100000010, 2),
        GameState(0b100001111, 0b011110000, 2),
    ]
    div = "---+---+---"
    # additional newline character accounts for the one added to the output
    # by the print function itself
    outputs = [
        "\n".join(("   |   |   ", div, "   |   |   ", div, "   |   |   ")) + "\n",
        "\n".join((" x | o |   ", div, "   | x |   ", div, " x |   | o ")) + "\n",
        "\n".join((" x | x | x ", div, " x | o | o ", div, " o | o | x ")) + "\n",
    ]

    @pytest.mark.parametrize(
        "state, expected_output", list(zip(states, outputs, strict=True))
    )
    def test_display_function_outputs_correct_string_for_3x3(
        self, state, expected_output, capsys, mocker
    ):
        mock_game = mocker.MagicMock(
            rows=3, columns=3, _actions_to_binary=actions_to_binary_list[0]
        )
        NoughtsAndCrosses.display(mock_game, state)

        output = capsys.readouterr().out
        assert output == expected_output
