import numpy as np
import pytest

import alphago.noughts_and_crosses_mxn as nacmn

sizes = [(3, 5), (4, 7), (5, 6), (9, 9)]


@pytest.mark.parametrize("rows, columns", sizes)
def test_noughts_and_crosses_instances_have_correct_size(rows, columns):
    game = nacmn.NoughtsAndCrosses(rows, columns)
    assert game.rows == rows
    assert game.columns == columns


# terminal state for 3x5 game - 1s top row
terminal_state_3x5 = (1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1)
# terminal state for 4x7 game - -1s 2nd major diagonal
terminal_state_4x7 = np.empty((4, 7))
terminal_state_4x7.fill(np.nan)
for i in range(4):
    terminal_state_4x7[i, i+1] = -1
terminal_state_4x7[3, :4] = 1
# terminal state for 5x6 game - 1s left column
terminal_state_5x6 = np.empty((5, 6))
terminal_state_5x6.fill(np.nan)
terminal_state_5x6[:, 0] = 1
terminal_state_5x6[1:, 1] = -1
# terminal state for 9x9 game - 1s both diagonals
even_row = np.array((1, -1) * 4 + (1,))
odd_row = even_row * -1
terminal_states_9x9 = np.row_stack((even_row, odd_row) * 4 + (even_row,))

terminal_states = (terminal_state_3x5, terminal_state_4x7,
                   terminal_state_5x6, terminal_states_9x9)


line_sums_list = [
    ((5, -1, -3),  (1, -1,  1, -1,  1),  (1, -1, -1,  1, -1, -1)),
    ((-1, -1, -1, 3), (1, 0, 0, 0, -1, 0, 0), (1, -4, 0, 0, 1, 0, 1, 0)),
    ((1, 0, 0, 0, 0), (5, -4, 0, 0, 0, 0), (0, 0, -1, 0)),
    ((1, -1, 1, -1, 1, -1, 1, -1, 1), (1, -1, 1, -1, 1, -1, 1, -1, 1), (9, 9)),
]


@pytest.mark.parametrize("size, state, line_sums",
                         zip(sizes, terminal_states, line_sums_list))
def test_line_sums_are_calculated_correctly(size, state, line_sums):
    # TODO: need to mock this
    game = nacmn.NoughtsAndCrosses(*size)
    calculated_line_sums = game._calculate_line_sums(state)
    calculated_line_sum_tuples = tuple(tuple(line_sums_)
                                       for line_sums_ in calculated_line_sums)
    assert calculated_line_sum_tuples == line_sums


@pytest.mark.parametrize("size, state", zip(sizes, terminal_states))
def test_is_terminal_returns_true_for_terminal_states(size, state):
    # TODO: need to mock this
    game = nacmn.NoughtsAndCrosses(*size)
    assert game.is_terminal(state) is True
