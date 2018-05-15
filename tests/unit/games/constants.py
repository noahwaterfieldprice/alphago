import itertools

import numpy as np

# terminal state constants for M by N noughts and Crosses

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

terminal_line_sums_list = [
    ((0, -1, 1), (-1, 0, 1), (1, -3)),
    ((5, -1, -3), (1, -1, 1, -1, 1), (1, -1, -1, 1, -1, -1)),
    ((-1, -1, -1, 3), (1, 0, 0, 0, -1, 0, 0), (1, -4, 0, 0, 1, 0, 1, 0)),
    ((4, -1, -1, -1, -1), (-5, 1, 1, 1, 1, 1), (-1, 1, 1, 0)),
    ((0,) * 8, (0,) * 8, (0, 0)),
    ((1, -1, 1, -1, 1, -1, 1, -1, 1), (1, -1, 1, -1, 1, -1, 1, -1, 1), (9, 9)),
]


terminal_states = (terminal_state_3x3, terminal_state_3x5, terminal_state_4x7,
                   terminal_state_5x6, terminal_state_8x8, terminal_state_9x9)

terminal_line_sums_arrays = [
    tuple(np.array(line_sums_i) for line_sums_i in line_sums)
    for line_sums in terminal_line_sums_list]

outcomes = ([{1: -1, 2: 1},  # player 2 wins
             {1: 1, 2: -1},  # player 1 wins
             {1: -1, 2: 1},  # player 1 wins
             {1: -1, 2: 1},  # player 2 wins
             {1: 0, 2: 0},  # draw
             {1: 1, 2: -1}])  # player 1 wins


# Non-terminal state constants for M by N noughts and crosses

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


non_terminal_line_sums_list = [
    ((0, 1, 0), (2, 1, -2), (1, 1)),
    ((4, -1, -3), (1, -1, 1, -1, 0), (1, -1, -1, 0, -1, -1)),
    ((0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0)),
    ((0., 1., 0., 0., 0.), (4., -3., 0., 0., 0., 0.), (0., 0., -1., 0.)),
    ((1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 1), (0, 1)),
    ((1, -1, 1, -1, 0, -1, 1, -1, 1), (1, -1, 1, -1, 0, -1, 1, -1, 1), (8, 8))

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

non_terminal_states = (
    non_terminal_state_3x3, non_terminal_state_3x5, non_terminal_state_4x7,
    non_terminal_state_5x6, non_terminal_state_8x8, non_terminal_state_9x9
)

non_terminal_line_sums_arrays = [
    tuple(np.array(line_sums_i) for line_sums_i in line_sums)
    for line_sums in non_terminal_line_sums_list]

expected_next_states_list = (next_states_3x3, next_states_3x5, next_states_4x7,
                             next_states_5x6, next_states_8x8, next_states_9x9)
