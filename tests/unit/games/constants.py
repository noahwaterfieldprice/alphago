import itertools

sizes = ((3, 3), (3, 5), (4, 7), (5, 6), (8, 8))

# generate position map from row, col tuple to binary constants
actions_to_binary_list = []
for (rows, columns) in sizes:
    actions_to_binary = {(row, col): 1 << columns * row + col
                         for row in range(rows)
                         for col in range(columns)}
    actions_to_binary_list.append(actions_to_binary)

# generate win bitmasks for each shape
win_bitmasks_list = []
for (rows, columns) in sizes:
    win_bitmasks = {}
    # calculate binary row masks
    base_row_mask = 2 ** columns - 1
    row_bit_masks = [base_row_mask << (i * columns) for i in range(rows)]
    win_bitmasks["row"] = row_bit_masks

    # calculate binary column masks
    base_column_mask = sum(2 ** (i * columns) for i in range(rows))
    column_bit_masks = [base_column_mask << i for i in range(columns)]
    win_bitmasks["column"] = column_bit_masks

    # calculate binary major diagonal masks
    major_diagonal_base_mask = sum(2 ** (i * columns) << i
                                   for i in range(rows))
    if rows == columns:
        major_diagonal_masks = [major_diagonal_base_mask]
    elif columns > rows:
        major_diagonal_masks = [major_diagonal_base_mask << i
                                for i in range(columns - rows + 1)]
    else:
        major_diagonal_masks = [major_diagonal_base_mask << (i * rows)
                                for i in range(rows - columns + 1)]
    win_bitmasks["major_diagonal"] = major_diagonal_masks

    # calculate binary minor diagonal masks
    minor_diagonal_base_mask = sum(2 ** (i * columns) << (rows - i - 1)
                                   for i in range(rows))
    if rows == columns:
        minor_diagonal_masks = [minor_diagonal_base_mask]
    elif columns > rows:
        minor_diagonal_masks = [minor_diagonal_base_mask << i
                                for i in range(columns - rows + 1)]
    else:
        minor_diagonal_masks = [minor_diagonal_base_mask << (i * rows)
                                for i in range(rows - columns + 1)]
    win_bitmasks["minor_diagonal"] = minor_diagonal_masks

    win_bitmasks_list.append(win_bitmasks)


# terminal state constants for M by N noughts and Crosses
# ----
# terminal state for 3x3 game - O's minor diagonal
terminal_state_3x3 = 0b100001011, 0b001110100, 1
# terminal state for 3x5 game - X's top row
terminal_state_3x5 = (4671, 28096, 2)
# terminal state for 4x7 game - O's 2nd major diagonal
terminal_state_4x7 = (31457280, 33686018, 1)
# terminal state for 5x6 game - O's  2nd left column
terminal_state_5x6 = (62, 17043521, 1)
# terminal state for 8x8 game - draw
terminal_state_8x8 = (14714328930390887475, 3732415143318664140, 1)

terminal_states = (terminal_state_3x3, terminal_state_3x5, terminal_state_4x7,
                   terminal_state_5x6, terminal_state_8x8)

outcomes = ([{1: -1, 2: 1},  # player 2 wins
             {1: 1, 2: -1},  # player 1 wins
             {1: -1, 2: 1},  # player 1 wins
             {1: -1, 2: 1},  # player 2 wins
             {1: 0, 2: 0}])  # draw

# Non-terminal state constants for M by N noughts and crosses

# non-terminal state for 3x3 game
non_terminal_state_3x3 = 0b100010100, 0b001000001, 2
# non-terminal state for 3x5 game
non_terminal_state_3x5 = (4655, 28096, 1)
# non-terminal state for 4x7 game - no moves played yet
non_terminal_state_4x7 = (0, 0, 1)
# non-terminal state for 5x6 game - almost all Xs in left column
non_terminal_state_5x6 = (17043520, 34087040, 1)
# non-terminal state for 8x8 game - checkerboard apart from top left corner
non_terminal_state_8x8 = (14714328930390887474, 3732415143318664140, 1)

# player 2s turn
next_states_3x3 = {
    (0, 1): (0b100010100, 0b001000011, 1),
    (1, 0): (0b100010100, 0b001001001, 1),
    (1, 2): (0b100010100, 0b001100001, 1),
    (2, 1): (0b100010100, 0b011000001, 1),
}
# player 1s turn - complete a row across first row
next_states_3x5 = {
    (0, 4): (1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1)
}
# player 1s turn - all states possible
next_states_4x7 = {}
# for action in itertools.product(range(4), range(7)):
#     row, col = action
#     next_state = list(non_terminal_state_4x7)
#     next_state[row * 7 + col] = 1
#     next_states_4x7[action] = tuple(next_state)
# player 2s turn - 1st two columns almost full
next_states_5x6 = {}
actions_5x6 = list(itertools.product(range(5), range(2, 6)))
actions_5x6.extend([(0, 0), (0, 1), (1, 1)])
# for action in actions_5x6:
#     row, col = action
#     next_state = list(non_terminal_state_5x6)
#     next_state[row * 6 + col] = -1
#     next_states_5x6[action] = tuple(next_state)
# player 2s turn - checkerboard except for top right corner
next_states_8x8 = {(0, 7): terminal_state_8x8}

non_terminal_states = (
    non_terminal_state_3x3, non_terminal_state_3x5, non_terminal_state_4x7,
    non_terminal_state_5x6, non_terminal_state_8x8
)

expected_next_states_list = (next_states_3x3, next_states_3x5, next_states_4x7,
                             next_states_5x6, next_states_8x8)
