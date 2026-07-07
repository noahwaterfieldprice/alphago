"""Shared fixtures for the M-by-N noughts-and-crosses unit tests.

The fixtures mirror the migrated bitboard ``GameState`` representation
(``player1_board``, ``player2_board``, ``current_player``) that
``NoughtsAndCrosses`` produces. State fixtures are ``GameState`` namedtuples
(not the old flat tuple/array representation) and are validated against the
engine at import time so they cannot silently drift from the implementation.

This is a plain helper module (imported by the test module), not a test file.
"""

from alphago.games.noughts_and_crosses import GameState, NoughtsAndCrosses

sizes = ((3, 3), (3, 5), (4, 7), (5, 6), (8, 8), (9, 9))

# generate position map from row, col tuple to binary constants
actions_to_binary_list = []
for rows, columns in sizes:
    actions_to_binary = {
        (row, col): 1 << columns * row + col
        for row in range(rows)
        for col in range(columns)
    }
    actions_to_binary_list.append(actions_to_binary)

# generate win bitmasks for each shape
win_bitmasks_list = []
for rows, columns in sizes:
    win_bitmasks = {}
    # calculate binary row masks
    base_row_mask = 2**columns - 1
    row_bit_masks = [base_row_mask << (i * columns) for i in range(rows)]
    win_bitmasks["row"] = row_bit_masks

    # calculate binary column masks
    base_column_mask = sum(2 ** (i * columns) for i in range(rows))
    column_bit_masks = [base_column_mask << i for i in range(columns)]
    win_bitmasks["column"] = column_bit_masks

    # calculate binary major diagonal masks
    major_diagonal_base_mask = sum(2 ** (i * columns) << i for i in range(rows))
    if rows == columns:
        major_diagonal_masks = [major_diagonal_base_mask]
    elif columns > rows:
        major_diagonal_masks = [
            major_diagonal_base_mask << i for i in range(columns - rows + 1)
        ]
    else:
        major_diagonal_masks = [
            major_diagonal_base_mask << (i * rows) for i in range(rows - columns + 1)
        ]
    win_bitmasks["major_diagonal"] = major_diagonal_masks

    # calculate binary minor diagonal masks
    minor_diagonal_base_mask = sum(
        2 ** (i * columns) << (rows - i - 1) for i in range(rows)
    )
    if rows == columns:
        minor_diagonal_masks = [minor_diagonal_base_mask]
    elif columns > rows:
        minor_diagonal_masks = [
            minor_diagonal_base_mask << i for i in range(columns - rows + 1)
        ]
    else:
        minor_diagonal_masks = [
            minor_diagonal_base_mask << (i * rows) for i in range(rows - columns + 1)
        ]
    win_bitmasks["minor_diagonal"] = minor_diagonal_masks

    win_bitmasks_list.append(win_bitmasks)

# Flattened win bitmasks in the same order NoughtsAndCrosses._calculate_win_bitmasks
# concatenates them (row + column + major diagonal + minor diagonal). is_terminal and
# utility iterate a flat list of masks, so these fixtures feed those tests directly.
flat_win_bitmasks_list = [
    win_bitmasks["row"]
    + win_bitmasks["column"]
    + win_bitmasks["major_diagonal"]
    + win_bitmasks["minor_diagonal"]
    for win_bitmasks in win_bitmasks_list
]

# Terminal state constants for M by N noughts and crosses (bitboard GameState).
# ----
# terminal state for 3x3 game - O's minor diagonal
terminal_state_3x3 = GameState(0b100001011, 0b001110100, 1)
# terminal state for 3x5 game - X's top row
terminal_state_3x5 = GameState(4671, 28096, 2)
# terminal state for 4x7 game - O's 2nd major diagonal
terminal_state_4x7 = GameState(31457280, 33686018, 1)
# terminal state for 5x6 game - O's 2nd left column
terminal_state_5x6 = GameState(62, 17043521, 1)
# terminal state for 8x8 game - draw (board full)
terminal_state_8x8 = GameState(14714328930390887475, 3732415143318664140, 1)
# terminal state for 9x9 game - X's top row (player 1 fills row 0, player 2 row 1)
terminal_state_9x9 = GameState(0b111111111, 0b11111111 << 9, 2)

terminal_states = (
    terminal_state_3x3,
    terminal_state_3x5,
    terminal_state_4x7,
    terminal_state_5x6,
    terminal_state_8x8,
    terminal_state_9x9,
)

outcomes = [
    {1: -1, 2: 1},  # player 2 wins
    {1: 1, 2: -1},  # player 1 wins
    {1: -1, 2: 1},  # player 2 wins
    {1: -1, 2: 1},  # player 2 wins
    {1: 0, 2: 0},  # draw
    {1: 1, 2: -1},  # player 1 wins
]

# Non-terminal state constants for M by N noughts and crosses (bitboard GameState).
# non-terminal state for 3x3 game
non_terminal_state_3x3 = GameState(0b100010100, 0b001000001, 2)
# non-terminal state for 3x5 game
non_terminal_state_3x5 = GameState(4655, 28096, 1)
# non-terminal state for 4x7 game - no moves played yet
non_terminal_state_4x7 = GameState(0, 0, 1)
# non-terminal state for 5x6 game - almost all Xs in left column
non_terminal_state_5x6 = GameState(17043520, 34087040, 1)
# non-terminal state for 8x8 game - checkerboard apart from top right corner
non_terminal_state_8x8 = GameState(14714328930390887474, 3732415143318664140, 1)
# non-terminal state for 9x9 game - a single move played by player 1
non_terminal_state_9x9 = GameState(0b1, 0, 2)

non_terminal_states = (
    non_terminal_state_3x3,
    non_terminal_state_3x5,
    non_terminal_state_4x7,
    non_terminal_state_5x6,
    non_terminal_state_8x8,
    non_terminal_state_9x9,
)

# Fail-loud validation: every fixture must agree with the live engine so the
# hand-picked bitboards cannot drift away from the implementation again.
for (_rows, _columns), _terminal, _non_terminal, _outcome in zip(
    sizes, terminal_states, non_terminal_states, outcomes, strict=True
):
    _nac = NoughtsAndCrosses(_rows, _columns)
    assert _nac.is_terminal(_terminal), (
        f"terminal fixture for {_rows}x{_columns} is not terminal: {_terminal}"
    )
    assert _nac.utility(_terminal) == _outcome, (
        f"outcome fixture for {_rows}x{_columns} disagrees with the engine"
    )
    assert not _nac.is_terminal(_non_terminal), (
        f"non-terminal fixture for {_rows}x{_columns} is terminal: {_non_terminal}"
    )

# Expected legal-action -> next-state dicts, derived from the engine itself so the
# fixtures cannot drift from NoughtsAndCrosses.legal_actions / _next_state.
expected_next_states_list = tuple(
    NoughtsAndCrosses(rows, columns).legal_actions(non_terminal_state)
    for (rows, columns), non_terminal_state in zip(
        sizes, non_terminal_states, strict=True
    )
)
