"""Implementation of Connect 4.

Seven columns, six rows. On your turn you can play any of the columns (if it is
not full).
"""

import numpy as np

from ..utilities import memoize


INITIAL_STATE = (np.nan,) * 42
ACTION_SPACE = [i for i in range(7)]
ACTION_INDICES = {a: ACTION_SPACE.index(a) for a in ACTION_SPACE}


@memoize
def which_player(state):
    """Returns the player to play in the current state.

    Parameters
    ----------
    state: ndarray
        A 6x7 numpy array. +1 denotes a player 1 move; -1 denotes a player 2
        move; np.nan denotes a blank space.

    Returns
    -------
    player: int
        The player to play.
    """
    return (np.sum(~np.isnan(state)) % 2) + 1


@memoize
def _calculate_line_sums_4_by_4(state):
    """Calculates the line sums for a 4x4 grid.

    Parameters
    ----------
    grid: ndarray
        A 4x4 np array with +1 for a player 1 move, -1 for a player 2 move and
        np.nan for an empty space.

    Returns
    -------
    line_sums: ndarray
        An array of sums along all the possible lines on the grid.
    """
    grid = np.array(state).reshape(4, 4)
    horizontals = np.nansum(grid, axis=1)
    verticals = np.nansum(grid, axis=0)
    major_diagonal = np.nansum(grid.diagonal()),
    minor_diagonal = np.nansum(np.fliplr(grid).diagonal()),

    line_sums = np.concatenate([horizontals, verticals, major_diagonal,
                                minor_diagonal])

    return line_sums


@memoize
def _calculate_line_sums(state):
    """Calculates the line sums for the connect four game state.

    Parameters
    ----------
    state: ndarray
        A 6x7 np array denoting the state of the game.

    Returns
    -------
    line_sums: nd array
        An array of sums along all possible length 4 lines on the board.
    """
    # Extract all 4x4 subarrays of the state, and compute their line sums.
    grid = np.array(state).reshape(6, 7)
    subgrids = (grid[i:i+4, j:j+4] for i in range(3) for j in range(4))
    tuple_subgrids = (tuple(subgrid.flatten()) for subgrid in subgrids)
    line_sums_list = [_calculate_line_sums_4_by_4(subgrid)
                      for subgrid in tuple_subgrids]
    return np.concatenate(line_sums_list)


@memoize
def is_terminal(state):
    """Returns whether the state is terminal or not.

    Parameters
    ----------
    state: ndarray
        A 6x7 numpy array denoting the board state.

    Returns
    -------
    bool
        True if and only if the state is terminal; otherwise False.
    """
    # If all slots have been filled, then the game is over.
    if not np.any(np.isnan(state)):
        return True

    # Check if there is a winner
    line_sums = _calculate_line_sums(state)
    if np.any(np.abs(line_sums) == 4):
        return True

    # Otherwise it is non-terminal
    return False


@memoize
def utility(state):
    """Compute the utility of a terminal state.

    Parameters
    ----------
    state: ndarray
        The state of the game. It's a 6x7 np array.

    Returns
    -------
    utility: dict
        Dictionary with keys the players (1 and 2) and values their utility in
        this state.
    """
    line_sums = _calculate_line_sums(state)

    # Player 1 wins
    if np.any(line_sums == 4):
        return {1: 1, 2: -1}

    # Player 2 wins
    if np.any(line_sums == -4):
        return {1: -1, 2: 1}

    # Draw, if all squares are filled and no-one has four-in-a-row.
    if not np.any(np.isnan(state)):
        return {1: 0, 2: 0}

    # Otherwise the state is non-terminal and the utility cannot be calculated
    raise ValueError("Utility cannot be calculated for a "
                     "non-terminal state.")


@memoize
def compute_next_states(state):
    """Computes the next states possible from this state.

    Parameters
    ----------
    state: ndarray
        State of the game. See above.

    Returns
    -------
    next_states: dict
        Dictionary with keys the available actions and values the state
        resulting from being in state 'state' and taking the action.
    """
    player = which_player(state)
    marker = 1 if player == 1 else -1

    next_states = {}

    # Consider each column in turn.
    for col in range(7):
        next_state = np.array(state).reshape(6, 7)

        # If the top element in the column is open, then this column is an
        # available action.
        if not np.isnan(next_state[0][col]):
            continue

        # Find the lowest open element in the column. Search from the bottom
        # until we find an np.nan.
        for row in reversed(range(6)):
            if np.isnan(next_state[row][col]):
                next_state[row][col] = marker
                next_states[col] = tuple(next_state.ravel())
                break

    return next_states


def display(state):
    """Display the connect four state in a 2-D ASCII grid.

    Parameters
    ---------
    state: ndarray
        A 6x7 np array representing a connect four grid to be printed to stdout.
    """
    divider = "\n---+---+---+---+---+---+---\n"
    symbol_dict = {1: "x", -1: "o", 0: " "}

    output_rows = []
    for state_row in np.array_split(tuple(state), indices_or_sections=6):
        # convert nans to 0s for symbol lookup
        state_row[np.isnan(state_row)] = 0
        y = "|". join([" {} ".format(symbol_dict[x]) for x in state_row])
        output_rows.append(y)

    ascii_grid = divider.join(output_rows)
    print(ascii_grid)
