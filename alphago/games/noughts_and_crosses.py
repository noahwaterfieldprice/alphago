"""Noughts and crosses game

This module provides functionality for representing a game of noughts
and crosses. The game state is represented by a tuple of shape (9,)
with -1 for 'O', 1 for 'X' and np.nan for empty square. The outcome of
the game is given by a named tuple showing each player's outcome, where
+1, -1 and 0 correspond to a win, loss or draw, respectively.

Functions
---------
is_terminal:
    Return a bool indicating whether or not the input state is
    terminal.
which_player:
    Given a state, return and int indicating the player whose turn it
    is.
next_states:
    Given a state, return a dict of all possible next states.
utility:
    Given a terminal state, return the outcomes for each player
display:
    Display a noughts and crosses state as an ASCII grid.

Attributes
----------
INITIAL_STATE:
    A tuple representing the initial state of the game.
"""

import numpy as np

__all__ = ["INITIAL_STATE", "ACTION_SPACE", "ACTION_INDICES", "is_terminal",
           "which_player", "compute_next_states", "utility", "display"]

INITIAL_STATE = (np.nan, ) * 9
ACTION_SPACE = [(i, j) for i in range(3) for j in range(3)]
ACTION_INDICES = {a: ACTION_SPACE.index(a) for a in ACTION_SPACE}


def _calculate_line_sums(state):
    """Calculate line sums along horizontal and vertical directions
    and along major and minor diagonals.

    Parameters
    ----------
    state: tuple
        A 1-D array representing a noughts and crosses game state.

    Returns
    -------
    line_sums: ndarray
        An array of sums along all the possible lines on the noughts
        and crosses grid.
    """
    grid = np.array(state).reshape(3, 3)
    horizontals = np.nansum(grid, axis=1)
    verticals = np.nansum(grid, axis=0)
    major_diagonal = np.nansum(grid.diagonal()),
    minor_diagonal = np.nansum(np.fliplr(grid).diagonal()),

    line_sums = np.concatenate([horizontals, verticals,
                                major_diagonal, minor_diagonal])
    return line_sums


def which_player(state):
    """Given a state, return and int indicating the player whose turn it
    is.

    Parameters
    ----------
    state: array_like
        A 1-D array representing a noughts and crosses game state.

    Returns
    -------
    player: int
        An int indicating the player whose turn it is, where 1 and 2
        correspond to player 1 and player 2, respectively.
    """
    return (np.sum(~np.isnan(state)) % 2) + 1


def is_terminal(state):
    """Given a state, returns whether it is terminal.

    Uses the fact that the noughts and crosses are represented by +1
    and -1, respectively, so if any line sums are equal to ±3 then
    one player must have won.

    Parameters
    ---------
    state: tuple
        A 1-D array representing a noughts and crosses game state.

    Returns
    -------
    bool:
        Return ``True`` if the state is terminal or ``False`` if it is
        not.
    """

    # check all squares have been played on
    if not np.any(np.isnan(state)):
        return True
    # check there is a winner
    line_sums = _calculate_line_sums(state)
    if np.any(np.abs(line_sums) == 3):
        return True
    # otherwise it is non-terminal
    return False


def utility(state):
    """Given a terminal noughts and crosses state, calculates the
    outcomes for both players. These outcomes are given by +1, -1
    and 0 for a win, loss, or draw, respectively.

    Parameters
    ---------
    state: tuple
        A 1-D array representing a terminal noughts and crosses game
        state, corresponding to either a win or a draw.

    Returns
    -------
    outcome: dict
        The outcome of the terminal state for both players, represented
        as a dictionary with keys ints indicating the players and
        values each players respective utility.

    Raises
    ------
    ValueError:
        If the input state is a non-terminal state.
    """
    line_sums = _calculate_line_sums(state)

    # player1 wins
    if np.any(line_sums == 3):
        return {1: 1, 2: -1}
    # player2 wins
    if np.any(line_sums == -3):
        return {1: -1, 2: 1}
    # draw
    if not np.any(np.isnan(state)):
        return {1: 0, 2: 0}

    # otherwise it is non-terminal and the utility cannot be calculated
    # TODO: Maybe have better message here
    raise ValueError("Utility can not be calculated for a "
                     "non-terminal state.")


def compute_next_states(state):
    """Given a non-terminal state, generate a dictionary mapping legal
    actions onto their resulting game states.

    Actions are indicated by the coordinate of the square where the
    player could place a 'x' or 'o', e.g. (2, 1) for the middle square
    on the bottom row.

    Parameters
    ----------
    state: tuple
        A 1-D array representing a non-terminal noughts and crosses
        game state.

    Returns
    -------
    next_states: dict
        A dictionary mapping all possible legal actions from the input
        game state to the corresponding game states resulting from
        taking each action.


    Examples
    --------
    >>> import alphago.games.noughts_and_crosses as nac

    Define a noughts and crosses board with only one move left.

    >>> penultimate_state = (1, np.nan, -1, -1, 1, 1, 1, -1, -1)

    Calculate the possible next states, of which there should only be one

    >>> nac.compute_next_states(penultimate_state)
    (0, 1): (1, 1, -1, -1, 1, 1, 1, -1, -1)
    """
    if is_terminal(state):
        raise ValueError("Next states can not be generated for a "
                         "terminal state.")

    # TODO: maybe ternary expressions are not the clearest in this func
    player_symbol = 1 if which_player(state) == 1 else -1
    grid = np.array(state).reshape(3, 3)

    # get a sequence of tuples (row_i, col_i) of available squares
    available_squares = tuple(zip(*np.where(np.isnan(grid))))

    # generate all possible next_states by placing the appropriate symbol
    # in each available square
    next_states = []
    for (row_i, col_i) in available_squares:
        # convert the (row_i, col_i) into a flattened index
        flattened_index = 3 * row_i + col_i
        # create copy of next state with a new 'o' or 'x in the
        # corresponding square # and add it to list of next states
        next_state = list(state)
        next_state[flattened_index] = player_symbol
        next_states.append(tuple(next_state))

    return {a: next_state for a, next_state
            in zip(available_squares, next_states)}


def display(state):
    """Display the noughts and crosses state in a 2-D ASCII grid.

    Parameters
    ---------
    state: tuple
        A tuple representing a noughts and crosses grid to be printed to
        stdout.
    """
    divider = "\n---+---+---\n"
    symbol_dict = {1: "x", -1: "o", 0: " "}

    output_rows = []
    for state_row in np.array_split(state, indices_or_sections=3):
        # convert nans to 0s for symbol lookup
        state_row[np.isnan(state_row)] = 0
        y = "|". join([" {} ".format(symbol_dict[x]) for x in state_row])
        output_rows.append(y)

    ascii_grid = divider.join(output_rows)
    print(ascii_grid)


def optimal_strategy(state):
    """Returns the optimal strategy for the game of noughts and crosses in the
    given state.

    Parameters
    ----------
    state: tuple
        A tuple denoting the game state.

    Returns
    -------
    action: tuple
        An action to play. It is a tuple of the form (i, j) denoting the ith row
        and jth column, where (0, 0) denotes the top left corner.
    """
    # TODO: Implement me.
    pass
