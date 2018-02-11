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
utility:
    Return the outcome of terminal state.

Attributes
----------
INITIAL_STATE:
    A tuple representing the initial state of the game.
"""

from typing import NamedTuple

import numpy as np

__all__ = ["INITIAL_STATE", "Outcome", "is_terminal", "utility"]


class Outcome(NamedTuple):
    player1: int
    player2: int


class GameState(NamedTuple):
    player: int
    board: tuple


INITIAL_STATE = GameState(1, (np.nan, ) * 9)


def _calculate_line_sums(board):
    """Calculate line sums along horizontal and vertical directions
    and along major and minor diagonals.

    Parameters
    ----------
    board: array_like
        A 1-D array representing a noughts and crosses grid.

    Returns
    -------
    ndarray:
        An array of sums along all the possible lines on the noughts
        and crosses grid.
    """
    grid = np.array(board).reshape(3, 3)
    horizontals = np.nansum(grid, axis=1)
    verticals = np.nansum(grid, axis=0)
    major_diagonal = np.nansum(grid.diagonal()),
    minor_diagonal = np.nansum(np.fliplr(grid).diagonal()),

    return np.concatenate([horizontals, verticals,
                           major_diagonal, minor_diagonal])


def is_terminal(state):
    """Given a state, returns whether it is terminal.

    Uses the fact that the noughts and crosses are represented by +1
    and -1, respectively, so if any line sums are equal to ±3 then
    one player must have won.

    Parameters
    ---------
    state: GameState
        A state representing a noughts and crosses grid.

    Returns
    -------
    bool:
        Return ``True`` if the state is terminal or ``False`` if it is
        not.
    """

    # check all squares have been played on
    if not np.any(np.isnan(state.board)):
        return True
    # check there is a winner
    line_sums = _calculate_line_sums(state.board)
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
        A terminal state representing a noughts and crosses grid
        corresponding to either a win or a draw.

    Returns
    -------
    Outcome:
        The outcome of the terminal state for both players, represented
        as a named tuple.

    Raises
    ------
    ValueError:
        If the input state is a non-terminal state.
    """
    line_sums = _calculate_line_sums(state.board)

    # player1 wins
    if np.any(line_sums == 3):
        return Outcome(1, -1)
    # player2 wins
    if np.any(line_sums == -3):
        return Outcome(-1, 1)
    # draw
    if not np.any(np.isnan(state.board)):
        return Outcome(0, 0)

    # otherwise it is non-terminal and the utility cannot be calculated
    # TODO: Maybe have better message here
    raise ValueError("Utility can not be calculated for a "
                     "non-terminal state.")


def next_states(state):  # TODO: write a better docstring
    """Calculate a dictionary for all possible next states"""
    if is_terminal(state):
        raise ValueError("Next states can not be generated for a "
                         "terminal state.")

    player_symbol = 1 if state.player == 0 else -1
    grid = np.array(state.board).reshape(3, 3)
    # get a sequence of tuples (row_i, col_i) of available squares
    available_squares = tuple(zip(*np.where(np.isnan(grid))))
    # generate all possible next_states by placing the appropriate symbol
    # in each available square
    next_states_ = []
    for square in available_squares:
        # convert the (row_i, col_i) into a flattened index
        row_i, col_i = square
        flattened_index = 3 * row_i + col_i
        # create copy of next state with a new 'o' or 'x in the
        # corresponding square # and add it to list of next states
        next_board = list(state.board)
        next_board[flattened_index] = player_symbol
        next_player = 1 if state.player == 0 else 0
        next_states_.append(GameState(next_player, tuple(next_board)))

    return {a: next_state for a, next_state
            in zip(available_squares, next_states_)}


def display(state):
    """Display the noughts and crosses state in a 2-D ASCII grid.

    Parameters
    ---------
    state: GameState
        A state representing a noughts and crosses grid to be printed.

    """
    divider = "\n---+---+---\n"
    symbol_dict = {1: "x", -1: "o", 0: " "}

    output_rows = []
    for state_row in np.array_split(state.board, indices_or_sections=3):
        # convert nans to 0s for symbol lookup
        state_row[np.isnan(state_row)] = 0
        y = "|". join([f" {symbol_dict[x]} " for x in state_row])
        output_rows.append(y)

    ascii_grid = divider.join(output_rows)
    print(ascii_grid)