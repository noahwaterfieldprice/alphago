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

INITIAL_STATE = (np.nan, ) * 9


class Outcome(NamedTuple):
    player1_score: int
    player2_score: int


def _calculate_line_sums(state):
    """Calculate line sums along horizontal and vertical directions
    and along major and minor diagonals.

    Parameters
    ----------
    state: array_like
        A state representing a noughts and crosses grid.

    Returns
    -------
    ndarray:
        An array of sums along all the possible lines on the noughts
        and crosses grid.
    """
    board = np.array(state).reshape(3, 3)
    horizontals = np.nansum(board, axis=1)
    verticals = np.nansum(board, axis=0)
    major_diagonal = np.nansum(board.diagonal()),
    minor_diagonal = np.nansum(np.fliplr(board).diagonal()),

    return np.concatenate([horizontals, verticals,
                           major_diagonal, minor_diagonal])


def is_terminal(state):
    """ Given a state, returns whether it is terminal.

    Uses the fact that the noughts and crosses are represented by +1
    and -1, respectively, so if any line sums are equal to ±3 then
    one player must have won.

    Parameters
    ---------
    state: array_like
        A state representing a noughts and crosses grid.

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
    line_sums = _calculate_line_sums(state)

    # player1 wins
    if np.any(line_sums == 3):
        return Outcome(1, -1)
    # player2 wins
    if np.any(line_sums == -3):
        return Outcome(-1, 1)
    # draw
    if not np.any(np.isnan(state)):
        return Outcome(0, 0)

    # otherwise it is non-terminal and the utility cannot be calculated
    # TODO: Maybe have better message here
    raise ValueError("Utility can not be calculated for a "
                     "non-terminal state.")
