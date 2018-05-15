"""Noughts and crosses game

This module provides functionality for representing a game of noughts
and crosses of arbitrary size.

Classes
-------
NoughtsAndCrosses
    A class for representing a game of noughts and crosses (or
    tic-tac-toe) that can
"""

from typing import Dict, Tuple

import numpy as np

from .game import Game
GameState = Tuple[int, ...]
Action = Tuple[int, int]

__all__ = ["NoughtsAndCrosses", "UltimateNoughtsAndCrosses"]


class NoughtsAndCrosses(Game):
    """A class to represent the game of noughts and crosses (or
    tic-tac-toe).

    The game state is represented by a tuple of shape (rows * columns,)
    with -1 for 'O', 1 for 'X' and 0 for an empty square. The outcome
    of the game is given by a dictionary showing each player's outcome,
    where +1, -1 and 0 correspond to a win, loss or draw, respectively.

    This class provides a number of methods for evaluating the state of
    the game i.e. checking if the state is terminal, which player is to
    play, what the next possible states are and what the utility of a
    terminal state is. The state can also be displayed in ASCII format.

    Attributes
    ----------
    rows, columns: int
        The number of rows and columns of the noughts and crosses
        board.
    initial_state: tuple
        A tuple representing the initial state of the game. This will be
        a tuple of 0s of length rows * columns.
    action_space: tuple[tuple]
        A tuple of all the possible actions in the game, each
        represented by a two-tuple (row, col) denoting the action of
        drawing a symbol at that position. (0-based indexing)
    action_indices: dict
        A dictionary mapping each action to an index. (This is merely
        an alternative representation of the action space, primarily
        for use as a feature vector for a learning algorithm.)

    Examples
    --------
    >>> from alphago.games import NoughtsAndCrosses
    >>> nac = NoughtsAndCrosses(rows=3, columns=3)

    Define a noughts and crosses board with only two moves left.

    >>> penultimate_state = (0, 0, -1, -1, 1, 1, -1, 1, -1)
    >>> nac.is_terminal(penultimate_state)
    False
    >>> nac.display(penultimate_state)
       |   | o
    ---+---+---
     o | x | x
    ---+---+---
     o | x | o

    Calculate the possible next states, of which there should only be
    two, and which player's turn it is.

    >>> next_states = nac.compute_next_states(penultimate_state)
    >>> next_states
    {(0, 0): (-1, 0, -1, -1, 1, 1, -1, 1, -1),
    (0, 1): (0, -1, -1, -1, 1, 1, -1, 1, -1)}
    >>> nac.which_player(penultimate_state)
    2

    Take the move in the top left corner, winning the game for player
    2 and calculate its utility.

    >>> action = (0, 0)
    >>> next_state = next_states[action]
    >>> nac.display(next_state)
     o |   | o
    ---+---+---
     o | x | x
    ---+---+---
     o | x | o

    >>> nac.is_terminal(next_state)
    True
    >>> nac.utility(next_state)
    {1: -1, 2: 1}
    """

    def __init__(self, rows: int = 3, columns: int = 3) -> None:
        self.rows = rows
        self.columns = columns
        self.initial_state = (0,) * rows * columns  # type: Tuple[int, ...]
        self.action_space = tuple(
            (i, j) for i in range(self.rows)
            for j in range(self.columns))  # type: Tuple[Action, ...]
        self.action_indices = {
            a: self.action_space.index(a)
            for a in self.action_space}  # type: Dict[Action, int]

    def _calculate_line_sums(self, state: GameState
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate line sums along horizontal and vertical directions
        and along major and minor diagonals.

        Parameters
        ----------
        state: tuple
            A 1-D array representing a noughts and crosses game state.

        Returns
        -------
        tuple[ndarray]
            A tuple of arrays of sums along all the possible
            horizontal, vertical and diagonal directions, respectively.
        """
        grid = np.array(state).reshape(self.rows, self.columns)
        horizontals = np.sum(grid, axis=1)
        verticals = np.sum(grid, axis=0)
        axis_difference = abs(self.rows - self.columns)
        major_diagonals = [np.sum(grid.diagonal(offset=i))
                           for i in range(axis_difference + 1)]
        minor_diagonals = [np.sum(np.fliplr(grid).diagonal(offset=i))
                           for i in range(axis_difference + 1)]
        diagonals = np.concatenate([major_diagonals, minor_diagonals])
        return horizontals, verticals, diagonals

    def is_terminal(self, state: GameState) -> bool:
        """Given a state, returns whether it is terminal.

        Uses the fact that the noughts and crosses are represented by +1
        and -1, respectively. Whether or not a player has won can be
        determined by checking if any of the horizontal, vertical or
        diagonal line sums are equal to a value corresponding to them
        having a completed line in that direction.

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
        if np.all(state):
            return True
        # check there is a winner
        horizontals, verticals, diagonals = self._calculate_line_sums(state)
        diagonal_length = min([self.rows, self.columns])
        win_condition = np.concatenate([
            np.abs(horizontals) == self.columns,
            np.abs(verticals) == self.rows,
            np.abs(diagonals) == diagonal_length
        ])
        if np.any(win_condition):
            return True
        # otherwise it is non-terminal
        return False

    def which_player(self, state: GameState) -> int:
        """Given a state, return an int indicating the player whose turn it is.

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
        return int(np.sum(np.abs(state)) % 2) + 1

    def utility(self, state: GameState) -> Dict[int, int]:
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
        horizontals, verticals, diagonals = self._calculate_line_sums(state)
        diagonal_length = min([self.rows, self.columns])
        # player1 wins
        if (np.any(horizontals == self.columns) or np.any(verticals == self.rows)
                or np.any(diagonals == diagonal_length)):
            return {1: 1, 2: -1}
        # player2 wins
        if (np.any(horizontals == -self.columns) or np.any(verticals == -self.rows)
                or np.any(diagonals == -diagonal_length)):
            return {1: -1, 2: 1}
        # draw
        if np.all(state):
            return {1: 0, 2: 0}

        # otherwise it is non-terminal and the utility cannot be calculated
        raise ValueError("Utility can not be calculated for a "
                         "non-terminal state.")

    def compute_next_states(self, state: GameState) -> Dict[Action, GameState]:
        """Given a non-terminal state, generate a dictionary mapping legal
         actions onto their resulting game states.

         Actions are indicated by the coordinate of the square where the
         player could place a 'x' or 'o', e.g. (0, 2) for the square in
         the 3rd column of the first row (0-based indexing).

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
         """
        if self.is_terminal(state):

            raise ValueError("Next states can not be generated for a "
                             "terminal state.")

        player_symbol = 1 if self.which_player(state) == 1 else -1
        grid = np.array(state).reshape(self.rows, self.columns)

        # get a sequence of tuples (row_i, col_i) of available squares
        available_squares = tuple(zip(*np.where(grid == 0)))

        # generate all possible next_states by placing the appropriate symbol
        # in each available square
        next_states = []
        for (row_i, col_i) in available_squares:
            # convert the (row_i, col_i) into a flattened index
            flattened_index = self.columns * row_i + col_i
            # create copy of next state with a new 'o' or 'x in the
            # corresponding square # and add it to list of next states
            next_state = list(state)
            next_state[flattened_index] = player_symbol
            next_states.append(tuple(next_state))

        return {a: next_state for a, next_state
                in zip(available_squares, next_states)}

    def display(self, state: GameState) -> None:
        """Display the noughts and crosses state in a 2-D ASCII grid.

        Parameters
        ---------
        state: tuple
            A tuple representing a noughts and crosses grid to be printed to
            stdout.
        """
        divider = "\n" + "+".join(["---"] * self.columns) + "\n"
        symbol_dict = {1: "x", -1: "o", 0: " "}

        output_rows = []
        for state_row in np.array_split(state, indices_or_sections=self.rows):
            y = "|". join([" {} ".format(symbol_dict[x]) for x in state_row])
            output_rows.append(y)

        ascii_grid = divider.join(output_rows)
        print(ascii_grid)

    def __repr__(self):
        return "{0}({1}, {2})".format(self.__class__.__name__,
                                      self.rows, self.columns)


class UltimateNoughtsAndCrosses():
    """A class to represent the game of ultimate noughts and crosses
    (or tic-tac-toe)."""

    def __init__(self) -> None:
        self.initial_state = (0,) * 82
        self.sub_game = NoughtsAndCrosses()

    def _compute_meta_board(self, state: GameState) -> GameState:
        board = np.array(state[1:]).reshape(9, 9)
        sub_boards = [board[i * 3:(i + 1) * 3, j * 3:(j * 3) + 3]
                      for i in range(3) for j in range(3)]

        meta_board = []
        for sub_board in sub_boards:
            state = tuple(sub_board.ravel())
            try:
                utility = self.sub_game.utility(state)
            except ValueError:
                symbol = 0
            else:
                sub_board_winner = max(utility, key=utility.get)
                symbol = 1 if sub_board_winner == 1 else -1

            meta_board.append(symbol)
        return tuple(meta_board)

    def is_terminal(self, state: GameState) -> bool:
        meta_board = self._compute_meta_board(state)
        return self.sub_game.is_terminal(meta_board)

    def utility(self, state: GameState) -> Dict[int, int]:
        meta_board = self._compute_meta_board(state)
        return self.sub_game.utility(meta_board)
