"""Noughts and crosses game

This module provides functionality for representing a game of noughts
and crosses of arbitrary size.

Classes
-------
NoughtsAndCrosses
    A class for representing a game of noughts and crosses (or
    tic-tac-toe) for any given two dimensions i.e. rows and columns

UltimateNoughtsAndCrosses
    A class for representing a game of ultimate noughts and crosses (or
    ultimate tic-tac-toe).
"""
import functools
import itertools
import operator
from typing import Dict, NamedTuple, Tuple

import numpy as np

from .game import Game

__all__ = ["NoughtsAndCrosses", "UltimateNoughtsAndCrosses"]


class GameState(NamedTuple):
    player1_board: int
    player2_board: int
    current_player: int


class Action(NamedTuple):
    row: int
    column: int


class NoughtsAndCrosses(Game):
    """A class to represent the game of noughts and crosses (or
    tic-tac-toe).

    The game state is represented by a named tuple consisting of
        (player1_board, player2_board, current_player)
    where each players board is represented by (rows * cols)-bit binary
    number.The outcome of the game is given by a dictionary showing each
    player's outcome, where +1, -1 and 0 correspond to a win, loss or
    draw, respectively.

    This class provides a number of methods for evaluating the state of
    the game i.e. checking if the state is terminal, which player is to
    play, what the next possible states are and what the utility of a
    terminal state is. The state can also be displayed in ASCII format.

    Attributes
    ----------
    rows, columns:
        The number of rows and columns of the noughts and crosses
        board.
    initial_state:
        A tuple representing the initial state of the game. This will be
        a tuple of 0s of length rows * columns.
    action_space:
        A tuple of all the possible actions in the game, each
        represented by a two-tuple (row, col) denoting the action of
        drawing a symbol at that position. (0-based indexing)
    action_indices:
        A dictionary mapping each action to an index. (This is merely
        an alternative representation of the action space, primarily
        for use as a feature vector for a learning algorithm.)

    Examples
    --------
    >>> from alphago.games import NoughtsAndCrosses
    >>> nac = NoughtsAndCrosses(rows=3, columns=3)

    Define a noughts and crosses board with only two moves left.

    >>> penultimate_state = (0b000110010, 0b100001101, 2)
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

    >>> legal_actions = nac.legal_actions(penultimate_state)
    >>> legal_actions
    {(2, 0): (50, 333, 1), (2, 1): (50, 397, 1)}
    >>> nac.current_player(penultimate_state)
    2

    Take the move in the top left corner, winning the game for player
    2 and calculate its utility.

    >>> action = (2, 0)
    >>> next_state = legal_actions[action]
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
        self.initial_state = GameState(0, 0, 1)  # type: GameState
        self.action_space = tuple(Action(row, col)
                                  for row in range(self.rows)
                                  for col in range(self.columns)
                                  )  # type: Tuple[Action, ...]
        self.action_indices = {
            a: self.action_space.index(a)
            for a in self.action_space}  # type: Dict[Action, int]

        self._actions_to_binary = {Action(row, col): 1 << self.columns * row + col
                                   for row in range(self.rows)
                                   for col in range(self.columns)}

        self._win_bitmasks = self._calculate_win_bitmasks()

    def _calculate_win_bitmasks(self):  # TODO: split into smaller functions
        row_wins, column_wins, diagonal_wins = [], [], []
        # construct bitmasks for wins corresponding to a full row
        for row in range(self.rows):
            row_win = functools.reduce(
                operator.or_, [self._actions_to_binary[Action(row, column)]
                               for column in range(self.columns)])
            row_wins.append(row_win)

        # construct bitmasks for wins corresponding to a full column
        for column in range(self.columns):
            column_win = functools.reduce(
                operator.or_, [self._actions_to_binary[(row, column)]
                               for row in range(self.rows)])
            column_wins.append(column_win)

        # determine if board is non-square and in which direction
        row_excess = max(0, self.rows - self.columns)
        col_excess = max(0, self.columns - self.rows)

        # construct bitmasks for win corresponding to a full major diagonal
        min_dim, max_dim = sorted([self.rows, self.columns])
        major_diagonals = [[(i + row_offset, i + col_offset)
                            for i in range(min_dim)]
                           for row_offset in range(row_excess + 1)
                           for col_offset in range(col_excess + 1)]
        for major_diag_indices in major_diagonals:
            major_diag_win = functools.reduce(
                operator.or_, [self._actions_to_binary[(row, column)]
                               for row, column in major_diag_indices])
            diagonal_wins.append(major_diag_win)

        # construct bitmasks for win corresponding to a full minor diagonal
        minor_diagonals = [[(min_dim - 1 - i + row_offset, i + col_offset)
                            for i in range(min_dim)]
                           for row_offset in range(row_excess + 1)
                           for col_offset in range(col_excess + 1)]
        for minor_diag_indices in minor_diagonals:
            minor_diag_win = functools.reduce(
                operator.or_, [self._actions_to_binary[(row, column)]
                               for row, column in minor_diag_indices])
            diagonal_wins.append(minor_diag_win)

        return row_wins + column_wins + diagonal_wins

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
            Return ``True`` if the state is terminal or ``False`` if it
            is not.
        """
        p1_state, p2_state, _ = state
        # check all squares have been played on (i.e. the binary
        # representation is all 1s)
        if (p1_state | p2_state) == 2 ** (self.rows * self.columns) - 1:
            return True
        # check if player 1 has won
        if any(p1_state & win == win for win in self._win_bitmasks):
            return True
        # check if player 2 has won
        if any(p2_state & win == win for win in self._win_bitmasks):
            return True
        # otherwise it is non-terminal
        return False

    def current_player(self, state: GameState) -> int:
        """Given a state, return an int indicating the player whose
        turn it is.

        Parameters
        ----------
        state:
            A 1-D array representing a noughts and crosses game state.

        Returns
        -------
        player:
            An int indicating the player whose turn it is, where 1 and 2
            correspond to player 1 and player 2, respectively.
        """
        return state[2]

    def utility(self, state: GameState) -> Dict[int, int]:
        """Given a terminal noughts and crosses state, calculates the
          outcomes for both players. These outcomes are given by +1, -1
          and 0 for a win, loss, or draw, respectively.

          Parameters
          ---------
          state:
              A 1-D array representing a terminal noughts and crosses
              game state, corresponding to either a win or a draw.

          Returns
          -------
          outcome:
              The outcome of the terminal state for both players,
              represented as a dictionary with keys ints indicating the
              players and values each players respective utility.

          Raises
          ------
          ValueError:
              If the input state is a non-terminal state.
          """

        if not self.is_terminal(state):
            raise ValueError("Utility can not be calculated for a "
                             "non-terminal state.")

        p1_state, p2_state, _ = state
        # check if player 1 has won
        if any(p1_state & win == win for win in self._win_bitmasks):
            return {1: 1, 2: -1}
        # check if player 2 has won
        if any(p2_state & win == win for win in self._win_bitmasks):
            return {1: -1, 2: 1}
        # otherwise it is a draw
        return {1: 0, 2: 0}

    def legal_actions(self, state: GameState) -> Dict[Action, GameState]:
        """Given a non-terminal state, generate a dictionary mapping
        legal actions onto their resulting game states.

         Actions are indicated by the coordinate of the square where the
         player could place a 'x' or 'o', e.g. (0, 2) for the square in
         the 3rd column of the first row (0-based indexing).

         Parameters
         ----------
         state
             A 1-D array representing a non-terminal noughts and crosses
             game state.

         Returns
         -------
         dict
             A dictionary mapping all possible legal actions from the
             input game state to the corresponding game states resulting
             from taking each action.
         """  # TODO: could split this up so it just returned actions?
        if self.is_terminal(state):
            raise ValueError("Legal actions can not be computed for a "
                             "terminal state.")

        p1_state, p2_state, current_player = state
        occupied_squares = p1_state | p2_state

        actions = [action
                   for action, action_binary in self._actions_to_binary.items()
                   if not occupied_squares & action_binary]

        return {action: self._next_state(state, action) for action in actions}

    def _next_state(self, state: GameState, action: Action) -> GameState:
        """Given a state and a legal action, return the state resulting
        from taking the action.

        Parameters
        ----------
        state


        """
        *player_states, current_player = state
        current_player_idx = current_player - 1
        player_states[current_player_idx] |= self._actions_to_binary[action]
        next_player = (current_player % 2) + 1
        return tuple(player_states) + (next_player,)

    def display(self, state: GameState) -> None:
        """Display the noughts and crosses state in a 2-D ASCII grid.

        Parameters
        ---------
        state:
            A tuple representing a noughts and crosses grid to be
            printed to stdout.
        """

        p1_state, p2_state, _ = state
        # construct a dictionary mapping board positions to symbols
        board_dict = {}
        for row in range(self.rows):
            for col in range(self.columns):
                if p1_state & self._actions_to_binary[(row, col)]:
                    board_dict[(row, col)] = "x"
                elif p2_state & self._actions_to_binary[(row, col)]:
                    board_dict[(row, col)] = "o"
                else:
                    board_dict[(row, col)] = " "

        divider = "\n" + "+".join(["---"] * self.columns) + "\n"

        # construct the string for each row
        row_strings = []
        for row in range(self.rows):
            row_string = "|".join([" {} ".format(board_dict[(row, col)])
                                   for col in range(self.columns)])
            row_strings.append(row_string)

        ascii_grid = divider.join(row_strings)
        print(ascii_grid)

    def __repr__(self):
        return "{0}({1}, {2})".format(self.__class__.__name__,
                                      self.rows, self.columns)


class UltimateAction(NamedTuple):
    sub_board: tuple
    sub_action: tuple


class UltimateGameState(NamedTuple):
    last_sub_action: tuple
    board: tuple


class UltimateNoughtsAndCrosses:
    """A class to represent the game of ultimate noughts and crosses
    (or tic-tac-toe)."""

    def __init__(self) -> None:
        self.initial_state = UltimateGameState((0, 0), (0,) * 81)
        self.sub_game = NoughtsAndCrosses()
        self.action_space = tuple(
            UltimateAction(sub_board, sub_action)
            for sub_board in itertools.product(range(3), range(3))
            for sub_action in itertools.product(range(3), range(3)))
        self.action_indices = {action: self._action_to_index(action)
                               for action in self.action_space}
        self.index_to_action = {index: action for action, index
                                in self.action_indices.items()}

    @staticmethod
    def _action_to_index(action: UltimateAction) -> int:
        sub_board_row, sub_board_col = action.sub_board
        sub_row, sub_col = action.sub_action
        return sub_board_row * 27 + sub_board_col * 3 + sub_row * 9 + sub_col

    def _compute_meta_board(self, state: UltimateGameState) -> Tuple[int, ...]:
        board = np.array(state.board).reshape(9, 9)
        sub_boards = [board[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                      for i in range(3) for j in range(3)]
        meta_board = []
        for sub_board in sub_boards:
            sub_board_state = tuple(sub_board.ravel())
            try:
                utility = self.sub_game.utility(sub_board_state)
            except ValueError:
                symbol = 0
            else:
                sub_board_winner = max(utility, key=utility.get)
                symbol = 1 if sub_board_winner == 1 else -1

            meta_board.append(symbol)
        return tuple(meta_board)

    def is_terminal(self, state: UltimateGameState) -> bool:
        meta_board = self._compute_meta_board(state)
        return self.sub_game.is_terminal(meta_board)

    def utility(self, state: UltimateGameState) -> Dict[int, int]:
        meta_board = self._compute_meta_board(state)
        return self.sub_game.utility(meta_board)

    def current_player(self, state: UltimateGameState) -> int:
        return self.sub_game.current_player(state.board)

    def compute_next_states(self, state: UltimateGameState
                            ) -> Dict[UltimateAction, UltimateGameState]:
        if self.is_terminal(state):
            raise ValueError("Next states can not be generated for a "
                             "terminal state.")
        board = np.array(state.board).reshape(9, 9)
        sub_board_row, sub_board_col = state.last_sub_action
        sub_board = board[sub_board_row * 3:(sub_board_row + 1) * 3,
                    sub_board_col * 3:(sub_board_col + 1) * 3]
        sub_board_state = tuple(sub_board.ravel())

        player_symbol = 1 if self.which_player(state) == 1 else -1

        next_states = {}
        if state == self.initial_state or self.sub_game.is_terminal(sub_board_state):
            [available_action_indices] = np.where(np.asarray(state.board) == 0)
            for action_index in available_action_indices:
                action = self.index_to_action[action_index]
                next_board = list(state.board)
                next_board[action_index] = player_symbol
                next_state = UltimateGameState(
                    last_sub_action=action.sub_action, board=tuple(next_board))
                next_states[action] = next_state
            return next_states
        else:
            available_sub_actions = tuple(zip(*np.where(sub_board == 0)))
            for sub_action in available_sub_actions:
                action = UltimateAction(sub_board=(sub_board_row, sub_board_col),
                                        sub_action=sub_action)
                next_board = list(state.board)
                next_board[self.action_indices[action]] = player_symbol
                next_state = UltimateGameState(
                    last_sub_action=action.sub_action, board=tuple(next_board))
                next_states[action] = next_state
            return next_states

    def display(self, state: UltimateGameState) -> None:
        game = NoughtsAndCrosses(rows=9, columns=9)
        game.display(state.board)
