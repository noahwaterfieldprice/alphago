"""Implementation of Connect 4.

Seven columns, six rows. On your turn you can play any of the columns (if it is
not full).
"""
from typing import Tuple

import numpy as np

from .game import Game

GameState, Action = Tuple[int, ...], int


class ConnectFour(Game):

    def __init__(self) -> None:
        self.initial_state = (0,) * 42
        self.action_space = tuple(i for i in range(7))
        self.action_indices = {a: self.action_space.index(a) for a in
                               self.action_space}

    def current_player(self, state):
        """Returns the player to play in the current state.

        Parameters
        ----------
        state: ndarray
            A 6x7 numpy array. +1 denotes a player 1 move; -1 denotes a player 2
            move; 0 denotes a blank space.

        Returns
        -------
        player: int
            The player to play.
        """
        return (np.sum(np.abs(state)) % 2) + 1

    @staticmethod
    def _calculate_line_sums_4_by_4(state):
        """Calculates the line sums for a 4x4 grid.

        Parameters
        ----------
        grid: ndarray
            A 4x4 np array with +1 for a player 1 move, -1 for a player 2 move and
            0 for an empty space.

        Returns
        -------
        line_sums: ndarray
            An array of sums along all the possible lines on the grid.
        """
        grid = np.array(state).reshape(4, 4)
        horizontals = np.sum(grid, axis=1)
        verticals = np.sum(grid, axis=0)
        major_diagonal = np.sum(grid.diagonal()),
        minor_diagonal = np.sum(np.fliplr(grid).diagonal()),

        line_sums = np.concatenate([horizontals, verticals, major_diagonal,
                                    minor_diagonal])

        return line_sums

    @staticmethod
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
        line_sums_list = [ConnectFour._calculate_line_sums_4_by_4(subgrid)
                          for subgrid in tuple_subgrids]
        return np.concatenate(line_sums_list)

    def is_terminal(self, state):
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
        if np.all(state):
            return True

        # Check if there is a winner
        line_sums = self._calculate_line_sums(state)
        if np.any(np.abs(line_sums) == 4):
            return True

        # Otherwise it is non-terminal
        return False

    def utility(self, state):
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
        line_sums = self._calculate_line_sums(state)

        # Player 1 wins
        if np.any(line_sums == 4):
            return {1: 1, 2: -1}

        # Player 2 wins
        if np.any(line_sums == -4):
            return {1: -1, 2: 1}

        # Draw, if all squares are filled and no-one has four-in-a-row.
        if np.all(state):
            return {1: 0, 2: 0}

        # Otherwise the state is non-terminal and the utility cannot be calculated
        raise ValueError("Utility cannot be calculated for a "
                         "non-terminal state.")

    def legal_actions(self, state):
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
        player = self.current_player(state)
        marker = 1 if player == 1 else -1

        next_states = {}

        # Consider each column in turn.
        for col in range(7):
            next_state = np.array(state).reshape(6, 7)

            # If the top element in the column is full, this column is not an
            # available action.
            if next_state[0][col]:
                continue

            # Find the lowest open element in the column. Search from the
            # bottom until we find a zero.
            for row in reversed(range(6)):
                if not next_state[row][col]:
                    next_state[row][col] = marker
                    next_states[col] = tuple(next_state.ravel())
                    break

        return next_states

    @staticmethod
    def display(state):
        """Display the connect four state in a 2-D ASCII grid.

        Parameters
        ---------
        state: tuple
            A length 42 tuple representing a connect four grid to be printed to
            stdout.
        """
        divider = "\n---+---+---+---+---+---+---\n"
        symbol_dict = {1: "x", -1: "o", 0: " "}

        output_rows = []
        for state_row in np.array_split(tuple(state), indices_or_sections=6):
            y = "|". join([" {} ".format(symbol_dict[x]) for x in state_row])
            output_rows.append(y)

        ascii_grid = divider.join(output_rows)
        print(ascii_grid)


def action_list_to_state(action_list):
    """Converts a list of columns played into a game state.

    Parameters
    ----------
    action_list: list
        The list of played columns corresponding to this state. The columns are
        indexed from 0.

    Returns
    -------
    state: tuple
        A tuple representing the state.
    """
    columns = {action: [] for action in range(7)}
    for i, action in enumerate(action_list):
        player = (i % 2) + 1
        player_symbol = 2 * (i % 2) - 1
        columns[action].append(player_symbol)
    
    for action in range(7):
        n = len(columns[action])
        columns[action].extend([0 for j in range(6-n)])

    state = np.zeros((6, 7), int)
    for action in range(7):
        for i in range(6):
            state[i][action] = columns[action][5-i]

    return tuple(state.ravel())


def heuristic(state):
    """Computes a heuristic score for the first player on the given state.

    The heuristic is the number of remaining ways for player 1 to win (ignoring
    player 2's moves) minus the number of remaining ways for player 2 to win
    (ignoring player 1's moves).

    Parameters
    ----------
    state: tuple
        The Connect Four state. Length 42 tuple reading across the rows.
    
    Returns
    -------
    heuristic: int
        The number of remaining ways for player 1 to win (ignoring player 2's
        moves) minus the number of remaining ways for player 2 to win (ignoring
        player 1's moves).
    """
    # First fill the state with 1s, and count the connect fours for player 1.
    # Then fill the state with -1s and count the connect fours for player 2.
    state_1 = np.array(state)
    state_1 = np.where(state_1 == 0, 1, state_1)
    line_sums = ConnectFour._calculate_line_sums(state_1)
    num_wins_1 = np.sum(line_sums == 4)

    state_2 = np.array(state)
    state_2 = np.where(state_2 == 0, -1, state_2)
    line_sums = ConnectFour._calculate_line_sums(state_2)
    num_wins_2 = np.sum(line_sums == -4)

    return num_wins_1 - num_wins_2
