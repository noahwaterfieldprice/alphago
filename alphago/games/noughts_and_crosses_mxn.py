from typing import Dict, Tuple

import numpy as np

from .game import Game
GameState = Tuple[int, ...]
Action = Tuple[int, int]


class NoughtsAndCrosses(Game):

    def __init__(self, rows: int, columns: int) -> None:
        self.rows = rows
        self.columns = columns
        self.initial_state = (0,) * rows * columns
        self.action_space = tuple(
            (i, j) for i in range(self.rows)
            for j in range(self.columns))  # type: Tuple[Action, ...]
        self.action_indices = {
            a: self.action_space.index(a)
            for a in self.action_space}  # type: Dict[Action, int]

    def _calculate_line_sums(self, state: GameState
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return int(np.sum(np.abs(state)) % 2) + 1

    def utility(self, state: GameState) -> Dict[int, int]:
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


