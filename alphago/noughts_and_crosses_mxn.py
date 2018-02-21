import numpy as np


class NoughtsAndCrosses:

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.initial_state = (np.nan,) * rows * columns

    def _calculate_line_sums(self, state):
        grid = np.array(state).reshape(self.rows, self.columns)
        horizontals = np.nansum(grid, axis=1)
        verticals = np.nansum(grid, axis=0)
        axis_difference = abs(self.rows - self.columns)
        major_diagonals = [np.nansum(grid.diagonal(offset=i))
                           for i in range(axis_difference + 1)]
        minor_diagonals = [np.nansum(np.fliplr(grid).diagonal(offset=i))
                           for i in range(axis_difference + 1)]
        diagonals = np.concatenate([major_diagonals, minor_diagonals])
        return horizontals, verticals, diagonals

    def is_terminal(self, state):
        # check all squares have been played on
        if not np.any(np.isnan(state)):
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


