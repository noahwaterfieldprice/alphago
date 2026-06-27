"""Loader for the Connect Four perfect-solver training data.

The on-disk solver file (``data/connect_four_data.txt``) has one record per
line in the space-separated format ``<moves> <opt_action> <value>``:

- ``moves``: a digit string of played columns (1-indexed, columns 1-7), e.g.
  ``"7564621524117"`` for a 13-move game.
- ``opt_action``: a single optimal action, an integer in 1..7.
- ``value``: the game-theoretic outcome to the current player, in {-1, 0, 1}.

Each parsed record becomes a ``(state, probs_vector, z)`` training tuple, where
``state`` is the 42-tuple board produced by
:func:`alphago.games.connect_four.action_list_to_state`, ``probs_vector`` is a
one-hot policy target over the 7 columns, and ``z`` is the integer outcome.
"""

from pathlib import Path

import numpy as np

from .games.connect_four import action_list_to_state


def parse_solved_line(line: str) -> tuple[tuple[int, ...], list[int], int]:
    """Parses one solver-data line into a ``(state, optimal_actions, z)`` tuple.

    Args:
        line: A single record in the format ``<moves> <opt_action> <value>``,
            space-separated with exactly three fields.

    Returns:
        A tuple ``(state, optimal_actions, z)`` where ``state`` is the 42-tuple
        board, ``optimal_actions`` is a single-element list holding the optimal
        column (1-indexed, 1..7), and ``z`` is the outcome in {-1, 0, 1}.

    Raises:
        ValueError: If the line does not have exactly three fields, the moves
            field contains a non-digit character or a column outside 1..7, the
            optimal action is outside 1..7, or the value is outside {-1, 0, 1}.
    """
    fields = line.split()
    if len(fields) != 3:
        raise ValueError(
            f"Expected 3 space-separated fields, got {len(fields)}: {line!r}"
        )

    moves, opt_action, value = fields

    if not moves.isdigit():
        raise ValueError(f"moves field must be a digit string, got {moves!r}")

    opt = int(opt_action)
    if not 1 <= opt <= 7:
        raise ValueError(f"opt_action must be in 1..7, got {opt}")

    z = int(value)
    if z not in (-1, 0, 1):
        raise ValueError(f"value must be in {{-1, 0, 1}}, got {z}")

    cols = [int(c) for c in moves]
    if any(not 1 <= c <= 7 for c in cols):
        raise ValueError(f"moves must use columns 1..7, got {moves!r}")
    state = action_list_to_state([c - 1 for c in cols])
    optimal_actions = [opt]

    return state, optimal_actions, z


def load_solved_states(
    path: str | Path, max_lines: int | None = None
) -> list[tuple[tuple[int, ...], np.ndarray, int]]:
    """Streams the solver file into ``(state, probs_vector, z)`` training tuples.

    The file is read line-by-line (never fully loaded into memory) so the
    multi-million-line, 95 MB solver file can be processed within a bounded
    memory budget. Blank or whitespace-only lines are skipped.

    Args:
        path: Path to the solver-data file.
        max_lines: If given, stop after this many parsed (non-blank) lines.

    Returns:
        A list of ``(state, probs_vector, z)`` tuples, where ``state`` is the
        42-tuple board, ``probs_vector`` is a one-hot policy target over the 7
        columns (summing to 1), and ``z`` is the integer outcome.
    """
    training_data: list[tuple[tuple[int, ...], np.ndarray, int]] = []

    with open(path) as f:
        for line in f:
            if max_lines is not None and len(training_data) >= max_lines:
                break
            if not line.strip():
                continue

            state, optimal_actions, z = parse_solved_line(line)
            probs_vector = np.array(
                [
                    1 / len(optimal_actions) if a + 1 in optimal_actions else 0
                    for a in range(7)
                ]
            )
            training_data.append((state, probs_vector, z))

    return training_data
