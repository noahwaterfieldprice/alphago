"""Unit tests for the Connect Four solver-data loader."""

import numpy as np
import pytest

from alphago.connect_four_data import load_solved_states, parse_solved_line
from alphago.games.connect_four import action_list_to_state


def test_parse_single_move_line():
    state, optimal_actions, z = parse_solved_line("4 4 -1")

    assert state == action_list_to_state([3])
    assert optimal_actions == [4]
    assert z == -1


def test_parse_multi_move_line():
    state, optimal_actions, z = parse_solved_line("7564621524117 4 1")

    assert optimal_actions == [4]
    assert z == 1
    assert isinstance(state, tuple)
    assert len(state) == 42


def test_parse_state_matches_action_list_to_state():
    moves = "7564621524117"
    state, _, _ = parse_solved_line(f"{moves} 4 1")

    expected = action_list_to_state([int(c) - 1 for c in moves])
    assert state == expected


@pytest.mark.parametrize(
    "line",
    [
        "4 4",  # too few fields
        "4 4 -1 0",  # too many fields
        "4x 4 -1",  # non-digit moves char
        "4 0 -1",  # opt_action below 1
        "4 8 -1",  # opt_action above 7
        "4 4 2",  # value outside {-1, 0, 1}
        "4 4 -2",  # value outside {-1, 0, 1}
    ],
)
def test_malformed_line_raises_value_error(line):
    with pytest.raises(ValueError):
        parse_solved_line(line)


def test_load_solved_states_respects_max_lines(tmp_path):
    data_file = tmp_path / "slice.txt"
    data_file.write_text("1 2 1\n2 3 1\n3 3 0\n4 4 -1\n5 2 0\n")

    training_data = load_solved_states(data_file, max_lines=2)

    assert len(training_data) == 2
    state, probs_vector, z = training_data[0]
    assert isinstance(state, tuple)
    assert len(state) == 42
    assert isinstance(probs_vector, np.ndarray)
    assert len(probs_vector) == 7
    assert isinstance(z, int)


def test_load_solved_states_builds_one_hot_probs(tmp_path):
    data_file = tmp_path / "slice.txt"
    # opt_action 4 -> one-hot at index 3.
    data_file.write_text("4 4 -1\n")

    training_data = load_solved_states(data_file)

    _, probs_vector, _ = training_data[0]
    assert probs_vector[3] == 1.0
    assert probs_vector.sum() == pytest.approx(1.0)
    assert (np.delete(probs_vector, 3) == 0).all()


def test_load_solved_states_skips_blank_lines(tmp_path):
    data_file = tmp_path / "slice.txt"
    data_file.write_text("1 2 1\n\n   \n2 3 1\n")

    training_data = load_solved_states(data_file)

    assert len(training_data) == 2
