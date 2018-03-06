import numpy as np
import pytest

from alphago import mcts_tree
import alphago.games.noughts_and_crosses as nac
from ..unit.games.mock_game import MockGame, mock_evaluator
from alphago.evaluator import create_trivial_evaluator, BasicNACNet
from alphago import comparator


def test_comparator_can_compare_two_trivial_evaluators():
    # Seed the random number generator.
    np.random.seed(0)

    mock_game = MockGame()

    evaluator1 = create_trivial_evaluator(mock_game.compute_next_states)
    evaluator2 = create_trivial_evaluator(mock_game.compute_next_states)

    # Check the evaluators aren't equal.
    assert evaluator1 != evaluator2

    evaluator1_wins, evaluator2_wins, draws = comparator.compare(
        mock_game.compute_next_states, mock_game.initial_state,
        mock_game.utility, mock_game.which_player, mock_game.is_terminal,
        evaluator1, evaluator2, num_games=100)

    assert evaluator1_wins == 50
    assert evaluator2_wins == 50
    assert draws == 0


def test_comparator_on_noughts_and_crosses():
    # Seed the random number generator.
    np.random.seed(0)

    evaluator1 = create_trivial_evaluator(nac.compute_next_states)
    evaluator2 = create_trivial_evaluator(nac.compute_next_states)

    # Check the evaluators aren't equal.
    assert evaluator1 != evaluator2

    evaluator1_wins, evaluator2_wins, draws = comparator.compare(
        nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
        nac.which_player, nac.is_terminal, evaluator1, evaluator2,
        num_games=20)

    # Just for consistency -- there is no rationale here. Hopefully the number
    # of wins for player 1 and player 2 should be similar.
    assert evaluator1_wins == 6
    assert evaluator2_wins == 5
    assert draws == 9


def test_comparator_on_noughts_and_crosses_with_nets():
    # Seed the random number generator.
    np.random.seed(0)

    # Create the nets.
    net1 = BasicNACNet()
    net2 = BasicNACNet()

    # Create the evaluators
    evaluator1 = net1.create_evaluator(nac.ACTION_INDICES)
    evaluator2 = net2.create_evaluator(nac.ACTION_INDICES)

    # Check the evaluators aren't equal.
    assert net1 != net2

    evaluator1_wins, evaluator2_wins, draws = comparator.compare(
        nac.compute_next_states, nac.INITIAL_STATE, nac.utility,
        nac.which_player, nac.is_terminal, evaluator1, evaluator2, num_games=6)

    # TODO: This doesn't seem to be deterministic.
