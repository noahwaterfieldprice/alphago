import numpy as np
import pytest

from alphago.player import MCTSPlayer, RandomPlayer
from alphago.games import noughts_and_crosses as nac
from alphago.evaluator import create_trivial_evaluator


def test_random_noughts_and_crosses_player_gives_equal_action_probabilities():
    player = RandomPlayer(game=nac, player_no=1)
    action_probs = player.action_probabilities(nac.INITIAL_STATE)

    next_states = nac.compute_next_states(nac.INITIAL_STATE)
    expected_action_probs = {action: 1 / len(next_states)
                             for action in next_states.keys()}

    for action in expected_action_probs.keys():
        np.testing.assert_almost_equal(action_probs[action],
                                       expected_action_probs[action])


@pytest.mark.parametrize("state, optimal_action", [
    (nac.INITIAL_STATE, (0, 0)),
    ((1, 1, np.nan, np.nan, np.nan, np.nan, -1, -1, np.nan), (0, 2)),
    ((1, -1, 1, 1, np.nan, -1, np.nan, 1, np.nan), (2, 0)),

])
def test_mcts_noughts_and_crosses_player_gives_optimal_moves(state, optimal_action):
    evaluator = create_trivial_evaluator(nac.compute_next_states)
    player = MCTSPlayer(game=nac, player_no=1, evaluator=evaluator,
                        mcts_iters=100, c_puct=0.5)
    action_probs = player.action_probabilities(state)

    assert optimal_action == max(action_probs, key=action_probs.get)
