import numpy as np
import pytest

from alphago.player import MCTSPlayer, RandomPlayer
from alphago.games import NoughtsAndCrosses
from alphago.estimator import create_trivial_estimator


def test_random_noughts_and_crosses_player_gives_equal_action_probabilities():
    nac = NoughtsAndCrosses()
    player = RandomPlayer(game=nac)
    action, action_probs = player.choose_action(nac.initial_state,
                                                return_probabilities=True)

    next_states = nac.compute_next_states(nac.initial_state)
    expected_action_probs = {action: 1 / len(next_states)
                             for action in next_states.keys()}

    for action in expected_action_probs.keys():
        np.testing.assert_almost_equal(action_probs[action],
                                       expected_action_probs[action])


@pytest.mark.parametrize("state, optimal_actions", [
    ((0, 0, 0, 0, 0, 0, 0, 0, 0), [(0, 0), (0, 2), (2, 0), (2, 2)]),
    ((1, 1, 0, 0, 0, 0, -1, -1, 0), [(0, 2)]),
    ((1, -1, 1, 1, 0, -1, 0, 1, 0), [(2, 0)]),
])
def test_mcts_noughts_and_crosses_player_gives_optimal_moves(state, optimal_actions):
    # seed the random number generator.
    np.random.seed(0)

    nac = NoughtsAndCrosses()
    estimator = create_trivial_estimator(nac)
    player = MCTSPlayer(game=nac, estimator=estimator, mcts_iters=100,
                        c_puct=0.5, tau=1)
    action, action_probs = player.choose_action(state, return_probabilities=True)
    print(action_probs)

    assert max(action_probs, key=action_probs.get) in optimal_actions
