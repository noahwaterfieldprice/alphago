import numpy as np

from alphago.games import NoughtsAndCrosses
from ..unit.games.mock_game import MockGame
from alphago.estimator import create_trivial_estimator
from alphago.player import RandomPlayer, MCTSPlayer
from alphago.evaluator import evaluate, play


def test_playing_two_random_players_against_each_other():
    # Seed the random number generator.
    np.random.seed(0)
    mock_game = MockGame()

    player1 = RandomPlayer(mock_game)
    player2 = RandomPlayer(mock_game)
    players = {1: player1, 2: player2}

    # Check the players aren't equal.
    assert player1 is not player2

    actions, game_states = play(mock_game, players)

    assert mock_game.is_terminal(game_states[-1])
    assert actions == [1, 1, 1]
    assert game_states == [0, 2, 6, 17]


def test_evaluator_can_compare_two_mcts_players_with_trivial_estimator():
    # Seed the random number generator.
    np.random.seed(0)
    mock_game = MockGame()

    estimator = create_trivial_estimator(mock_game.compute_next_states)
    player1 = MCTSPlayer(mock_game, estimator, 100, 0.5)
    player2 = MCTSPlayer(mock_game, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    # Check the players aren't equal.
    assert player1 is not player2

    player1_results, _ = evaluate(mock_game, players, 100)

    assert player1_results == {1: 100, -1: 0, 0: 0}


def test_evaluator_on_noughts_and_crosses():
    # Seed the random number generator.
    np.random.seed(0)

    nac = NoughtsAndCrosses()
    estimator = create_trivial_estimator(nac.compute_next_states)
    player1 = MCTSPlayer(nac, estimator, 100, 0.5)
    player2 = MCTSPlayer(nac, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    # Check the evaluators aren't equal.
    assert player1 is not player2

    player1_results = evaluate(nac, players, 20)

    # TODO: Test something here!


# def test_evaluator_on_noughts_and_crosses_with_nets():
#     # Seed the random number generator.
#     np.random.seed(0)
#
#     # Create the nets.
#     net1 = BasicNACNet()
#     net2 = BasicNACNet()
#
#     # Create the evaluators
#     evaluator1 = net1.create_estimator(nac.ACTION_INDICES)
#     evaluator2 = net2.create_estimator(nac.ACTION_INDICES)
#
#     # Check the evaluators aren't equal.
#     assert net1 != net2
#
#     evaluator1_wins, evaluator2_wins, draws = evaluator.evaluate(
#         nac, evaluator1, evaluator2, mcts_iters=100, num_games=6)
#
#     # TODO: This doesn't seem to be deterministic.
