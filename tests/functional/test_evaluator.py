import numpy as np

from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate, play, run_tournament
from alphago.games import NoughtsAndCrosses
from alphago.player import MCTSPlayer, RandomPlayer

from ..unit.games.mock_game import MockGame


def test_playing_two_random_players_against_each_other():
    np.random.seed(0)
    mock_game = MockGame()

    player1 = RandomPlayer(mock_game)
    player2 = RandomPlayer(mock_game)
    players = {1: player1, 2: player2}

    # Check the players aren't equal.
    assert player1 is not player2

    actions, game_states, utility = play(mock_game, players)

    assert mock_game.is_terminal(game_states[-1])
    assert actions == [1, 1, 1]
    assert game_states == [0, 2, 6, 17]


def test_evaluator_can_compare_two_mcts_players_with_trivial_estimator():
    np.random.seed(0)
    mock_game = MockGame()

    estimator = create_trivial_estimator(mock_game)
    player1 = MCTSPlayer(mock_game, estimator, 100, 0.5)
    player2 = MCTSPlayer(mock_game, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    # Check the players aren't equal.
    assert player1 is not player2

    player1_results, _ = evaluate(mock_game, players, 100)

    assert player1_results == {1: 100, -1: 0, 0: 0}


def test_evaluator_on_noughts_and_crosses():
    np.random.seed(0)

    nac = NoughtsAndCrosses()
    estimator = create_trivial_estimator(nac)
    player1 = MCTSPlayer(nac, estimator, 100, 0.5)
    player2 = MCTSPlayer(nac, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    # Check the evaluators aren't equal.
    assert player1 is not player2

    num_games = 20
    player1_results, game_logs = evaluate(nac, players, num_games)

    # Every game must be accounted for as a win, loss or draw for player 1.
    assert set(player1_results) == {1, -1, 0}
    assert sum(player1_results.values()) == num_games
    assert len(game_logs) == num_games


def test_running_tournament_between_mcts_players():
    np.random.seed(0)
    mock_game = MockGame()

    estimator = create_trivial_estimator(mock_game)
    player1 = MCTSPlayer(mock_game, estimator, 100, 0.5)
    player2 = MCTSPlayer(mock_game, estimator, 100, 0.5)
    players = {1: player1, 2: player2}

    num_rounds = 3
    results = run_tournament(mock_game, players, num_rounds)

    # With two players there is one pairing played twice per round, so the
    # tournament plays num_rounds * 2 games; update_results credits exactly
    # 1.0 total per game (a win is 1.0, a draw is 0.5 + 0.5). Every game must
    # therefore be accounted for in the results.
    assert sum(n for _, _, n in results) == num_rounds * 2
    # Every player number appearing in the results is one of the two players.
    for i, j, _ in results:
        assert i in {1, 2}
        assert j in {1, 2}
