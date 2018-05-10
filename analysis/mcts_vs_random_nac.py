from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer
from alphago.games import NoughtsAndCrosses, connect_four
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate, play


nac = NoughtsAndCrosses()
trivial_estimator = create_trivial_estimator(
    nac.compute_next_states)
player2 = MCTSPlayer(nac, trivial_estimator, 20, 0.5, 0.01)
player1 = MCTSPlayer(nac, trivial_estimator, 20, 0.5, 0.01)

evaluate(nac, {2: player2, 1: player1}, 1000)
