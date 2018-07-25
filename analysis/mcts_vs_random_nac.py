from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer
from alphago.games import NoughtsAndCrosses, connect_four
from alphago.utilities import memoize_instance
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate, play

nac = NoughtsAndCrosses(3, 6)
# memoize_instance(nac)

trivial_estimator = create_trivial_estimator(nac)
player2 = MCTSPlayer(nac, trivial_estimator, 30, 0.5, 0.01)
player1 = MCTSPlayer(nac, trivial_estimator, 30, 0.5, 0.01)

evaluate(nac, {2: player2, 1: player1}, 1000)
