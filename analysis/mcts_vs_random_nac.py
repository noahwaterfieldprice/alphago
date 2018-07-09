import inspect

from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer
from alphago.games import NoughtsAndCrosses, connect_four
from alphago.utilities import memoize_instance
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate, play

import pprint

nac = NoughtsAndCrosses()
memoize_instance(nac)

trivial_estimator = create_trivial_estimator(nac)
player2 = MCTSPlayer(nac, trivial_estimator, 20, 0.5, 0.01)
player1 = MCTSPlayer(nac, trivial_estimator, 20, 0.5, 0.01)

evaluate(nac, {2: player2, 1: player1}, 1000)
