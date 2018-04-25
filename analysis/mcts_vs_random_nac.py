from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer
from alphago.games import noughts_and_crosses, connect_four
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate, play
import numpy as np

trivial_estimator = create_trivial_estimator(
    noughts_and_crosses.compute_next_states)
player2 = MCTSPlayer(noughts_and_crosses, trivial_estimator, 2, 0.5, 0.01)
player1 = OptimalPlayer(noughts_and_crosses)

play(noughts_and_crosses, {2: player2, 1: player1})

#
#
# for game_log in game_logs:
#     if game_log.result == 1:
#         print(game_log.actions)
#         print("="*30)
#
#
# from collections import Counter
#
# counter = Counter([game_log.actions[1] for game_log in game_logs])
# print(counter)
