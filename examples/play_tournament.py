from alphago.games import NoughtsAndCrosses, ConnectFour
from alphago.evaluator import run_tournament
from alphago.player import RandomPlayer, MCTSPlayer
from alphago.estimator import create_trivial_estimator, create_rollout_estimator
from alphago.elo import elo

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tqdm
tqdm.tqdm.monitor_interval = 0

game = ConnectFour()

trivial_estimator = create_trivial_estimator(game)
rollout_estimator_10 = create_rollout_estimator(game, 10)
rollout_estimator_100 = create_rollout_estimator(game, 100)
rollout_estimator_200 = create_rollout_estimator(game, 200)

mcts_args = 50, 0.5, 0.01
random_player = RandomPlayer(game)
trivial_mcts_player = MCTSPlayer(game, trivial_estimator, *mcts_args)
rollout_mcts_player_10 = MCTSPlayer(game, rollout_estimator_10, *mcts_args)
rollout_mcts_player_100 = MCTSPlayer(game, rollout_estimator_100, *mcts_args)
rollout_mcts_player_200 = MCTSPlayer(game, rollout_estimator_200, *mcts_args)

players = {
    1: random_player,
    2: trivial_mcts_player,
    3: rollout_mcts_player_10,
    4: rollout_mcts_player_100,
    #5: rollout_mcts_player_200
}

results_list = run_tournament(game, players, 5)
print(results_list)

results = np.zeros(shape=(len(players), len(players)))
for result in results_list:
    i, j, n = result
    results[i - 1, j - 1] = n

fig, ax = plt.subplots()
gammas = elo(results_list)
ax.text(0.4, -0.8, str(["{:d}: {:.2f}".format(player_no, gamma)
                        for player_no, gamma in sorted(gammas.items())]))
a = ax.matshow(results, cmap=plt.cm.coolwarm)
plt.colorbar(a)
fig.savefig('results.png')
