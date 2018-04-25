from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer
from alphago.games import noughts_and_crosses, connect_four
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate

trivial_estimator = create_trivial_estimator(
    noughts_and_crosses.compute_next_states)
player1 = MCTSPlayer(2, noughts_and_crosses, trivial_estimator,
                     1000, 0.5, tau=0.1)
player2 = OptimalPlayer(1, noughts_and_crosses)

player1_results, game_logs = evaluate(noughts_and_crosses,
                                      {2: player1, 1: player2}, 1000)


for game_log in game_logs:
    if game_log.result == 1:
        print(game_log.actions)
        print("="*30)


from collections import Counter

counter = Counter([game_log.actions[1] for game_log in game_logs])
print(counter)