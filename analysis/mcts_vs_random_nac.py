from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer
from alphago.games import noughts_and_crosses, connect_four
from alphago.estimator import create_trivial_estimator
from alphago.evaluator import evaluate
from alphago.alphago import evaluate_mcts_against_random_player

game = noughts_and_crosses

trivial_estimator = create_trivial_estimator(game.compute_next_states)
mcts_iters = 1000
c_puct = 0.5
num_games = 200

# mcts_player = MCTSPlayer(2, game, trivial_estimator,
#                          mcts_iters, c_puct, tau=0.1)
# random_player = RandomPlayer(1, game)
#
# player1_results, game_logs = evaluate(game, {2: mcts_player,
#                                              1: random_player}, num_games)
#
# for game_log in game_logs:
#     if game_log.result == 1:
#         print(game_log.actions)
#         print("="*30)


mcts_iters_list = [10, 50, 100, 500, 1000]
tau_list = [0.01, 0.1, 1.0]
wins1_dict = {}
wins2_dict = {}
draws_dict = {}
for tau in tau_list:
    for mcts_iters in mcts_iters_list:
        print("Evaluating with mcts iters: {}, tau: {}".format(mcts_iters,
                                                               tau))
        wins1, wins2, draws = evaluate_mcts_against_random_player(
            game, trivial_estimator, mcts_iters, c_puct, num_games, tau=tau)

        wins1_dict[(mcts_iters, tau)] = wins1
        wins2_dict[(mcts_iters, tau)] = wins2
        draws_dict[(mcts_iters, tau)] = draws

print("MCTS Iters: {}".format(mcts_iters_list))
print("MCTS wins: {}".format(wins1_dict))
print("Random wins: {}".format(wins2_dict))
print("Draws: {}".format(draws_dict))
