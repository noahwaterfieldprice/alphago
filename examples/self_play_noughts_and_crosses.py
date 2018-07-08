import alphago.games as games
from alphago.evaluator import play, evaluate
from alphago.estimator import create_trivial_estimator, NACNetEstimator
from alphago.player import MCTSPlayer, RandomPlayer, OptimalPlayer

if __name__ == "__main__":

    max_iters = 30
    c_puct = 1.0

    nac = games.NoughtsAndCrosses(3, 6)

    estimator = create_trivial_estimator(nac)

    # nn_estimator = NACNetEstimator(learning_rate=1E-4,
    #                                l2_weight=0.01,
    #                                action_indices=nac.action_indices)

    players = {
       #1: RandomPlayer(nac),
       #2: RandomPlayer(nac),
       1: MCTSPlayer(nac, estimator, max_iters, c_puct),
       2: MCTSPlayer(nac, estimator, max_iters, c_puct),
               }

    players_switched = {1: players[2], 2: players[1]}

    # I think this doesn't work because before the nodes were explicitly
    # expanded by the MCTS algorithm before - now this is called contained
    # inside the player object so maybe they dont interact in the play
    # function?
    player1_results_a, _ = evaluate(nac, players, 1000)
    player1_results_b, _ = evaluate(nac, players_switched, 1000)
    print(player1_results_a, player1_results_b)
    # for state in game_states:
    #     nac.display(state)
    #     print("\n")
