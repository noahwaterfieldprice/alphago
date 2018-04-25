import alphago.games.noughts_and_crosses as nac
from alphago.evaluator import play
from alphago.estimator import create_trivial_estimator
from alphago.player import MCTSPlayer, RandomPlayer

if __name__ == "__main__":

    max_iters = 1000
    c_puct = 1.0

    evaluator = create_trivial_estimator(nac.compute_next_states)

    players = {2: MCTSPlayer(2, nac, evaluator, max_iters, c_puct),
               1: RandomPlayer(1, nac)}

    # I think this doesn't work because before the nodes were explicitly
    # expanded by the MCTS algorithm before - now this is called contained
    # inside the player object so maybe they dont interact in the play
    # function?
    game_states_list, _ = play(nac, players)

    for state in game_states_list:
        nac.display(state)
        print("\n")
