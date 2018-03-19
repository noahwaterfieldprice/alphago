import alphago.games.noughts_and_crosses as nac
import alphago.games.connect_four as cf
from alphago import alphago
from alphago.evaluator import create_trivial_evaluator, BasicNACNet
from alphago.evaluator import BasicConnectFourNet
from alphago import comparator

if __name__ == "__main__":
    if True:
        game = nac
        net1 = BasicNACNet()
    else:
        game = cf
        net1 = BasicConnectFourNet()

    evaluator1 = net1.create_evaluator(game.ACTION_INDICES)
    evaluator2 = create_trivial_evaluator(game.compute_next_states)

    mcts_iters = 100

    for i in range(50):
        # Train net 1, then compare against net 2.
        print("Training net 1")
        alphago.alphago(evaluator1, net1.train, game.ACTION_INDICES, game,
                        self_play_iters=100, mcts_iters=mcts_iters,
                        c_puct=1.0)

        print("We are player 1.")
        comparator.compare(game, evaluator1, evaluator2,
                           mcts_iters=mcts_iters, num_games=50)
        print("We are player 2.")
        comparator.compare(game, evaluator2, evaluator1,
                           mcts_iters=mcts_iters, num_games=50)
